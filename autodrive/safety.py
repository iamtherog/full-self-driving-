"""Safety supervisor: the last layer between the software and the actuators.

Everything upstream (perception, planning, control) is allowed to be wrong.
This layer enforces a hard command envelope, hands control back to the driver
the moment they intervene, detects faults, and runs automatic emergency
braking independently of whether the system is engaged.
"""

import math
from dataclasses import dataclass
from enum import Enum

from .config import SafetyLimits, VehicleParams
from .perception import PerceptionFrame, time_to_collision
from .vehicle import DriverInput, VehicleCommand, VehicleState


class Mode(str, Enum):
    OFF = "off"                    # driver drives
    ENGAGED = "engaged"            # system steers and controls speed
    LAT_OVERRIDE = "lat_override"  # driver is steering; system keeps speed control
    FAULT = "fault"                # controlled slowdown, driver must take over


@dataclass(frozen=True)
class SafetyOutput:
    command: VehicleCommand
    controls_lateral: bool
    controls_longitudinal: bool
    mode: Mode
    aeb_active: bool
    alert: str | None


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits, vehicle: VehicleParams):
        self.lim, self.vp = limits, vehicle
        self.mode = Mode.OFF
        self.fault_reason: str | None = None
        self._last_steer = 0.0
        self._last_accel = 0.0

    def update(self, t: float, frame: PerceptionFrame | None, driver: DriverInput,
               requested: VehicleCommand, ego: VehicleState, dt: float) -> SafetyOutput:
        lim = self.lim
        self._update_mode(t, frame, driver, ego)

        # --- Automatic emergency braking: always on, even when disengaged.
        fresh = frame is not None and t - frame.t <= lim.sensor_timeout
        ttc = time_to_collision(frame.lead) if fresh and frame is not None else math.inf
        aeb = ttc < lim.aeb_ttc and ego.speed > 0.5

        accel, steer = requested.accel, requested.steer
        controls_lat = self.mode == Mode.ENGAGED
        controls_long = self.mode in (Mode.ENGAGED, Mode.LAT_OVERRIDE, Mode.FAULT)
        alert = None

        if self.mode == Mode.FAULT:
            # Slow down while steering on the dead-reckoned lane estimate.
            accel = -lim.fault_decel
            controls_lat = True
            alert = f"TAKE CONTROL: {self.fault_reason}"

        # --- Command envelope. Accel is also rate-limited here so that no
        # upstream transition (engage, fault entry, a controller reset) can
        # produce a step in the command.
        accel = _clamp(accel, -lim.max_decel_cmd, lim.max_accel_cmd)
        step = lim.max_jerk_cmd * dt
        accel = _clamp(accel, self._last_accel - step, self._last_accel + step)
        steer = self._limit_steer(steer, ego.speed, dt)

        if aeb:
            accel = -lim.aeb_decel
            controls_long = True
            alert = alert or "EMERGENCY BRAKING"

        if controls_lat:
            self._last_steer = steer
        else:
            self._last_steer = ego.steer  # track the driver for a smooth re-engage
        # AEB may step the brakes on; afterwards, ramp back from where it left off.
        self._last_accel = accel if controls_long else ego.accel

        return SafetyOutput(VehicleCommand(accel, steer), controls_lat, controls_long,
                            self.mode, aeb, alert)

    def _update_mode(self, t: float, frame: PerceptionFrame | None,
                     driver: DriverInput, ego: VehicleState) -> None:
        lim = self.lim
        fault = self._check_faults(t, frame)

        if self.mode == Mode.OFF:
            if driver.engage_button and fault is None and not driver.brake_pressed:
                self.mode = Mode.ENGAGED
            return

        if driver.brake_pressed:
            # The driver always wins. Braking cancels the system outright.
            self.mode, self.fault_reason = Mode.OFF, None
            return

        if self.mode == Mode.FAULT:
            if ego.speed < 0.1:
                self.mode = Mode.OFF
            return

        if fault is not None:
            self.mode, self.fault_reason = Mode.FAULT, fault
            return

        steering = abs(driver.steering_torque) > lim.driver_torque_override
        self.mode = Mode.LAT_OVERRIDE if steering else Mode.ENGAGED

    def _check_faults(self, t: float, frame: PerceptionFrame | None) -> str | None:
        if frame is None or t - frame.t > self.lim.sensor_timeout:
            return "perception timeout"
        if abs(frame.lane.lateral) > self.lim.max_lateral_error:
            return "lane departure"
        return None

    def _limit_steer(self, steer: float, speed: float, dt: float) -> float:
        lim, L = self.lim, self.vp.wheelbase
        # Cap the angle so lateral acceleration v^2 * tan(delta) / L stays in bounds.
        max_angle = self.vp.max_steer
        if speed > 1.0:
            max_angle = min(max_angle, math.atan(lim.max_lat_accel_cmd * L / speed ** 2))
        steer = _clamp(steer, -max_angle, max_angle)
        step = lim.max_steer_rate_cmd * dt
        return _clamp(steer, self._last_steer - step, self._last_steer + step)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
