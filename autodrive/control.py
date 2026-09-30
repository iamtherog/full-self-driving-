"""Low-level controllers: turn the plan into steering and acceleration requests."""

import math

from .config import ControllerParams, VehicleParams
from .perception import LaneEstimate


class LateralController:
    """Stanley path tracking with curvature feedforward.

    Stanley acts on the front-axle cross-track error; the feedforward term
    supplies the steady-state angle a curve needs so the feedback only has
    to correct errors.
    """

    def __init__(self, params: ControllerParams, vehicle: VehicleParams):
        self.p, self.vp = params, vehicle

    def update(self, lane: LaneEstimate, curvature: float, speed: float) -> float:
        p, L = self.p, self.vp.wheelbase
        # Lane estimate is at the rear axle; move it to the front axle.
        front_lateral = lane.lateral + L * math.sin(lane.heading_error)
        # Kinematic angle for the curve, plus the extra angle an understeering
        # car needs at this speed (zero on the kinematic plant).
        feedforward = math.atan(L * curvature) + p.understeer_gradient * speed ** 2 * curvature
        # With rear tire slip the car holds a curve nose-in; expect that angle.
        expected_heading = p.rear_slip_gradient * speed ** 2 * curvature
        heading = -p.heading_gain * (lane.heading_error - expected_heading)
        cross_track = -math.atan2(p.stanley_gain * front_lateral, p.stanley_soft_speed + speed)
        return feedforward + heading + cross_track


class LongitudinalController:
    """Feedforward on desired accel, plus PI on the accel error to reject
    disturbances (grade, drag, actuator lag). Output is jerk-limited."""

    def __init__(self, params: ControllerParams):
        self.p = params
        self.integral = 0.0
        self.last_cmd = 0.0

    def update(self, target_accel: float, measured_accel: float, dt: float,
               stopped: bool = False) -> float:
        p = self.p
        error = target_accel - measured_accel
        if stopped and target_accel <= 0.0:
            # Holding at a standstill: don't wind up against the brakes.
            self.integral = 0.0
        else:
            self.integral = _clamp(self.integral + p.accel_ki * error * dt,
                                   -p.integral_limit, p.integral_limit)
        cmd = target_accel + p.accel_kp * error + self.integral
        step = p.max_jerk * dt
        cmd = _clamp(cmd, self.last_cmd - step, self.last_cmd + step)
        self.last_cmd = cmd
        return cmd

    def reset(self, current_accel: float = 0.0) -> None:
        self.integral = 0.0
        self.last_cmd = current_accel


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
