"""Simulated vehicle: kinematic bicycle model with actuator dynamics.

This is the "plant" the software drives. Everything else in the package only
talks to it through `VehicleCommand` (what we ask for) and `VehicleState`
(what the car reports), the same boundary a real vehicle interface would have.
"""

import math
from dataclasses import dataclass

from .config import VehicleParams


@dataclass
class VehicleState:
    x: float = 0.0          # m, rear-axle position
    y: float = 0.0          # m
    yaw: float = 0.0        # rad
    speed: float = 0.0      # m/s
    accel: float = 0.0      # m/s^2, actual (after actuator lag)
    steer: float = 0.0      # rad, actual road-wheel angle

    def front_axle(self, wheelbase: float) -> tuple[float, float]:
        return (self.x + wheelbase * math.cos(self.yaw),
                self.y + wheelbase * math.sin(self.yaw))


@dataclass
class VehicleCommand:
    accel: float = 0.0      # m/s^2 requested
    steer: float = 0.0      # rad requested road-wheel angle


@dataclass
class DriverInput:
    """What the human is doing with the controls at this instant."""
    brake_pressed: bool = False
    gas_pressed: bool = False
    steering_torque: float = 0.0   # Nm applied to the wheel
    engage_button: bool = False


class Vehicle:
    def __init__(self, params: VehicleParams, state: VehicleState | None = None):
        self.p = params
        self.state = state or VehicleState()

    def step(self, cmd: VehicleCommand, dt: float) -> VehicleState:
        s, p = self.state, self.p

        # Steering actuator: rate-limited toward the requested angle.
        target_steer = _clamp(cmd.steer, -p.max_steer, p.max_steer)
        max_delta = p.max_steer_rate * dt
        s.steer += _clamp(target_steer - s.steer, -max_delta, max_delta)

        # Powertrain / brakes: first-order lag toward the requested accel.
        target_accel = _clamp(cmd.accel, -p.max_decel, p.max_accel)
        s.accel += (target_accel - s.accel) * min(1.0, dt / p.accel_lag)

        # Kinematic bicycle about the rear axle; the car can't roll backward.
        s.x += s.speed * math.cos(s.yaw) * dt
        s.y += s.speed * math.sin(s.yaw) * dt
        s.yaw = wrap_angle(s.yaw + s.speed / p.wheelbase * math.tan(s.steer) * dt)
        s.speed = max(0.0, s.speed + s.accel * dt)
        if s.speed == 0.0 and s.accel < 0.0:
            s.accel = 0.0
        return s


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
