"""Simulated vehicle: bicycle model with actuator dynamics.

This is the "plant" the software drives. Everything else in the package only
talks to it through `VehicleCommand` (what we ask for) and `VehicleState`
(what the car reports).

Two plant models share the same actuators:

* ``kinematic`` (default): the wheels roll exactly where they point. Accurate
  while lateral acceleration stays well below the friction limit; Polack et al.
  put the boundary near 0.5 g.
* ``dynamic``: a single-track (bicycle) model with lateral tire slip, following
  Rajamani, *Vehicle Dynamics and Control* (2nd ed., 2012), section 2.3. Tire
  forces come from the Fiala brush model (Pacejka, *Tire and Vehicle Dynamics*,
  3rd ed., 2012, chapter 3), which is linear for small slip angles and
  saturates at the friction limit. The car understeers, can run wide when
  asked for more grip than the road has, and never produces more than
  ``friction * g`` of lateral acceleration.

References:
    P. Polack, F. Altche, B. d'Andrea-Novel, A. de La Fortelle. The kinematic
    bicycle model: a consistent model for planning feasible trajectories for
    autonomous vehicles? IEEE Intelligent Vehicles Symposium, 2017.
    J. Kong, M. Pfeiffer, G. Schildbach, F. Borrelli. Kinematic and dynamic
    vehicle models for autonomous driving control design. IEEE Intelligent
    Vehicles Symposium, 2015.
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
    lateral_velocity: float = 0.0   # m/s at the center of gravity, body frame (dynamic model)
    yaw_rate: float = 0.0           # rad/s, as a yaw-rate gyro would report it
    lateral_accel: float = 0.0      # m/s^2, as a lateral accelerometer would report it

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

        if p.model == "kinematic":
            _kinematic(s, p, dt)
        elif p.model == "dynamic":
            for _ in range(DYNAMIC_SUBSTEPS):
                _dynamic(s, p, dt / DYNAMIC_SUBSTEPS)
        else:
            raise ValueError(f"unknown vehicle model {p.model!r}")
        if s.speed == 0.0 and s.accel < 0.0:
            s.accel = 0.0
        return s


#: The yaw mode of the dynamic model has a time constant of tens of
#: milliseconds at low speed; four substeps keep explicit Euler well inside
#: its stability region at the 100 Hz control rate.
DYNAMIC_SUBSTEPS = 4
GRAVITY = 9.81


def _kinematic(s: VehicleState, p: VehicleParams, dt: float) -> None:
    """Kinematic bicycle about the rear axle; the car can't roll backward."""

    s.x += s.speed * math.cos(s.yaw) * dt
    s.y += s.speed * math.sin(s.yaw) * dt
    s.yaw = wrap_angle(s.yaw + s.speed / p.wheelbase * math.tan(s.steer) * dt)
    s.speed = max(0.0, s.speed + s.accel * dt)
    s.yaw_rate = s.speed * math.tan(s.steer) / p.wheelbase   # what a yaw-rate gyro would report
    s.lateral_accel = s.speed * s.yaw_rate


def fiala(slip_angle: float, stiffness: float, peak: float) -> float:
    """Lateral tire force (N) of the Fiala brush model for one axle.

    Linear (``-C alpha``) for small slip, a cubic roll-off, and full sliding
    at ``peak = mu * Fz`` once ``|tan alpha| >= 3 * peak / C``. Continuous with
    a continuous first derivative at the sliding point.
    """

    t = math.tan(slip_angle)
    sliding = 3.0 * peak / stiffness
    if abs(t) >= sliding:
        return -math.copysign(peak, t)
    return (-stiffness * t
            + stiffness ** 2 / (3.0 * peak) * abs(t) * t
            - stiffness ** 3 / (27.0 * peak ** 2) * t ** 3)


def _dynamic(s: VehicleState, p: VehicleParams, dt: float) -> None:
    """One explicit-Euler step of the single-track model with Fiala tires.

    States: ``speed`` is the longitudinal velocity ``vx`` and
    ``lateral_velocity`` the lateral velocity ``vy``, both at the center of
    gravity in the body frame; ``yaw_rate`` is ``r``. Position is still
    reported at the rear axle, like the kinematic model, so everything
    downstream is unchanged.

    Below ``blend_speeds`` the tire equations become ill-conditioned (slip
    angles are ratios of small velocities), so the lateral states are blended
    toward the kinematic solution: fully kinematic below the first speed,
    fully dynamic above the second.
    """

    lf = p.cg_to_front
    lr = p.wheelbase - lf
    vx, vy, r, delta = s.speed, s.lateral_velocity, s.yaw_rate, s.steer

    # Static axle loads.
    fzf = p.mass * GRAVITY * lr / p.wheelbase
    fzr = p.mass * GRAVITY * lf / p.wheelbase

    if vx > 0.1:
        alpha_f = math.atan2(vy + lf * r, vx) - delta
        alpha_r = math.atan2(vy - lr * r, vx)
        fyf = fiala(alpha_f, p.cornering_stiffness_front, p.friction * fzf)
        fyr = fiala(alpha_r, p.cornering_stiffness_rear, p.friction * fzr)
        ay_dyn = (fyf * math.cos(delta) + fyr) / p.mass      # |ay_dyn| <= mu g by the tire limits
        vy_dyn = vy + (ay_dyn - vx * r) * dt
        r_dyn = r + (lf * fyf * math.cos(delta) - lr * fyr) / p.yaw_inertia * dt
    else:
        ay_dyn, vy_dyn, r_dyn = 0.0, 0.0, 0.0

    # Kinematic solution for the same speed and steer (rear axle does not slip).
    r_kin = vx * math.tan(delta) / p.wheelbase
    vy_kin = lr * r_kin
    low, high = p.blend_speeds
    w = min(1.0, max(0.0, (vx - low) / (high - low)))
    vy_new = w * vy_dyn + (1.0 - w) * vy_kin
    r_new = w * r_dyn + (1.0 - w) * r_kin

    # Rear-axle velocity in the body frame is (vx, vy - lr r).
    c, sn = math.cos(s.yaw), math.sin(s.yaw)
    vy_rear = vy - lr * r
    s.x += (vx * c - vy_rear * sn) * dt
    s.y += (vx * sn + vy_rear * c) * dt
    s.yaw = wrap_angle(s.yaw + r * dt)
    # Longitudinal: the actuator's acceleration is taken as dvx/dt. The exact
    # body-frame equation adds vy*r and the drag of the steered front tire,
    # -Fyf*sin(delta)/m, which cancel in a steady turn; keeping one without the
    # other would make a car circling at constant throttle speed up.
    s.speed = max(0.0, vx + s.accel * dt)
    s.lateral_velocity, s.yaw_rate = vy_new, r_new
    s.lateral_accel = w * ay_dyn + (1.0 - w) * vx * r_kin


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
