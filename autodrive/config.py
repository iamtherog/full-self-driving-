"""Tunable parameters for the vehicle model, planner, controllers and safety layer.

Vehicle numbers are simulation parameters for a 2026 Toyota Corolla sedan (2.0 L,
CVT), not values read from a real car. Checked against published specifications
(2026 LE unless noted; tests/test_corolla.py):

    wheelbase        2.70 m    published 106 in (2.69 m)
    length           4.63 m    published 182.5 in (4.64 m)
    width            1.78 m    published 70.1 in (1.78 m)
    mass             1415 kg   published curb weight 2,955 lb (1,340 kg) + 75 kg driver
    max_steer        0.60 rad  gives a 35.7-36.3 ft turning circle; published 36 ft
    max_accel        3.0 m/s^2 gives 0-60 mph in 9.2 s; published tests 7.8-8.9 s,
                               so the simulated car is slightly less capable

Not verified against a published source, and chosen conservatively: max_decel
(0.82 g), friction (0.9), cornering stiffnesses, yaw_inertia and cg_to_front.
steer_ratio is informational only; nothing uses it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VehicleParams:
    wheelbase: float = 2.70            # m
    length: float = 4.63               # m, bumper to bumper
    front_overhang: float = 0.93       # m, front axle to front bumper
    width: float = 1.78                # m
    steer_ratio: float = 13.9          # steering-wheel angle / road-wheel angle
    max_steer: float = 0.60            # rad, road-wheel angle
    max_steer_rate: float = 0.50       # rad/s, road-wheel angle
    accel_lag: float = 0.30            # s, first-order powertrain/brake lag
    max_accel: float = 3.0             # m/s^2, physical capability
    max_decel: float = 8.0             # m/s^2, physical capability (~0.8 g)
    # Plant model. "kinematic" (default) assumes the tires never slip, which is
    # accurate at the moderate lateral accelerations the planner allows.
    # "dynamic" adds tire slip and saturation; see vehicle.py.
    model: str = "kinematic"
    # Dynamic model only. Representative values for a C-segment sedan, not
    # measured on a Corolla.
    mass: float = 1415.0               # kg: 2026 LE curb weight (1,340 kg) + 75 kg driver
    yaw_inertia: float = 2200.0        # kg m^2
    cg_to_front: float = 1.08          # m, lf; lr = 2.70 - 1.08 = 1.62, so 60 % of weight is on the front
    cornering_stiffness_front: float = 80_000.0   # N/rad, whole axle
    cornering_stiffness_rear: float = 90_000.0    # N/rad, whole axle
    friction: float = 0.9              # dry asphalt
    blend_speeds: tuple[float, float] = (2.0, 4.0)  # m/s, kinematic below the first, dynamic above the second


@dataclass(frozen=True)
class PlannerParams:
    lookahead_time: float = 8.0        # s, preview horizon for speed planning
    speed_gain: float = 0.8            # 1/s, accel per m/s of speed error
    max_speed_decel: float = 2.5       # m/s^2, cap when slowing for a speed cap
    max_lat_accel: float = 2.0         # m/s^2, comfort limit in curves
    comfort_decel: float = 1.5         # m/s^2, for speed-limit / curve slowdowns
    # Intelligent Driver Model (car following / stopping)
    idm_max_accel: float = 1.5         # m/s^2
    idm_comfort_decel: float = 2.0     # m/s^2
    idm_time_headway: float = 1.8      # s
    idm_min_gap: float = 4.0           # m, standstill gap
    # Traffic lights
    yellow_max_decel: float = 3.0      # m/s^2, stop for yellow only if this suffices
    red_max_decel: float = 4.5         # m/s^2, beyond this we're already committed
    stop_line_margin: float = 1.5      # m, stop this far before the line
    # Time before commanded braking is fully established: powertrain lag (0.3 s)
    # plus the jerk-limited ramp to the yellow decel (3.0 / (2 * 5.0) = 0.3 s).
    # The stop/go decision must allow for the distance covered meanwhile.
    brake_buildup: float = 0.6         # s
    commit_min_speed: float = 3.0      # m/s, below this the car can always stop; no commitment to go


@dataclass(frozen=True)
class ControllerParams:
    # Lateral (Stanley + curvature feedforward)
    stanley_gain: float = 1.2
    stanley_soft_speed: float = 2.0    # m/s, avoids division blow-up at low speed
    heading_gain: float = 1.0
    curvature_preview: float = 0.3     # s, look ahead to cover steering lag
    # Steady-state understeer compensation in the feedforward, rad per m/s^2 of
    # lateral acceleration. 0 for the kinematic plant, which cannot understeer;
    # use_dynamic_vehicle() sets it to the dynamic plant's value.
    understeer_gradient: float = 0.0
    # Steady-state rear-axle sideslip per m/s^2 of lateral acceleration. A car
    # whose rear tires slip holds a curve with its nose turned in by this angle,
    # which the heading term must expect rather than fight. 0 on the kinematic plant.
    rear_slip_gradient: float = 0.0
    # Longitudinal (feedforward + PI on acceleration error)
    accel_kp: float = 0.4
    accel_ki: float = 0.8
    integral_limit: float = 1.0        # m/s^2
    max_jerk: float = 5.0              # m/s^3, comfort


@dataclass(frozen=True)
class SafetyLimits:
    # Command envelope (mirrors the conservative limits used by production ADAS)
    max_accel_cmd: float = 2.0         # m/s^2
    max_decel_cmd: float = 3.5         # m/s^2, normal driving
    aeb_decel: float = 7.0             # m/s^2, automatic emergency braking only
    aeb_ttc: float = 1.6               # s, time-to-collision that triggers AEB
    max_lat_accel_cmd: float = 3.0     # m/s^2, caps steering angle vs speed
    max_steer_rate_cmd: float = 0.35   # rad/s, road-wheel angle
    max_jerk_cmd: float = 5.0          # m/s^3, accel rate limit (AEB exempt)
    # Driver interaction
    driver_torque_override: float = 1.5  # Nm on the wheel -> lateral override
    # Health monitoring
    sensor_timeout: float = 0.25       # s without fresh perception -> fault
    max_lateral_error: float = 1.2     # m off lane center -> fault
    fault_decel: float = 2.0           # m/s^2, controlled slowdown after a fault


@dataclass(frozen=True)
class SensorParams:
    rate_hz: float = 20.0
    radar_range: float = 150.0         # m
    radar_gap_noise: float = 0.25      # m (1 sigma)
    radar_speed_noise: float = 0.15    # m/s (1 sigma)
    camera_lateral_noise: float = 0.03  # m (1 sigma)
    camera_heading_noise: float = 0.004  # rad (1 sigma)
    light_detect_range: float = 90.0   # m


@dataclass(frozen=True)
class Config:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    sensors: SensorParams = field(default_factory=SensorParams)
    control_rate_hz: float = 100.0


def understeer_gradient(v: VehicleParams) -> float:
    """K = m / L * (lr / Cf - lf / Cr), rad per m/s^2 (Rajamani 2012, section 3.3).

    Positive means understeer: holding a curve of curvature k at speed v needs
    ``L k + K v^2 k`` of road-wheel angle instead of the kinematic ``L k``.
    """

    lr = v.wheelbase - v.cg_to_front
    return v.mass / v.wheelbase * (lr / v.cornering_stiffness_front
                                   - v.cg_to_front / v.cornering_stiffness_rear)


def rear_slip_gradient(v: VehicleParams) -> float:
    """Rear slip angle per m/s^2 in a steady turn: m lf / (L Cr), rad per m/s^2.

    The rear axle carries ``m a_y lf / L`` of lateral force, and a linear tire
    needs slip angle ``F / Cr`` to produce it.
    """

    return v.mass * v.cg_to_front / (v.wheelbase * v.cornering_stiffness_rear)


def use_dynamic_vehicle(config: Config) -> Config:
    """The same configuration on the tire-slip plant, with the controller's
    feedforward matched to that plant's understeer gradient."""

    from dataclasses import replace
    vehicle = replace(config.vehicle, model="dynamic")
    controller = replace(config.controller, understeer_gradient=understeer_gradient(vehicle),
                         rear_slip_gradient=rear_slip_gradient(vehicle))
    return replace(config, vehicle=vehicle, controller=controller)
