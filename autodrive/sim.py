"""Closed-loop simulation: world -> sensors -> perception -> planner ->
controllers -> safety supervisor -> vehicle, at realistic rates."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .control import LateralController, LongitudinalController
from .observe import Advisory, Observer, Snapshot
from .perception import (
    LaneEstimate,
    LeadEstimate,
    PerceptionFrame,
    SensorSuite,
    dead_reckon,
    time_to_collision,
)
from .planner import Planner
from .route import Route
from .safety import Mode, SafetyOutput, SafetySupervisor
from .vehicle import DriverInput, Vehicle, VehicleCommand, VehicleState
from .world import LeadVehicle

HUMAN_BRAKE_DECEL = 3.0   # m/s^2, how hard the simulated driver brakes
LOG_FIELDS = ("t", "x", "y", "s", "speed", "accel", "steer", "lateral", "mode", "behavior",
              "speed_cap", "gap", "lead_speed", "ttc", "aeb", "alert", "advisory")


@dataclass
class Scenario:
    name: str
    description: str
    route: Route
    duration: float
    initial_speed: float = 0.0
    lead: LeadVehicle | None = None
    driver: Callable[[float], DriverInput] = field(
        default=lambda t: DriverInput(engage_button=t < 0.5))
    sensor_dropout: tuple[float, float] | None = None   # (start, end) seconds
    seed: int = 0


@dataclass
class Result:
    scenario: str
    log: dict[str, list[Any]]
    collision: bool
    red_light_violations: int
    min_gap: float
    min_ttc: float
    max_lateral_error: float       # while the system was steering
    max_jerk: float
    aeb_activations: int
    distance: float
    final_mode: str
    advisories: tuple[Advisory, ...] = ()
    observer_error: str | None = None    # set if an observer raised and was detached

    @property
    def passed(self) -> bool:
        return not self.collision and self.red_light_violations == 0

    def summary(self) -> str:
        rows = [
            ("Result", "PASS" if self.passed else "FAIL"),
            ("Collision", "yes" if self.collision else "no"),
            ("Red-light violations", self.red_light_violations),
            ("Min gap to lead", _fmt(self.min_gap, "m")),
            ("Min time-to-collision", _fmt(self.min_ttc, "s")),
            ("Max lane-center error", _fmt(self.max_lateral_error, "m")),
            ("Max jerk", _fmt(self.max_jerk, "m/s^3")),
            ("AEB activations", self.aeb_activations),
            ("Distance driven", _fmt(self.distance, "m")),
            ("Final mode", self.final_mode),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)


def run(scenario: Scenario, config: Config | None = None,
        observer: Observer | None = None) -> Result:
    """Drive a scenario closed-loop.

    ``observer``, if given, sees a read-only :class:`Snapshot` after every
    control step. Its advisories are recorded and nothing else: they never reach
    the planner, the controllers or the safety supervisor. An observer that
    raises is detached and the drive continues unchanged, because a failure in
    an advisory layer must not become a failure of the vehicle.
    """
    cfg = config or Config()
    dt = 1.0 / cfg.control_rate_hz
    sense_every = round(cfg.control_rate_hz / cfg.sensors.rate_hz)
    route, lead = scenario.route, scenario.lead

    x0, y0, yaw0 = route.pose_at(0.0)
    vehicle = Vehicle(cfg.vehicle, VehicleState(x0, y0, yaw0, scenario.initial_speed))
    sensors = SensorSuite(route, cfg.vehicle, cfg.sensors, scenario.seed)
    planner = Planner(route, cfg.planner)
    lat = LateralController(cfg.controller, cfg.vehicle)
    lon = LongitudinalController(cfg.controller)
    safety = SafetySupervisor(cfg.safety, cfg.vehicle)
    human = LateralController(cfg.controller, cfg.vehicle)  # stand-in for a human steering
    metrics = _Metrics(route)
    log: dict[str, list[Any]] = {k: [] for k in LOG_FIELDS}

    frame: PerceptionFrame | None = None
    lane: LaneEstimate | None = None
    plan = None
    hint = None
    advisories: list[Advisory] = []
    observer_error: str | None = None

    for k in range(int(scenario.duration / dt)):
        t = k * dt
        ego = vehicle.state
        truth = route.project(ego.x, ego.y, ego.yaw, hint)
        hint = truth.index
        front_s = sensors.ego_front_s(truth.s)
        if lead is not None:
            lead.step(t, dt, front_s)

        # Sensors and planning at the perception rate; between frames (or
        # during an injected dropout) the lane estimate is dead-reckoned.
        dropout = scenario.sensor_dropout
        dropped = dropout is not None and dropout[0] <= t < dropout[1]
        if k % sense_every == 0 and not dropped:
            frame = sensors.sense(t, ego, lead)
            lane = frame.lane
            plan = planner.plan(frame, ego.speed)
        elif lane is not None:
            yaw_rate = ego.speed * math.tan(ego.steer) / cfg.vehicle.wheelbase
            lane = dead_reckon(lane, ego.speed, yaw_rate, route.curvature_at(lane.s), dt)

        # Controllers at the control rate.
        requested = VehicleCommand()
        if lane is not None and plan is not None:
            preview = max(ego.speed, 1.0) * cfg.controller.curvature_preview
            requested = VehicleCommand(
                lon.update(plan.accel, ego.accel, dt, stopped=ego.speed < 0.1),
                lat.update(lane, route.curvature_at(lane.s + preview), ego.speed),
            )
        driver = scenario.driver(t)
        was_mode = safety.mode
        out = safety.update(t, frame, driver, requested, ego, dt)
        if out.aeb_active or (out.mode != was_mode and out.mode in (Mode.ENGAGED, Mode.OFF)):
            lon.reset(ego.accel)  # resume smoothly from wherever the car actually is

        advisory = None
        if observer is not None:
            fresh = frame is not None and t - frame.t <= cfg.safety.sensor_timeout
            snapshot = Snapshot(
                t=t, mode=out.mode.value, aeb=out.aeb_active, alert=out.alert,
                behavior=plan.behavior if plan else "", speed=ego.speed, accel=ego.accel,
                lane_lateral=lane.lateral if lane is not None else None,
                ttc=time_to_collision(frame.lead) if fresh and frame is not None else math.inf,
                perception_age=t - frame.t if frame is not None else math.inf,
            )
            try:
                advisory = observer.observe(snapshot)
            except Exception as exc:
                observer_error = f"{type(exc).__name__}: {exc}"
                observer = None
            if advisory is not None:
                advisories.append(advisory)

        # The human fills in whatever the system doesn't control.
        truth_lane = LaneEstimate(truth.s, truth.lateral, truth.heading_error, truth.curvature)
        steer = (out.command.steer if out.controls_lateral
                 else human.update(truth_lane, truth.curvature, ego.speed))
        accel = (out.command.accel if out.controls_longitudinal
                 else -HUMAN_BRAKE_DECEL if driver.brake_pressed else 0.0)
        prev_accel = ego.accel
        vehicle.step(VehicleCommand(accel, steer), dt)

        gap = lead.rear_s - front_s if lead is not None and lead.present else math.inf
        ttc = time_to_collision(frame.lead) if frame is not None else math.inf
        metrics.update(t, dt, front_s, gap, frame.lead if frame else None, truth.lateral,
                       out, prev_accel, vehicle.state)

        for key, value in (
            ("t", t), ("x", ego.x), ("y", ego.y), ("s", truth.s), ("speed", ego.speed),
            ("accel", ego.accel), ("steer", ego.steer), ("lateral", truth.lateral),
            ("mode", out.mode.value), ("behavior", plan.behavior if plan else ""),
            ("speed_cap", plan.speed_cap if plan else math.nan), ("gap", gap),
            ("lead_speed", lead.speed if lead is not None and lead.present else math.nan),
            ("ttc", ttc), ("aeb", out.aeb_active), ("alert", out.alert),
            ("advisory", advisory.message if advisory is not None else None),
        ):
            log[key].append(value)

        stopped_at_end = truth.s >= route.length - 25.0 and ego.speed < 0.05
        if stopped_at_end or truth.s >= route.length - 1.0:
            break

    m = metrics
    return Result(scenario.name, log, m.collision, m.red_light_violations, m.min_gap,
                  m.min_ttc, m.max_lateral_error, m.max_jerk, m.aeb_activations,
                  float(log["s"][-1]), safety.mode.value, tuple(advisories), observer_error)


class _Metrics:
    """Scores a drive against ground truth."""

    def __init__(self, route: Route):
        self.route = route
        self.collision = False
        self.red_light_violations = 0
        self.min_gap = self.min_ttc = math.inf
        self.max_lateral_error = self.max_jerk = 0.0
        self.aeb_activations = 0
        self._prev_front_s: float | None = None
        self._prev_aeb = False
        self._last_aeb_t = -math.inf

    def update(self, t: float, dt: float, front_s: float, gap: float,
               lead: LeadEstimate | None, lateral: float, out: SafetyOutput, prev_accel: float,
               after: VehicleState) -> None:
        self.collision |= gap <= 0.0
        self.min_gap = min(self.min_gap, gap)
        ttc = time_to_collision(lead)
        if ttc < 60.0:
            self.min_ttc = min(self.min_ttc, ttc)
        if out.controls_lateral:
            self.max_lateral_error = max(self.max_lateral_error, abs(lateral))

        # Comfort: only while the system drives normally (not AEB, not stopping).
        if out.aeb_active:
            self._last_aeb_t = t
            if not self._prev_aeb:
                self.aeb_activations += 1
        self._prev_aeb = out.aeb_active
        if out.controls_longitudinal and t - self._last_aeb_t > 1.0 and after.speed > 0.5:
            self.max_jerk = max(self.max_jerk, abs(after.accel - prev_accel) / dt)

        # Crossing a stop line while the light is red.
        if self._prev_front_s is not None:
            light = self.route.next_light(self._prev_front_s)
            if light is not None and light.s <= front_s and light.state(t) == "red":
                self.red_light_violations += 1
        self._prev_front_s = front_s


def _fmt(v: float, unit: str) -> str:
    return "n/a" if math.isinf(v) else f"{v:.2f} {unit}"
