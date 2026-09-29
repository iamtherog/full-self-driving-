"""Sensors and perception.

The simulated sensors add noise to ground truth, and perception turns those
raw measurements into the estimates the planner consumes. The planner never
sees ground truth.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import SensorParams, VehicleParams
from .route import Route
from .vehicle import VehicleState
from .world import LeadVehicle


@dataclass(frozen=True)
class LaneEstimate:
    s: float
    lateral: float        # m, positive = left of lane center
    heading_error: float  # rad
    curvature: float      # 1/m


@dataclass(frozen=True)
class LeadEstimate:
    gap: float            # m, ego front bumper to lead rear bumper
    rel_speed: float      # m/s, lead minus ego (negative = closing)
    lead_speed: float     # m/s


@dataclass(frozen=True)
class LightEstimate:
    distance: float       # m, ego front bumper to stop line
    state: str            # "green" | "yellow" | "red"


@dataclass(frozen=True)
class PerceptionFrame:
    t: float
    lane: LaneEstimate
    lead: LeadEstimate | None
    light: LightEstimate | None


class LeadTracker:
    """Constant-velocity Kalman filter on [gap, relative speed]."""

    def __init__(self, gap_noise: float, speed_noise: float):
        self.R = np.diag([gap_noise ** 2, speed_noise ** 2])
        self.x: np.ndarray | None = None
        self.P = np.eye(2)

    def update(self, gap: float, rel_speed: float, dt: float) -> tuple[float, float]:
        z = np.array([gap, rel_speed])
        # (Re)initialize on a new target or an implausible jump (e.g. cut-in).
        if self.x is None or abs(gap - self.x[0]) > 8.0:
            self.x, self.P = z.copy(), self.R.copy()
            return float(z[0]), float(z[1])
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = 2.0  # m/s^2 of unmodeled relative acceleration
        Q = q ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        K = self.P @ np.linalg.inv(self.P + self.R)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(2) - K) @ self.P
        return float(self.x[0]), float(self.x[1])

    def reset(self) -> None:
        self.x = None


class SensorSuite:
    def __init__(self, route: Route, vehicle: VehicleParams, sensors: SensorParams,
                 seed: int = 0):
        self.route, self.vp, self.sp = route, vehicle, sensors
        self.rng = np.random.default_rng(seed)
        self.tracker = LeadTracker(sensors.radar_gap_noise, sensors.radar_speed_noise)
        self._hint: int | None = None
        self._last_t: float | None = None

    def ego_front_s(self, s_rear_axle: float) -> float:
        return s_rear_axle + self.vp.wheelbase + self.vp.front_overhang

    def sense(self, t: float, ego: VehicleState, lead: LeadVehicle | None) -> PerceptionFrame:
        dt = 1.0 / self.sp.rate_hz if self._last_t is None else t - self._last_t
        self._last_t = t
        n = self.rng.normal

        # Camera: lane position relative to the centerline.
        fp = self.route.project(ego.x, ego.y, ego.yaw, self._hint)
        self._hint = fp.index
        lane = LaneEstimate(
            s=fp.s,
            lateral=fp.lateral + n(0.0, self.sp.camera_lateral_noise),
            heading_error=fp.heading_error + n(0.0, self.sp.camera_heading_noise),
            curvature=fp.curvature,
        )
        front_s = self.ego_front_s(fp.s)

        # Radar: nearest in-lane vehicle ahead, if within range.
        lead_est = None
        if lead is not None and lead.present:
            true_gap = lead.rear_s - front_s
            if 0.0 < true_gap < self.sp.radar_range:
                gap, rel = self.tracker.update(
                    true_gap + n(0.0, self.sp.radar_gap_noise),
                    lead.speed - ego.speed + n(0.0, self.sp.radar_speed_noise),
                    dt,
                )
                lead_est = LeadEstimate(max(0.0, gap), rel, max(0.0, ego.speed + rel))
            else:
                self.tracker.reset()
        else:
            self.tracker.reset()

        # Camera: next traffic light and its color.
        light_est = None
        light = self.route.next_light(front_s)
        if light is not None:
            dist = light.s - front_s
            if dist < self.sp.light_detect_range:
                light_est = LightEstimate(dist, light.state(t))

        return PerceptionFrame(t, lane, lead_est, light_est)


def dead_reckon(lane: LaneEstimate, speed: float, yaw_rate: float, road_curvature: float,
                dt: float) -> LaneEstimate:
    """Carry the last lane estimate forward using wheel-speed and yaw-rate odometry.

    Bridges perception gaps so the car keeps tracking its lane while the
    safety supervisor slows it down and asks the driver to take over.
    """
    heading = lane.heading_error + (yaw_rate - speed * road_curvature) * dt
    return LaneEstimate(
        s=lane.s + speed * math.cos(heading) * dt,
        lateral=lane.lateral + speed * math.sin(heading) * dt,
        heading_error=heading,
        curvature=road_curvature,
    )


def time_to_collision(lead: LeadEstimate | None) -> float:
    if lead is None or lead.rel_speed >= -1e-3:
        return math.inf
    return lead.gap / -lead.rel_speed
