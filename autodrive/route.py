"""Road geometry: a lane centerline parameterized by arc length.

A route is built from (length, curvature) segments, the way road designers
lay out straights and constant-radius curves. The map also carries speed
limits and stop lines for traffic lights.
"""

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from .vehicle import wrap_angle


@dataclass(frozen=True)
class Segment:
    length: float        # m
    curvature: float     # 1/m, positive = left turn
    speed_limit: float   # m/s


@dataclass(frozen=True)
class TrafficLight:
    s: float             # m, stop line position along the route
    green: float = 20.0  # s
    yellow: float = 4.0  # s
    red: float = 15.0    # s
    offset: float = 0.0  # s, phase at t=0

    def state(self, t: float) -> str:
        phase = (t + self.offset) % (self.green + self.yellow + self.red)
        if phase < self.green:
            return "green"
        if phase < self.green + self.yellow:
            return "yellow"
        return "red"


@dataclass(frozen=True)
class FrenetPoint:
    s: float             # m along the route
    lateral: float       # m, positive = left of centerline
    heading_error: float  # rad, vehicle yaw minus path heading
    curvature: float     # 1/m at the closest point
    index: int           # closest sample, used as a search hint


@dataclass
class Route:
    segments: list[Segment]
    lights: list[TrafficLight] = field(default_factory=list)
    ds: float = 0.5

    def __post_init__(self) -> None:
        first = self.segments[0]
        xs, ys, yaws, ss = [0.0], [0.0], [0.0], [0.0]
        ks, limits = [first.curvature], [first.speed_limit]
        for seg in self.segments:
            n = max(1, round(seg.length / self.ds))
            step = seg.length / n
            for _ in range(n):
                # Exact integration of a constant-curvature arc.
                yaw0 = yaws[-1]
                yaw1 = yaw0 + seg.curvature * step
                if abs(seg.curvature) > 1e-9:
                    dx = (math.sin(yaw1) - math.sin(yaw0)) / seg.curvature
                    dy = (math.cos(yaw0) - math.cos(yaw1)) / seg.curvature
                else:
                    dx, dy = step * math.cos(yaw0), step * math.sin(yaw0)
                xs.append(xs[-1] + dx)
                ys.append(ys[-1] + dy)
                yaws.append(yaw1)
                ss.append(ss[-1] + step)
                ks.append(seg.curvature)
                limits.append(seg.speed_limit)
        self.x = np.array(xs)
        self.y = np.array(ys)
        self.yaw = np.array(yaws)
        self.curvature = np.array(ks)
        self.s = np.array(ss)
        self.speed_limit = np.array(limits)
        self.length = float(self.s[-1])
        self.lights = sorted(self.lights, key=lambda l: l.s)

    def pose_at(self, s: float) -> tuple[float, float, float]:
        i = self._index(s)
        return float(self.x[i]), float(self.y[i]), float(self.yaw[i])

    def curvature_at(self, s: float) -> float:
        return float(self.curvature[self._index(s)])

    def speed_limit_at(self, s: float) -> float:
        return float(self.speed_limit[self._index(s)])

    def project(self, x: float, y: float, yaw: float, hint: int | None = None) -> FrenetPoint:
        """Find the closest centerline point to (x, y).

        With a hint (last known index) only a local window is searched, which
        keeps this O(1) per control tick and prevents snapping to a far-away
        part of the road where it loops near itself.
        """
        if hint is None:
            lo, hi = 0, len(self.s)
        else:
            lo, hi = max(0, hint - 40), min(len(self.s), hint + 80)
        d2 = (self.x[lo:hi] - x) ** 2 + (self.y[lo:hi] - y) ** 2
        i = lo + int(np.argmin(d2))
        path_yaw = float(self.yaw[i])
        dx, dy = x - self.x[i], y - self.y[i]
        # Project the offset onto the path tangent to refine s between samples.
        along = dx * math.cos(path_yaw) + dy * math.sin(path_yaw)
        lateral = -dx * math.sin(path_yaw) + dy * math.cos(path_yaw)
        s = float(np.clip(self.s[i] + along, 0.0, self.length))
        return FrenetPoint(s, lateral, wrap_angle(yaw - path_yaw),
                           float(self.curvature[i]), i)

    def next_light(self, s: float) -> TrafficLight | None:
        i = bisect.bisect_right([l.s for l in self.lights], s)
        return self.lights[i] if i < len(self.lights) else None

    def _index(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))
