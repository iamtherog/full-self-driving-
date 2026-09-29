"""Behavior and speed planning.

Each longitudinal "concern" (road speed, curves, a lead car, a traffic light)
proposes an acceleration; the most conservative one wins. This keeps each
rule simple to reason about and test on its own.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import PlannerParams
from .perception import PerceptionFrame
from .route import Route


@dataclass(frozen=True)
class Plan:
    accel: float           # m/s^2, desired
    speed_cap: float       # m/s, from road limits and curves
    behavior: str          # "cruise" | "curve" | "follow" | "stop_for_light"


class Planner:
    def __init__(self, route: Route, params: PlannerParams):
        self.route, self.p = route, params
        self._committed_light_s: float | None = None

    def plan(self, frame: PerceptionFrame, speed: float) -> Plan:
        p, lane = self.p, frame.lane
        cap, cap_reason = self.speed_cap(lane.s, speed)

        track_cap = np.clip(p.speed_gain * (cap - speed), -p.max_speed_decel, p.idm_max_accel)
        candidates = [(float(track_cap), cap_reason)]

        if frame.lead is not None:
            candidates.append((self._idm(speed, cap, frame.lead.gap, frame.lead.lead_speed,
                                         p.idm_min_gap, p.idm_time_headway), "follow"))

        light_accel = self._light_accel(frame, speed, cap)
        if light_accel is not None:
            candidates.append((light_accel, "stop_for_light"))

        accel, behavior = min(candidates, key=lambda c: c[0])
        return Plan(accel, cap, behavior)

    def speed_cap(self, s: float, speed: float) -> tuple[float, str]:
        """Fastest speed from which every upcoming limit is reachable at comfort decel."""
        p = self.p
        horizon = max(40.0, speed * p.lookahead_time)
        d = np.arange(0.0, horizon, 2.0)
        s_ahead = np.minimum(s + d, self.route.length)
        idx = np.clip(np.searchsorted(self.route.s, s_ahead), 0, len(self.route.s) - 1)
        limit = self.route.speed_limit[idx]
        k = np.abs(self.route.curvature[idx])
        curve = np.sqrt(p.max_lat_accel / np.maximum(k, 1e-6))
        local = np.minimum(limit, curve)
        reachable = np.sqrt(local ** 2 + 2.0 * p.comfort_decel * d)
        i = int(np.argmin(reachable))
        reason = "curve" if curve[i] < limit[i] else "cruise"
        # Stop at the end of the route.
        end = math.sqrt(2.0 * p.comfort_decel * max(0.0, self.route.length - 20.0 - s))
        if end < reachable[i]:
            return end, "cruise"
        return float(reachable[i]), reason

    def _light_accel(self, frame: PerceptionFrame, speed: float, cap: float) -> float | None:
        light, p = frame.light, self.p
        if light is None or light.state == "green":
            return None
        light_s = frame.lane.s + light.distance
        if self._committed_light_s is not None and abs(self._committed_light_s - light_s) < 1.0:
            return None  # already decided to go through this one
        d_stop = light.distance - p.stop_line_margin
        required = speed ** 2 / (2.0 * max(d_stop, 0.1))
        # Dilemma zone: if we can't stop comfortably, commit to going through.
        limit = p.yellow_max_decel if light.state == "yellow" else p.red_max_decel
        if required > limit and speed > 3.0:
            self._committed_light_s = light_s
            return None
        return self._idm(speed, cap, max(d_stop, 0.05), 0.0, 0.5, 1.0)

    def _idm(self, v: float, v0: float, gap: float, lead_speed: float,
             s0: float, headway: float) -> float:
        """Intelligent Driver Model acceleration toward a lead (or stop line)."""
        p = self.p
        v0 = max(v0, 0.1)
        dv = v - lead_speed
        s_star = s0 + max(0.0, v * headway + v * dv / (2.0 * math.sqrt(p.idm_max_accel * p.idm_comfort_decel)))
        gap = max(gap, 0.1)
        return p.idm_max_accel * (1.0 - (v / v0) ** 4 - (s_star / gap) ** 2)
