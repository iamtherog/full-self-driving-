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
        self._committed_light_s: float | None = None   # decided to go through this light
        self._stopping_light_s: float | None = None    # decided to stop for this light

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
        """Stop for a yellow or red light, or commit to going through it.

        The decision is the classic dilemma-zone test: stop if the deceleration
        needed to halt before the line is within the limit for the light's color,
        otherwise go. Three details make it hold up in closed loop:

        * A fresh decision allows for brake build-up. Braking reaches the
          commanded level only after ``brake_buildup`` seconds, so the distance
          covered meanwhile is not available for stopping. Once a stop is under
          way the brakes are already applied, and the allowance no longer applies.
        * Both decisions are latched per light. Re-deciding every tick lets a car
          that has begun to stop drift just past the yellow limit and flip to
          "go" when it is too close to clear the intersection. Once stopping, only
          the harder red-light limit can reverse the decision.
        * A commitment to go lapses if the car slows to ``commit_min_speed``
          before the line (for example braking for a turn). A car that slow can
          always stop, and must not creep across on red.
        """

        light, p = frame.light, self.p
        if light is None or light.state == "green":
            return None
        light_s = frame.lane.s + light.distance
        same = lambda latched: latched is not None and abs(latched - light_s) < 1.0  # noqa: E731

        if same(self._committed_light_s):
            if speed > p.commit_min_speed:
                return None                       # already decided to go through this one
            self._committed_light_s = None        # slowed right down: stop after all

        stopping = same(self._stopping_light_s)
        d_stop = light.distance - p.stop_line_margin
        d_brake = d_stop if stopping else d_stop - speed * p.brake_buildup
        required = speed ** 2 / (2.0 * d_brake) if d_brake > 0.0 else math.inf
        limit = p.red_max_decel if stopping or light.state == "red" else p.yellow_max_decel
        if required > limit and speed > p.commit_min_speed:
            self._committed_light_s, self._stopping_light_s = light_s, None
            return None
        self._stopping_light_s = light_s
        return self._idm(speed, cap, max(d_stop, 0.05), 0.0, 0.5, 1.0)

    def _idm(self, v: float, v0: float, gap: float, lead_speed: float,
             s0: float, headway: float) -> float:
        """Intelligent Driver Model acceleration toward a lead (or stop line)."""
        p = self.p
        v0 = max(v0, 0.1)
        dv = v - lead_speed
        braking_term = v * dv / (2.0 * math.sqrt(p.idm_max_accel * p.idm_comfort_decel))
        s_star = s0 + max(0.0, v * headway + braking_term)
        gap = max(gap, 0.1)
        return p.idm_max_accel * (1.0 - (v / v0) ** 4 - (s_star / gap) ** 2)
