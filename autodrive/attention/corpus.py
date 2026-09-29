"""Training data: randomized *nominal* drives, run through the real stack.

The attention model learns what ordinary driving looks like, so its corpus must
contain only ordinary driving. Each seed produces a random highway or urban
route with gentle traffic, and the drive is recorded through the same observer
interface the monitor uses online, so training sees exactly what deployment
sees (no simulator ground truth, no train/serve skew).

"Nominal" is enforced, not assumed. A generated drive is excluded from the
corpus if the stack ever used emergency braking, faulted, collided, ran a red
light or left the engaged modes, and every exclusion is counted and reported
with the trained model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from ..observe import Advisory, Snapshot
from ..route import Route, Segment, TrafficLight
from ..sim import Scenario, run
from ..world import LeadVehicle, SpeedChange
from .events import TokenSpec, events

MPH = 0.44704
#: ITE yellow change interval, y = t + v / (2a) on level grade, with perception-
#: reaction time t = 1.0 s and deceleration a = 3.05 m/s^2 (10 ft/s^2). Signals
#: are timed this way so that a driver at the limit is never caught unable either
#: to stop or to clear. Random yellows shorter than this create a dilemma zone no
#: driver, human or software, can resolve.
ITE_REACTION, ITE_DECEL, MIN_YELLOW = 1.0, 3.05, 3.0


def ite_yellow(limit: float) -> float:
    return max(MIN_YELLOW, ITE_REACTION + limit / (2.0 * ITE_DECEL))
MAX_DURATION = 45.0    # s per drive; bounds corpus cost without cutting routes short often
SEED_DOMAIN = 0x5EED_A77E  # separates corpus seeds from any other use of the same integers


class Recorder:
    """An observer that keeps every snapshot and never advises."""

    def __init__(self) -> None:
        self.snapshots: list[Snapshot] = []

    def observe(self, snapshot: Snapshot) -> Advisory | None:
        self.snapshots.append(snapshot)
        return None


def nominal_scenario(seed: int) -> Scenario:
    """A random ordinary drive. Deterministic in ``seed``."""

    rng = np.random.default_rng([SEED_DOMAIN, seed])
    if rng.random() < 0.5:
        return _highway(seed, rng)
    return _urban(seed, rng)


def _highway(seed: int, rng: np.random.Generator) -> Scenario:
    limit = float(rng.choice([55, 60, 65, 70])) * MPH
    segments = []
    for _ in range(int(rng.integers(4, 7))):
        if rng.random() < 0.5:
            segments.append(Segment(float(rng.uniform(150, 400)), 0.0, limit))
        else:
            radius = float(rng.uniform(300, 900))
            segments.append(Segment(float(rng.uniform(100, 350)), float(rng.choice([-1, 1])) / radius, limit))
    lead = None
    if rng.random() < 0.7:
        speed = limit - float(rng.uniform(0, 4))
        plan = [SpeedChange(t=float(rng.uniform(5, 35)),
                            target=max(10.0, speed + float(rng.uniform(-4, 4))),
                            accel=float(rng.uniform(0.4, 1.0)))
                for _ in range(int(rng.integers(0, 3)))]
        lead = LeadVehicle(s=float(rng.uniform(50, 110)), speed=speed, plan=plan)
    return Scenario(f"nominal-{seed}", "highway", Route(segments),
                    duration=MAX_DURATION, initial_speed=limit * float(rng.uniform(0.7, 1.0)),
                    lead=lead, seed=seed)


def _urban(seed: int, rng: np.random.Generator) -> Scenario:
    limit = float(rng.choice([25, 30, 35])) * MPH
    segments, lights, s = [], [], 0.0
    for i in range(int(rng.integers(3, 6))):
        if i and rng.random() < 0.5:
            radius = 25.0
            segments.append(Segment(math.pi / 2 * radius, float(rng.choice([-1, 1])) / radius, 15 * MPH))
            s += math.pi / 2 * radius
        length = float(rng.uniform(120, 260))
        segments.append(Segment(length, 0.0, limit))
        if rng.random() < 0.4:
            lights.append(TrafficLight(s=s + float(rng.uniform(60, length - 20)),
                                       green=float(rng.uniform(10, 20)),
                                       yellow=ite_yellow(limit) + float(rng.uniform(0, 0.5)),
                                       red=float(rng.uniform(10, 20)), offset=float(rng.uniform(0, 40))))
        s += length
    lead = None
    if rng.random() < 0.5:
        lead = LeadVehicle(s=float(rng.uniform(25, 45)), speed=float(rng.uniform(6, 11)),
                           leave_at=float(rng.uniform(5, 20)))
    initial = 0.0 if rng.random() < 0.5 else limit * float(rng.uniform(0.5, 1.0))
    if lead is not None:
        # The lead starts only 25-45 m ahead. Starting faster than it would put
        # the drive a couple of seconds from collision at t = 0: not nominal.
        initial = min(initial, lead.speed)
    return Scenario(f"nominal-{seed}", "urban", Route(segments, lights),
                    duration=MAX_DURATION, initial_speed=initial, lead=lead, seed=seed)


@dataclass(frozen=True)
class Drive:
    seed: int
    kind: str
    tokens: tuple[str, ...]


@dataclass
class Corpus:
    drives: list[Drive] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)   # reason -> count

    @property
    def tokens(self) -> int:
        return sum(len(d.tokens) for d in self.drives)


def exclusion_reason(result_mode_log: Sequence[str], aeb_activations: int, collision: bool,
                     red_light_violations: int) -> str | None:
    if collision:
        return "collision"
    if red_light_violations:
        return "red_light"
    if aeb_activations:
        return "aeb"
    if any(m != "engaged" for m in result_mode_log):
        return "left_engaged_mode"
    return None


def record(seed: int, spec: TokenSpec) -> tuple[Drive, str | None]:
    """Drive one nominal scenario and tokenise it. Returns the drive and any exclusion reason."""

    scenario = nominal_scenario(seed)
    recorder = Recorder()
    result = run(scenario, observer=recorder)
    reason = exclusion_reason(result.log["mode"], result.aeb_activations, result.collision,
                              result.red_light_violations)
    tokens = tuple(e.token for e in events(recorder.snapshots, spec))
    return Drive(seed, scenario.description, tokens), reason


def build(seeds: Sequence[int], spec: TokenSpec, workers: int | None = None) -> Corpus:
    """Record every seed. Parallel across processes; results keep seed order, so the
    corpus (and everything trained from it) is identical for any worker count."""

    job = partial(record, spec=spec)
    if workers == 1:
        results = [job(seed) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(job, seeds, chunksize=8))
    corpus = Corpus()
    for drive, reason in results:
        if reason is None:
            corpus.drives.append(drive)
        else:
            corpus.excluded[reason] = corpus.excluded.get(reason, 0) + 1
    return corpus
