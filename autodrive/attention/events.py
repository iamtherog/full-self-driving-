"""Turn a stream of snapshots into a sequence of discrete driving events.

A language model needs a vocabulary. Here a "word" is what the system knew
about one fixed window of time (0.5 s by default), quantised into a handful of
bands and written as a readable string, for example::

    engaged|follow|a2|t2|l0

which reads: system engaged, planner following a lead, mild braking, time to
collision 3-6 s, within 15 cm of lane center.

Each window is summarised by its *worst* value (most severe mode, strongest
braking, shortest time-to-collision, largest lane offset, any emergency
braking), not its mean or its last sample. A single 10 ms step of emergency
braking must mark its window. Averaging it away would hide exactly the events
the monitor exists to notice.

The band edges live in :class:`TokenSpec`. A trained model records the digest
of the spec it was trained with and refuses to run against a different one,
because the same token string would mean something else.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

from ..observe import Snapshot

#: Severity order used to pick a window's mode. A window that touched FAULT is
#: a FAULT window, however briefly.
MODE_SEVERITY: dict[str, int] = {"engaged": 0, "lat_override": 1, "off": 2, "fault": 3}


@dataclass(frozen=True)
class TokenSpec:
    """Band edges that define the vocabulary. Part of every trained model's identity."""

    window: float = 0.5                                        # s per event
    accel_edges: tuple[float, ...] = (-3.0, -1.5, -0.3, 0.3)  # m/s^2; band 0 = hardest braking
    ttc_edges: tuple[float, ...] = (1.6, 3.0, 6.0)            # s; 1.6 = SafetyLimits.aeb_ttc
    lateral_edges: tuple[float, ...] = (0.15, 0.4)            # m from lane center
    stale_after: float = 0.25                                  # s; = SafetyLimits.sensor_timeout

    def __post_init__(self) -> None:
        if self.window <= 0.0:
            raise ValueError("window must be positive")
        for name in ("accel_edges", "ttc_edges", "lateral_edges"):
            edges = getattr(self, name)
            if list(edges) != sorted(set(edges)):
                raise ValueError(f"{name} must be strictly increasing")

    def digest(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(b"autodrive.tokenspec.v1\x00" + body.encode()).hexdigest()


@dataclass(frozen=True)
class Features:
    """The worst case observed within one window, raw and banded."""

    mode: str
    aeb: bool
    behavior: str
    stale: bool
    min_accel: float
    min_ttc: float
    max_lateral: float | None
    accel_band: int
    ttc_band: int | None        # None: no lead vehicle tracked
    lateral_band: int | None    # None: no lane estimate yet


@dataclass(frozen=True)
class Event:
    t_start: float
    t_end: float
    token: str
    features: Features


def band(value: float, edges: tuple[float, ...]) -> int:
    """Index of the half-open band ``[edges[i-1], edges[i])`` containing ``value``."""

    return bisect.bisect_right(edges, value)


def token_of(f: Features) -> str:
    ttc = "x" if f.ttc_band is None else str(f.ttc_band)
    lat = "x" if f.lateral_band is None else str(f.lateral_band)
    parts = [f.mode + ("+aeb" if f.aeb else ""), f.behavior or "none",
             f"a{f.accel_band}", f"t{ttc}", f"l{lat}"]
    if f.stale:
        parts.append("stale")
    return "|".join(parts)


def summarise(window: list[Snapshot], spec: TokenSpec) -> Features:
    """Worst-case summary of one window of snapshots."""

    if not window:
        raise ValueError("cannot summarise an empty window")
    mode = max((s.mode for s in window), key=lambda m: MODE_SEVERITY.get(m, len(MODE_SEVERITY)))
    min_accel = min(s.accel for s in window)
    max_accel = max(s.accel for s in window)
    min_ttc = min(s.ttc for s in window)
    offsets = [abs(s.lane_lateral) for s in window if s.lane_lateral is not None]
    max_lateral = max(offsets) if offsets else None

    # Braking dominates acceleration: if the window contains any deceleration
    # beyond the dead band, it is labelled by that deceleration.
    braking = band(min_accel, spec.accel_edges)
    dead_band = band(0.0, spec.accel_edges)
    accel_band = braking if braking < dead_band else band(max_accel, spec.accel_edges)

    return Features(
        mode=mode,
        aeb=any(s.aeb for s in window),
        behavior=window[-1].behavior,
        stale=any(s.perception_age > spec.stale_after for s in window),
        min_accel=min_accel,
        min_ttc=min_ttc,
        max_lateral=max_lateral,
        accel_band=accel_band,
        ttc_band=band(min_ttc, spec.ttc_edges) if math.isfinite(min_ttc) else None,
        lateral_band=band(max_lateral, spec.lateral_edges) if max_lateral is not None else None,
    )


class Windower:
    """Groups snapshots into fixed windows and emits one event per completed window.

    A window is complete when the first snapshot of the next window arrives, so
    an event is available at most one window plus one control step after the
    moment it describes. Windows are aligned to multiples of ``spec.window``
    on the simulation clock, which makes the grouping independent of when the
    observer was attached.
    """

    def __init__(self, spec: TokenSpec):
        self.spec = spec
        self._index: int | None = None
        self._buffer: list[Snapshot] = []

    def push(self, snapshot: Snapshot) -> Event | None:
        # The epsilon absorbs t = k * dt landing a hair below a window edge.
        index = math.floor(snapshot.t / self.spec.window + 1e-9)
        event = None
        if self._index is not None and index != self._index and self._buffer:
            features = summarise(self._buffer, self.spec)
            event = Event(self._index * self.spec.window, (self._index + 1) * self.spec.window,
                          token_of(features), features)
            self._buffer = []
        self._index = index
        self._buffer.append(snapshot)
        return event


def events(snapshots: Iterable[Snapshot], spec: TokenSpec) -> Iterator[Event]:
    """Every complete window of a recorded drive. A trailing partial window is dropped."""

    windower = Windower(spec)
    for snapshot in snapshots:
        event = windower.push(snapshot)
        if event is not None:
            yield event
