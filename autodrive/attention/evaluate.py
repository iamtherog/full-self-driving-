"""Score the attention monitor on the named scenarios.

The named scenarios were never part of training (training uses randomized
nominal drives only), so this is an out-of-sample check of two claims:

* On ordinary driving (``highway``, ``city``) the monitor should stay quiet.
* Before an incident, which here means the first step of emergency braking or a
  system fault, it should flag the build-up. Lead time is measured from the
  first flag in the five seconds before the incident (and separately from the
  first advisory actually shown to the driver) to the incident itself.

Everything here is recomputed from a fresh run; nothing is read from the artifact's
stored metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import scenarios
from ..sim import Result, run
from .events import Event
from .model import AttentionModel
from .monitor import AttentionMonitor, Flag

#: How far before an incident a flag still counts as anticipating it.
LOOKBACK = 5.0


@dataclass(frozen=True)
class Row:
    scenario: str
    result: Result
    flags: tuple[Flag, ...]
    scores: tuple[tuple[Event, float], ...]   # every event and its surprisal, in order
    incident_at: float | None
    first_warning_at: float | None     # first flag (shown or kept for review) within LOOKBACK before it
    first_shown_at: float | None       # first advisory shown to the driver within LOOKBACK before it

    @property
    def lead_time(self) -> float | None:
        if self.incident_at is None or self.first_warning_at is None:
            return None
        return self.incident_at - self.first_warning_at

    @property
    def shown_lead_time(self) -> float | None:
        if self.incident_at is None or self.first_shown_at is None:
            return None
        return self.incident_at - self.first_shown_at

    @property
    def delivered(self) -> tuple[Flag, ...]:
        return tuple(f for f in self.flags if f.delivered)


def incident_onset(result: Result) -> float | None:
    for t, aeb, mode in zip(result.log["t"], result.log["aeb"], result.log["mode"], strict=True):
        if aeb or mode == "fault":
            return float(t)
    return None


def score(name: str, model: AttentionModel) -> Row:
    monitor = AttentionMonitor(model)
    result = run(scenarios.ALL[name](), observer=monitor)
    onset = incident_onset(result)
    warning = shown = None
    if onset is not None:
        before = [f for f in monitor.flags if onset - LOOKBACK <= f.t <= onset]
        warning = before[0].t if before else None
        shown = next((f.t for f in before if f.delivered), None)
    return Row(name, result, tuple(monitor.flags), tuple(monitor.scores), onset, warning, shown)


def score_all(model: AttentionModel) -> list[Row]:
    return [score(name, model) for name in scenarios.ALL]
