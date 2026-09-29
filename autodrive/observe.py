"""The read-only boundary between the driving loop and anything that watches it.

An observer is handed a :class:`Snapshot` after every control step and may
return an :class:`Advisory` for the driver. Nothing an observer returns is read
by perception, planning, control or the safety supervisor; :func:`autodrive.sim.run`
only records it. Two properties make that hold:

* A snapshot is a frozen copy of plain values, so an observer cannot mutate
  loop state through it.
* A snapshot carries only what the system itself knows (perceived lane offset,
  perceived time-to-collision, mode, planner behavior, reported acceleration),
  never simulator ground truth, so an observer trained offline sees the same
  inputs it would see online.

``tests/test_attention.py`` drives every scenario with and without an observer
and requires the two logs to be identical, so the isolation is checked rather
than assumed.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Snapshot:
    t: float                     # s
    mode: str                    # safety supervisor mode
    aeb: bool                    # emergency braking active this step
    alert: str | None            # safety supervisor alert, if any
    behavior: str                # planner behavior, "" before the first plan
    speed: float                 # m/s, reported by the vehicle
    accel: float                 # m/s^2, reported by the vehicle
    lane_lateral: float | None   # m, perceived (or dead-reckoned) offset from lane center
    ttc: float                   # s, from a fresh lead track; inf when there is none
    perception_age: float        # s since the last perception frame; inf before the first


@dataclass(frozen=True)
class Advisory:
    """A message offered to the driver. Never an instruction to the vehicle."""

    t: float
    message: str
    surprisal_bits: float


class Observer(Protocol):
    def observe(self, snapshot: Snapshot) -> Advisory | None: ...
