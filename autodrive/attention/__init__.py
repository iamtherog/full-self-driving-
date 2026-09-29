"""Attention monitor: an n-gram model of ordinary driving that helps the driver stay engaged.

What it is
    A Kneser-Ney n-gram language model trained on the *event stream* of
    randomized nominal drives (mode, planner behavior, banded braking,
    time-to-collision and lane offset, one token per 0.5 s). Online it scores
    each new event by its surprisal and, when driving stops looking ordinary,
    offers the driver a short templated advisory: "Stay alert: closing on the
    vehicle ahead, time to collision 2.4 s."

What it is not
    It is not part of the control or safety path, and it cannot become part of
    it. It receives read-only snapshots through :mod:`autodrive.observe`, and
    :func:`autodrive.sim.run` records its advisories without passing them to
    perception, planning, control or the safety supervisor. A test drives every
    scenario with and without the monitor and requires identical logs. It never
    generates free text: the model decides *when* to speak, and fixed templates
    filled with measured values decide *what* is said. It defers to the safety
    supervisor, staying silent whenever a supervisor alert is up.

Where it runs
    * Online, as an observer in the simulation loop (``python -m autodrive --attention``).
    * Offline, over recorded drives, for post-drive review of every flag
      including the ones it withheld (``python -m autodrive.attention report``).

Credits
    The design follows the **Lean N-gram Generator** by **Roger Feeley
    Lussier**: an offline Kneser-Ney learner, a sequential test (rather than a
    comparison of two point estimates) to choose between candidate models, and
    a trained artifact bound by digest to the definitions it was built against.
    This package is an independent implementation from the published
    algorithms and contains none of that project's source:

    * R. Kneser, H. Ney. Improved backing-off for m-gram language modeling. ICASSP 1995.
    * S. F. Chen, J. Goodman. An empirical study of smoothing techniques for
      language modeling. Computer Speech and Language 13(4), 1999.
    * A. Wald. Sequential tests of statistical hypotheses. Annals of
      Mathematical Statistics 16(2), 1945.

    It departs from the Lean N-gram Generator in two places: the sequential test
    uses one trial per independent drive instead of one per token (consecutive
    tokens are correlated, which voids the test's error bounds), and the model
    is used only as a scorer, never as a text generator.

Offline by construction
    No network access, no remote model, no API calls, no per-use cost: the
    package uses the Python standard library and numpy (through the simulator).
    ``tests/test_attention.py`` trains and runs it with sockets disabled.
"""

from .events import Event, TokenSpec
from .kneser_ney import KneserNey
from .model import DEFAULT_PATH, ArtifactError, AttentionModel
from .monitor import AttentionMonitor, Flag

__all__ = ["DEFAULT_PATH", "ArtifactError", "AttentionModel", "AttentionMonitor", "Event", "Flag",
           "KneserNey", "TokenSpec"]
