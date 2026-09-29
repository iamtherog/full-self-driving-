"""autodrive: a driver-assistance software stack running against a simulated car.

Written by Claude Code (an AI coding assistant). The request that started it:

    "Computer pretty please can you take all the publicly Available github
    information about full self driving cars? then can you make it into a
    rwelly good pwogwam I can use to drive my car with computer?"

The full prompt history and a build timeline are in README.md under
"How this was made".
"""

from .config import Config
from .sim import Result, Scenario, run

__all__ = ["Config", "Result", "Scenario", "run"]
