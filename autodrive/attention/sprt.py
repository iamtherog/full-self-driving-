"""Wald's sequential probability ratio test on a Bernoulli rate, used to pick a model order.

Reference: A. Wald, "Sequential tests of statistical hypotheses", Annals of
Mathematical Statistics 16(2), 1945.

The question "is model A better than model B?" is posed as a test on the rate
at which A beats B on independent trials: H0 is ``p = p0`` (a coin flip) and H1
is ``p = p1`` (A usually wins). The log-likelihood ratio is accumulated one
trial at a time and the test stops at the first crossing of

    upper = log((1 - beta) / alpha)     accept H1
    lower = log(beta / (1 - alpha))     accept H0

which bounds the error rates by ``alpha / (1 - beta)`` and ``beta / (1 - alpha)``
(Wald's inequalities). If the trials run out first, the result is
``undecided`` and the caller keeps the simpler model.

Independence matters, and this is where the choice of trial comes in. Per-token
comparisons within one drive are serially correlated, which silently
invalidates the error bounds. The trainer therefore uses one trial per
*drive*: drives are generated from independent seeds, so trials are i.i.d. by
construction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["accept", "reject", "undecided"]


@dataclass(frozen=True)
class SPRTResult:
    verdict: Verdict
    trials: int
    successes: int
    log_ratio: float


@dataclass(frozen=True)
class SPRT:
    p0: float = 0.5
    p1: float = 0.8
    alpha: float = 0.01
    beta: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < self.p1 < 1.0:
            raise ValueError("require 0 < p0 < p1 < 1")
        if not (0.0 < self.alpha < 0.5 and 0.0 < self.beta < 0.5):
            raise ValueError("alpha and beta must lie in (0, 0.5)")

    @property
    def upper(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    def run(self, outcomes: Iterable[bool | None]) -> SPRTResult:
        """Consume outcomes until a boundary is crossed. ``None`` is a tie and carries no evidence."""

        win = math.log(self.p1 / self.p0)
        loss = math.log((1.0 - self.p1) / (1.0 - self.p0))
        llr, trials, successes = 0.0, 0, 0
        for outcome in outcomes:
            if outcome is None:
                continue
            trials += 1
            successes += outcome
            llr += win if outcome else loss
            if llr >= self.upper:
                return SPRTResult("accept", trials, successes, llr)
            if llr <= self.lower:
                return SPRTResult("reject", trials, successes, llr)
        return SPRTResult("undecided", trials, successes, llr)
