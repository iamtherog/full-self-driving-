"""Train, select, calibrate and measure the attention model.

Pipeline, every step deterministic in the plan:

1. Record ``plan.drives`` nominal drives (seeds ``first_seed ..``) through the real stack.
2. Split them by seed into train / calibration / test. Contiguous seed ranges,
   so no drive is ever used in more than one role.
3. Fit a Kneser-Ney model for each candidate order on the train split.
4. Select the order by sequential tests on the calibration split: a higher
   order replaces the incumbent only if it assigns a higher likelihood to
   significantly more calibration drives than not (Wald SPRT, one trial per
   drive). An undecided test keeps the simpler model.
5. Set the flag threshold so that, over the calibration split, familiar events
   are flagged no more often than ``plan.false_alarms_per_hour`` allows. (Event
   types never seen in training are flagged regardless; see
   :meth:`AttentionModel.flags`.)
6. Measure on the test split, which played no part in steps 3-5: held-out
   perplexity for every order, and the rate of flags on nominal driving
   (the false-alarm rate a driver would experience).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from . import corpus as corpus_module
from .events import TokenSpec
from .kneser_ney import KneserNey
from .model import AttentionModel
from .sprt import SPRT


@dataclass(frozen=True)
class Plan:
    drives: int = 600
    first_seed: int = 0
    orders: tuple[int, ...] = (1, 2, 3, 4, 5)
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)   # train, calibration, test
    false_alarms_per_hour: float = 1.0     # flag budget on nominal driving, used to set the threshold
    holdoff: float = 3.0
    sprt: SPRT = field(default_factory=SPRT)

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-9 or min(self.fractions) <= 0.0:
            raise ValueError("fractions must be positive and sum to 1")
        if self.false_alarms_per_hour <= 0.0:
            raise ValueError("false-alarm budget must be positive")
        if not self.orders or list(self.orders) != sorted(set(self.orders)):
            raise ValueError("orders must be non-empty and strictly increasing")


def surprisals(lm: KneserNey, tokens: Sequence[str]) -> list[float]:
    """Per-event surprisal with the same history the online monitor uses."""

    width = lm.order - 1
    return [lm.surprisal(tokens[max(0, i - width):i], token) for i, token in enumerate(tokens)]


def perplexity(lm: KneserNey, drives: Sequence[Sequence[str]]) -> float:
    total = sum(lm.sequence_logprob(d) for d in drives)
    count = sum(len(d) + 1 for d in drives)     # +1: each drive's end symbol is predicted too
    return math.exp(-total / count)


def threshold_at(values: Sequence[float], quantile: float) -> float:
    """Smallest threshold that flags at most ``floor((1 - quantile) * n)`` of ``values``.

    Surprisal takes few distinct values (common events repeat), so a plain
    percentile can land on a tie and flag far more than intended. Placing the
    threshold just above the cut value guarantees the bound.
    """

    ranked = sorted(values, reverse=True)
    allowed = math.floor((1.0 - quantile) * len(ranked))
    return math.nextafter(ranked[allowed], math.inf)


Split = tuple[list[corpus_module.Drive], list[corpus_module.Drive], list[corpus_module.Drive]]


def _split(items: Sequence[corpus_module.Drive], fractions: tuple[float, float, float]) -> Split:
    n = len(items)
    a = round(n * fractions[0])
    b = a + round(n * fractions[1])
    return list(items[:a]), list(items[a:b]), list(items[b:])


def train(plan: Plan | None = None, spec: TokenSpec | None = None) -> AttentionModel:
    plan = plan or Plan()
    spec = spec or TokenSpec()
    seeds = range(plan.first_seed, plan.first_seed + plan.drives)
    data = corpus_module.build(seeds, spec)
    train_set, calibration, test = _split(data.drives, plan.fractions)
    if not (train_set and calibration and test):
        raise ValueError(f"too few nominal drives ({len(data.drives)}) to fill three splits")

    def tokens(drives: Sequence[corpus_module.Drive]) -> list[tuple[str, ...]]:
        return [d.tokens for d in drives]

    models = {k: KneserNey.fit(tokens(train_set), order=k) for k in plan.orders}

    best = plan.orders[0]
    selection = []
    for k in plan.orders[1:]:
        def outcome(d: corpus_module.Drive, k: int = k, best: int = best) -> bool | None:
            gap = models[k].sequence_logprob(d.tokens) - models[best].sequence_logprob(d.tokens)
            return None if gap == 0.0 else gap > 0.0
        result = plan.sprt.run(outcome(d) for d in calibration)
        selection.append({"candidate": k, "incumbent": best, **asdict(result)})
        if result.verdict == "accept":
            best = k
    lm = models[best]

    familiar = [s for d in calibration for t, s in zip(d.tokens, surprisals(lm, d.tokens), strict=True)
                if t in lm.vocabulary]
    budget = plan.false_alarms_per_hour * spec.window / 3600.0     # flags per event
    threshold = threshold_at(familiar, 1.0 - budget)
    model = AttentionModel(lm=lm, spec=spec, threshold_bits=threshold, holdoff=plan.holdoff)

    def flagged(d: corpus_module.Drive) -> list[bool]:
        return [model.flags(t, s) for t, s in zip(d.tokens, surprisals(lm, d.tokens), strict=True)]

    test_flagged = [f for d in test for f in flagged(d)]
    test_flags = sum(test_flagged)
    hours = len(test_flagged) * spec.window / 3600.0
    metrics = {
        "vocabulary": len(lm.vocabulary),
        "stored_ngrams": len(lm.counts),
        "test_perplexity": {str(k): round(perplexity(m, tokens(test)), 6) for k, m in models.items()},
        "calibration_events": sum(len(d.tokens) for d in calibration),
        "test_events": len(test_flagged),
        "test_hours": round(hours, 4),
        "test_flags": test_flags,
        "test_flags_per_hour": round(test_flags / hours, 3),
        "test_drives_with_a_flag": sum(any(flagged(d)) for d in test),
        "test_novel_events": sum(t not in lm.vocabulary for d in test for t in d.tokens),
    }
    provenance = {
        "plan": asdict(plan),
        "corpus": {
            "seeds": [plan.first_seed, plan.first_seed + plan.drives],
            "kept": len(data.drives),
            "excluded": dict(sorted(data.excluded.items())),
            "tokens": data.tokens,
            "splits": {
                name: {"drives": len(part), "kinds": dict(sorted(Counter(d.kind for d in part).items()))}
                for name, part in (("train", train_set), ("calibration", calibration), ("test", test))
            },
        },
        "selection": selection,
        "selected_order": best,
    }
    return AttentionModel(lm=lm, spec=spec, threshold_bits=threshold, holdoff=plan.holdoff,
                          provenance=provenance, metrics=metrics)
