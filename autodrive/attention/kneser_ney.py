"""Interpolated modified Kneser-Ney n-gram model over string tokens.

References:
    R. Kneser and H. Ney, "Improved backing-off for m-gram language modeling",
    ICASSP 1995.
    S. F. Chen and J. Goodman, "An empirical study of smoothing techniques for
    language modeling", Computer Speech and Language 13(4), 1999. Section 3.5
    (interpolated Kneser-Ney) and equation 26 (the three modified discounts).

The model stores exactly one table, the raw counts of top-order n-grams. Every
lower order is derived from it: at order ``k < n`` the count of ``(h, w)`` is
the number of distinct tokens seen immediately before ``h w`` (its continuation
count). Storing only the raw table keeps a single source of truth, so the
serialised model is small, and loading it re-derives the lower orders, the
discounts and the normalisers deterministically.

For a context ``h`` seen at level ``k``, with count ``c(h w)`` for each
successor ``w`` and total ``c(h) = sum_w c(h w)``:

    P_k(w | h) = ( max(c(h w) - D_k(c(h w)), 0)  +  gamma_k(h) * P_{k-1}(w | h') ) / c(h)
    gamma_k(h) = D_k1 * N1(h) + D_k2 * N2(h) + D_k3 * N3+(h)

where ``h'`` drops the oldest token of ``h`` and ``N_i(h)`` counts successors
seen exactly ``i`` times (``i or more`` for ``N3+``). The recursion starts from
the uniform distribution over the predictable vocabulary. A context never seen
at level ``k`` carries no evidence of its own, so ``P_k = P_{k-1}`` there.

Normalisation. If ``P_{k-1}`` sums to one over the vocabulary ``V``, then

    sum_w P_k(w | h) = ( c(h) - sum_w D(c(h w)) + gamma(h) ) / c(h) = 1,

because ``gamma(h)`` is exactly the discounted mass ``sum_w D(c(h w))``, and
``D(c) <= c`` holds for every discount class (``D1 <= 1``, ``D2 <= 2``,
``D3+ <= 3``). By induction every level is a proper distribution over ``V``.
``tests/test_attention.py`` checks this numerically for seen and unseen
contexts.

Strict positivity. Every discount is floored at :data:`MIN_DISCOUNT`, so a seen
context always reserves some mass for the level below, and the uniform base
gives every token in ``V`` (including ``<unk>``) a positive share. Surprisal is
therefore always finite: an unseen event is scored as very surprising, never as
impossible. A monitor that divides by zero on the first truly novel situation
would fail exactly when it matters.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
MAX_ORDER = 6

#: Lower bound on every discount; see "Strict positivity" in the module docstring.
MIN_DISCOUNT = 0.05

#: Used when counts-of-counts are too sparse for the Chen-Goodman estimate
#: (a zero denominator). These are the customary fixed values.
FALLBACK_DISCOUNTS = (0.5, 0.75, 0.9)


@dataclass(frozen=True)
class Discounts:
    d1: float
    d2: float
    d3: float

    def __call__(self, count: int) -> float:
        if count <= 0:
            return 0.0
        return self.d1 if count == 1 else self.d2 if count == 2 else self.d3


def estimate_discounts(counts: Iterable[int]) -> Discounts:
    """Modified Kneser-Ney discounts from counts-of-counts (Chen and Goodman eq. 26).

        Y = n1 / (n1 + 2 n2),  D1 = 1 - 2Y n2/n1,  D2 = 2 - 3Y n3/n2,  D3+ = 3 - 4Y n4/n3

    Each estimate needs a non-zero denominator; where the counts cannot supply
    one, that discount falls back to its customary fixed value. Results are
    clamped to ``[MIN_DISCOUNT, class]`` so the normalisation and positivity
    arguments in the module docstring hold for any input.
    """

    n = Counter(c for c in counts if 1 <= c <= 4)
    n1, n2, n3, n4 = n[1], n[2], n[3], n[4]
    f1, f2, f3 = FALLBACK_DISCOUNTS
    if n1 == 0 or n2 == 0:
        d1, d2, d3 = f1, f2, f3
    else:
        y = n1 / (n1 + 2 * n2)
        d1 = 1.0 - 2.0 * y * n2 / n1
        d2 = 2.0 - 3.0 * y * n3 / n2
        d3 = 3.0 - 4.0 * y * n4 / n3 if n3 else f3
    return Discounts(
        d1=min(max(d1, MIN_DISCOUNT), 1.0),
        d2=min(max(d2, MIN_DISCOUNT), 2.0),
        d3=min(max(d3, MIN_DISCOUNT), 3.0),
    )


@dataclass
class _Level:
    """Successor counts for every context of one length, plus derived statistics."""

    table: dict[tuple[str, ...], dict[str, int]]
    discounts: Discounts
    totals: dict[tuple[str, ...], int]
    reserved: dict[tuple[str, ...], float]    # gamma(h): mass passed to the level below

    @classmethod
    def build(cls, table: dict[tuple[str, ...], dict[str, int]]) -> _Level:
        discounts = estimate_discounts(c for successors in table.values() for c in successors.values())
        totals = {h: sum(s.values()) for h, s in table.items()}
        # fsum is correctly rounded, so gamma does not depend on the order the
        # successors were inserted in. A plain sum can differ in the last bit
        # between a freshly trained model and the same model loaded from disk.
        reserved = {h: math.fsum(discounts(c) for c in s.values()) for h, s in table.items()}
        return cls(table, discounts, totals, reserved)


class KneserNey:
    """An order-``n`` model. Construct with :meth:`fit` or :meth:`from_dict`."""

    def __init__(self, order: int, counts: Mapping[tuple[str, ...], int]):
        """Build from raw counts of ``order``-grams (padded with BOS/EOS as :meth:`fit` does)."""

        if not 1 <= order <= MAX_ORDER:
            raise ValueError(f"order must lie in [1, {MAX_ORDER}], got {order}")
        if not counts:
            raise ValueError("cannot build a model from no counts")
        for gram, count in counts.items():
            if len(gram) != order or count < 1:
                raise ValueError(f"invalid count entry {gram!r}: {count}")
        self.order = order
        self.counts: dict[tuple[str, ...], int] = dict(sorted(counts.items()))   # canonical order

        top: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
        for gram, count in self.counts.items():
            top[gram[:-1]][gram[-1]] = count
        tables = [dict(top)]
        # Continuation counts: level k counts (h, w) by distinct left extensions v
        # of (v, h, w) at level k + 1.
        for _ in range(order - 1):
            lower: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
            for context, successors in tables[-1].items():
                shorter = context[1:]
                for token in successors:
                    lower[shorter][token] = lower[shorter].get(token, 0) + 1
            tables.append(dict(lower))
        tables.reverse()  # tables[k] is keyed by contexts of length k
        self._levels = [_Level.build(t) for t in tables]

        #: Tokens the model can predict. BOS is padding and is never predicted.
        self.vocabulary: frozenset[str] = frozenset(gram[-1] for gram in self.counts) | {UNK}
        self._uniform = 1.0 / len(self.vocabulary)

    # ----------------------------------------------------------------- fitting

    @classmethod
    def fit(cls, sequences: Iterable[Sequence[str]], order: int) -> KneserNey:
        """Count ``order``-grams over sequences padded with ``order - 1`` BOS and one EOS."""

        counts: Counter[tuple[str, ...]] = Counter()
        for sequence in sequences:
            for token in sequence:
                if token in (BOS, EOS, UNK):
                    raise ValueError(f"reserved symbol {token!r} in training data")
            padded = [BOS] * (order - 1) + list(sequence) + [EOS]
            for i in range(order - 1, len(padded)):
                counts[tuple(padded[i - order + 1:i + 1])] += 1
        return cls(order, counts)

    # ------------------------------------------------------------- inference

    def _normalise(self, context: Sequence[str]) -> tuple[str, ...]:
        width = self.order - 1
        if width == 0:
            return ()
        known = [t if t in self.vocabulary or t == BOS else UNK for t in context[-width:]]
        return tuple([BOS] * (width - len(known)) + known)

    def prob(self, context: Sequence[str], token: str) -> float:
        """P(token | context). Context is left-padded with BOS; unknown tokens map to UNK."""

        h = self._normalise(context)
        w = token if token in self.vocabulary else UNK
        p = self._uniform
        for k, level in enumerate(self._levels):
            ctx = h[len(h) - k:] if k else ()
            successors = level.table.get(ctx)
            if successors is None:
                continue      # unseen context: keep the lower-order estimate
            count = successors.get(w, 0)
            p = (max(count - level.discounts(count), 0.0) + level.reserved[ctx] * p) / level.totals[ctx]
        return p

    def surprisal(self, context: Sequence[str], token: str) -> float:
        """Information content of ``token`` after ``context``, in bits."""

        return -math.log2(self.prob(context, token))

    def sequence_logprob(self, sequence: Sequence[str]) -> float:
        """Natural-log probability of a whole sequence, including its EOS."""

        history: list[str] = []
        total = 0.0
        for token in [*sequence, EOS]:
            total += math.log(self.prob(history, token))
            history.append(token)
        return total

    @property
    def discounts(self) -> tuple[Discounts, ...]:
        """Per level, indexed by context length."""

        return tuple(level.discounts for level in self._levels)

    # --------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "counts": [[list(gram), count] for gram, count in sorted(self.counts.items())],
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> KneserNey:
        return cls(int(blob["order"]), {tuple(g): int(c) for g, c in blob["counts"]})
