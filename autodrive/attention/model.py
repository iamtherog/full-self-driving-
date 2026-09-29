"""The trained attention model and its on-disk artifact.

An artifact is one canonical JSON document: the n-gram counts, the token spec
it was trained with, the calibrated alert threshold, the provenance needed to
retrain it (seeds, splits, exclusions, model-selection record) and the
measurements taken at training time. A SHA-256 digest covers all of it.

Loading refuses an artifact whose digest does not match its contents, or whose
token spec differs from the one in this build. The digest detects corruption
and accidental edits. It is not a signature: anyone can recompute it, so it
says nothing about who produced the file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .events import TokenSpec
from .kneser_ney import KneserNey

FORMAT = "autodrive.attention.v1"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "attention.json"


class ArtifactError(ValueError):
    """The artifact is unreadable, corrupted, or trained for a different token spec."""


def canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class AttentionModel:
    lm: KneserNey
    spec: TokenSpec
    threshold_bits: float          # surprisal at or above which an event is flagged
    holdoff: float                 # s between delivered advisories
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def flags(self, token: str, surprisal_bits: float) -> bool:
        """Whether an event deserves the driver's attention.

        Two routes, for two kinds of evidence:

        * A familiar event type in an unfamiliar order is judged by probability:
          flagged when its surprisal reaches the calibrated threshold.
        * An event type that never occurred in nominal training data is flagged
          unconditionally. The model has no evidence about it; the probability it
          assigns comes from the smoothing floor, not from data. Worse, once one
          novel event enters the context the model backs off, and a second novel
          event can score as *less* surprising than the first. Novelty is the
          better test there.
        """

        return token not in self.lm.vocabulary or surprisal_bits >= self.threshold_bits

    def body(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "token_spec": asdict(self.spec),
            "token_spec_digest": self.spec.digest(),
            "model": self.lm.to_dict(),
            "threshold_bits": self.threshold_bits,
            "holdoff": self.holdoff,
            "provenance": dict(self.provenance),
            "metrics": dict(self.metrics),
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical(self.body())).hexdigest()

    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {**self.body(), "sha256": self.digest()}
        target.write_text(json.dumps(document, sort_keys=True, indent=1, allow_nan=False) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH, *, spec: TokenSpec | None = None) -> AttentionModel:
        expected_spec = spec or TokenSpec()
        try:
            blob = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read attention model {path}: {exc}") from exc
        if blob.get("format") != FORMAT:
            raise ArtifactError(f"unsupported format {blob.get('format')!r}, expected {FORMAT!r}")
        stored = blob.pop("sha256", None)
        if stored != hashlib.sha256(canonical(blob)).hexdigest():
            raise ArtifactError("attention model contents do not match their SHA-256 digest")
        if blob["token_spec_digest"] != expected_spec.digest():
            raise ArtifactError(
                "attention model was trained with a different token spec; "
                "retrain with `python -m autodrive.attention train`"
            )
        model = cls(
            lm=KneserNey.from_dict(blob["model"]),
            spec=expected_spec,
            threshold_bits=float(blob["threshold_bits"]),
            holdoff=float(blob["holdoff"]),
            provenance=blob["provenance"],
            metrics=blob["metrics"],
        )
        if model.digest() != stored:   # round trip must be exact, or the artifact means something else
            raise ArtifactError("attention model does not round-trip to the same digest")
        return model
