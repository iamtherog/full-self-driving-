"""Command line for the attention monitor.

    python -m autodrive.attention train              # retrain and write the shipped artifact
    python -m autodrive.attention verify             # check the artifact: digest, reproducibility, metrics
    python -m autodrive.attention report             # score the named scenarios
    python -m autodrive.attention report --figure docs/attention.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import Row, score_all
from .model import DEFAULT_PATH, ArtifactError, AttentionModel
from .training import Plan, train


def _print_rows(rows: list[Row]) -> None:
    print(f"{'scenario':<16} {'incident':>9} {'first warning':>14} {'lead':>6}  "
          f"{'flags':>5} {'shown':>5}  first message shown")
    for row in rows:
        incident = f"{row.incident_at:.2f} s" if row.incident_at is not None else "none"
        warning = f"{row.first_warning_at:.2f} s" if row.first_warning_at is not None else "-"
        lead = f"{row.lead_time:.2f}" if row.lead_time is not None else "-"
        shown = row.delivered[0].message if row.delivered else "-"
        print(f"{row.scenario:<16} {incident:>9} {warning:>14} {lead:>6}  "
              f"{len(row.flags):>5} {len(row.delivered):>5}  {shown}")


def cmd_train(args: argparse.Namespace) -> int:
    model = train(Plan(drives=args.drives))
    path = model.save(args.output)
    print(json.dumps({"selected_order": model.provenance["selected_order"],
                      "threshold_bits": model.threshold_bits, **model.metrics}, indent=1))
    print(f"wrote {path}  sha256={model.digest()}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        model = AttentionModel.load(args.model)
    except ArtifactError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    print(f"ok    digest and token spec match ({model.digest()[:16]}...)")
    plan = Plan(**{k: v for k, v in model.provenance["plan"].items() if k in ("drives", "first_seed")})
    print(f"...   retraining from recorded provenance ({plan.drives} drives)")
    rebuilt = train(plan)
    if rebuilt.digest() != model.digest():
        print("FAIL  retrained model differs from the artifact", file=sys.stderr)
        return 1
    print("ok    retraining reproduces the artifact bit for bit")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    model = AttentionModel.load(args.model)
    rows = score_all(model)
    _print_rows(rows)
    if args.figure:
        from .figure import save_figure
        save_figure(rows, model, args.figure)
        print(f"wrote {args.figure}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autodrive.attention", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="retrain from randomized nominal drives")
    p.add_argument("--drives", type=int, default=Plan().drives)
    p.add_argument("--output", type=Path, default=DEFAULT_PATH)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("verify", help="check digest, token spec and bit-for-bit reproducibility")
    p.add_argument("--model", type=Path, default=DEFAULT_PATH)
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("report", help="score the named scenarios")
    p.add_argument("--model", type=Path, default=DEFAULT_PATH)
    p.add_argument("--figure", type=Path)
    p.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
