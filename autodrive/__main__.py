"""Command line: run scenarios, print a scorecard, optionally save plots.

    python -m autodrive                      # all scenarios
    python -m autodrive city highway --plots out/
"""

import argparse
import os
import sys

from . import scenarios
from .sim import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autodrive", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="scenario",
                        help=f"any of: {', '.join(scenarios.ALL)} (default: all)")
    parser.add_argument("--plots", metavar="DIR", help="write a PNG report per scenario")
    args = parser.parse_args(argv)
    unknown = [n for n in args.names if n not in scenarios.ALL]
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")

    failed = 0
    for name in args.names or list(scenarios.ALL):
        scenario = scenarios.ALL[name]()
        result = run(scenario)
        print(f"\n{scenario.name} - {scenario.description}")
        print(result.summary())
        failed += not result.passed
        if args.plots:
            from .plot import save_report
            os.makedirs(args.plots, exist_ok=True)
            path = os.path.join(args.plots, f"{name}.png")
            save_report(scenario, result, path)
            print(f"  Report                 {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
