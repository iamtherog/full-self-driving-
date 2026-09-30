"""Command line: run scenarios, print a scorecard, optionally save plots.

    python -m autodrive                      # all scenarios
    python -m autodrive city highway --plots out/
    python -m autodrive --attention          # also run the attention monitor (advisory only)
    python -m autodrive --dynamic            # drive the tire-slip vehicle model instead
"""

import argparse
import sys
from pathlib import Path

from . import scenarios
from .config import Config, use_dynamic_vehicle
from .sim import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autodrive", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="scenario",
                        help=f"any of: {', '.join(scenarios.ALL)} (default: all)")
    parser.add_argument("--plots", metavar="DIR", help="write a PNG report per scenario")
    parser.add_argument("--dynamic", action="store_true",
                        help="use the dynamic (tire-slip) vehicle model instead of the kinematic one")
    parser.add_argument("--attention", action="store_true",
                        help="run the attention monitor and list its advisories")
    args = parser.parse_args(argv)
    unknown = [n for n in args.names if n not in scenarios.ALL]
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")

    model = None
    if args.attention:
        from .attention import AttentionModel
        model = AttentionModel.load()

    config = use_dynamic_vehicle(Config()) if args.dynamic else Config()
    failed = 0
    for name in args.names or list(scenarios.ALL):
        scenario = scenarios.ALL[name]()
        monitor = None
        if model is not None:
            from .attention import AttentionMonitor
            monitor = AttentionMonitor(model)
        result = run(scenario, config, observer=monitor)
        print(f"\n{scenario.name} - {scenario.description}")
        print(result.summary())
        if monitor is not None:
            for flag in monitor.flags:
                status = "shown" if flag.delivered else f"withheld ({flag.withheld_because})"
                print(f"  {flag.t:6.2f} s  {flag.message:<62} {status}")
            if not monitor.flags:
                print("  Attention monitor      no flags")
        failed += not result.passed
        if args.plots:
            from .plot import save_report
            Path(args.plots).mkdir(parents=True, exist_ok=True)
            path = str(Path(args.plots) / f"{name}.png")
            save_report(scenario, result, path)
            print(f"  Report                 {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
