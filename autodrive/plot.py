"""Drive report: one figure per scenario showing what the system saw and did."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .sim import Result, Scenario  # noqa: E402

MODE_COLORS = {"engaged": "#2e7d32", "lat_override": "#f9a825", "fault": "#c62828", "off": "#9e9e9e"}


def save_report(scenario: Scenario, result: Result, path: str) -> None:
    log = result.log
    t = np.array(log["t"])
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[1, 1.4])
    fig.suptitle(f"{scenario.name}: {scenario.description}   [{'PASS' if result.passed else 'FAIL'}]",
                 fontsize=12)

    # Map with the driven path colored by system mode.
    ax = fig.add_subplot(grid[:, 0])
    r = scenario.route
    ax.plot(r.x, r.y, color="#bbbbbb", lw=8, solid_capstyle="round", label="lane")
    x, y, modes = np.array(log["x"]), np.array(log["y"]), log["mode"]
    for mode, color in MODE_COLORS.items():
        m = np.array([md == mode for md in modes])
        if m.any():
            ax.scatter(x[m], y[m], s=2, color=color, label=mode, zorder=3)
    for light in r.lights:
        lx, ly, _ = r.pose_at(light.s)
        ax.plot(lx, ly, marker="s", color="#c62828", ms=8, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Route (color = system mode, squares = traffic lights)")
    ax.legend(loc="best", markerscale=5, fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(t, np.array(log["speed"]) * 2.237, label="ego speed")
    ax.plot(t, np.array(log["speed_cap"]) * 2.237, "--", lw=1, label="planned speed cap")
    if not all(math.isnan(v) for v in log["lead_speed"]):
        ax.plot(t, np.array(log["lead_speed"]) * 2.237, lw=1, label="lead speed")
    ax.set_ylabel("mph")
    ax.legend(fontsize=8)
    _shade(ax, t, log)

    ax = fig.add_subplot(grid[1, 1], sharex=ax)
    ax.plot(t, log["accel"], label="accel")
    ax.axhline(-3.5, color="#999", lw=0.8, ls=":")
    ax.set_ylabel("m/s²")
    ax.legend(fontsize=8)
    _shade(ax, t, log)

    ax = fig.add_subplot(grid[2, 1], sharex=ax)
    gap = np.array([g if g < 200 else np.nan for g in log["gap"]])
    if np.isnan(gap).all():
        ax.text(0.5, 0.5, "no vehicle ahead", ha="center", va="center",
                transform=ax.transAxes, color="#777")
        ax.set_yticks([])
    else:
        ax.plot(t, gap, label="gap to lead")
        ax.legend(fontsize=8)
    ax.set_ylabel("m")
    _shade(ax, t, log)

    ax = fig.add_subplot(grid[3, 1], sharex=ax)
    ax.plot(t, np.array(log["lateral"]) * 100, label="lane-center error")
    ax.set_ylabel("cm")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    _shade(ax, t, log)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def _shade(ax, t, log) -> None:
    """Red bands where emergency braking fired."""
    aeb = np.array(log["aeb"])
    if aeb.any():
        ax.fill_between(t, 0, 1, where=aeb, color="#c62828", alpha=0.15,
                        transform=ax.get_xaxis_transform(), lw=0)
