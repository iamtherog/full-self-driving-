"""Figure for the README: event surprisal over time for each named scenario.

Small multiples on a shared surprisal axis, one panel per scenario. Identity is
carried by shape and label as well as color: a line for surprisal, open circles
for event types never seen in training, a dashed rule for the threshold, a solid
rule for the incident, triangles for advisories actually shown to the driver.

Typeset in IBM Plex Sans / Mono (Google Fonts, SIL OFL; files in docs/fonts).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .evaluate import Row
from .model import AttentionModel

FONTS = Path(__file__).resolve().parents[2] / "docs" / "fonts"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES, NOVEL = "#2a78d6", "#eb6834"     # reference categorical slots 1 and 2, validated together
CAP = 20.0                               # bits; novel events are drawn at the cap, above the threshold


def _use_web_fonts() -> None:
    for path in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})


def save_figure(rows: list[Row], model: AttentionModel, path: str | Path) -> None:
    _use_web_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True, constrained_layout=True,
                             facecolor=SURFACE)
    for ax, row in zip(axes.flat, rows, strict=True):
        monitor_scores = _scores(row, model)
        ts = [t for t, _, _ in monitor_scores]
        bits = [min(b, CAP) for _, b, _ in monitor_scores]
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.step(ts, bits, where="post", color=SERIES, lw=2, label="event surprisal")
        novel = [(t, CAP) for t, _, n in monitor_scores if n]
        if novel:
            ax.plot(*zip(*novel, strict=True), "o", ms=8, mfc="none", mec=NOVEL, mew=2,
                    label="event type never seen in training")
        ax.axhline(model.threshold_bits, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="flag threshold")
        if row.incident_at is not None:
            ax.axvline(row.incident_at, color=INK, lw=1.2, label="incident (AEB or fault)")
        for flag in row.delivered:
            ax.plot(flag.t, CAP + 2.2, "v", ms=9, color=INK, label="advisory shown")
        ax.set_title(row.scenario, loc="left", fontsize=11, fontweight="semibold", color=INK)
        ax.set_ylim(0, CAP + 4)
        ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("surprisal (bits)")
    for ax in axes[1, :]:
        ax.set_xlabel("time (s)")

    handles, labels = {}, []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    fig.legend([handles[lab] for lab in labels], labels, loc="outside lower center", ncol=len(labels),
               frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Attention monitor on the named scenarios (none were used in training)",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold", color=INK)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def _scores(row: Row, model: AttentionModel) -> list[tuple[float, float, bool]]:
    """(time the event completed, surprisal, novel) for every event the monitor scored."""

    return [(e.t_end, bits, e.token not in model.lm.vocabulary) for e, bits in row.scores]
