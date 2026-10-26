"""Common matplotlib style: fixed colour per entity, thin marks, recessive grid."""
from __future__ import annotations

import matplotlib.pyplot as plt

from machining import PLOT_DIR

C = {
    "cut": "#2a78d6",        # measured cutting run
    "cut_light": "#9ec5f4",
    "air": "#eb6834",        # measured air-cutting run (spindle only)
    "pc": "#1baf7a",         # cutting power Pc = Ptot - P0
    "pred": "#222222",       # model prediction
    "p1": "#2a78d6",         # Program I  (Vc = 150 m/min)
    "p2": "#eb6834",         # Program II (N = 1200 rpm)
    "ff": "#e87ba4",         # drilling feed force
    "band": "#eeeeec",
    "mark": "#6b6b66",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8a85",
        "axes.grid": True,
        "grid.color": "#e2e2df",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })


def mark_times(ax, marks: dict[str, float]) -> None:
    """Vertical dotted lines labelled with the characteristic instants."""
    for i, (label, t) in enumerate(marks.items()):
        ax.axvline(t, color=C["mark"], ls=":", lw=1)
        ax.annotate(label, (t, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -10 - 12 * (i % 2)), textcoords="offset points",
                    fontsize=9, color="#3a3a37")


def save(fig, name: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{name}.pdf")
    fig.savefig(PLOT_DIR / f"{name}.png")
    plt.close(fig)
