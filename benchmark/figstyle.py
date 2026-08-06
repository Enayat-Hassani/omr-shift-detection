"""Shared figure style for the published figures.

Both figures import these values, so type scale, palette, grid weight, margins
and caption placement are identical by construction. One accent colour marks
the recommended detector; every comparison mark is neutral grey. Bold is
reserved for the title.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams

SURFACE = "#fcfcfb"
INK     = "#3d3c39"    # annotations, axis labels
INK_HI  = "#1a1a19"    # titles only
MUTED   = "#9a978f"    # tick labels, footnotes
ACCENT  = "#2a78d6"    # the proposed method, nothing else
CONTEXT = "#b4b0a8"    # every comparison mark
GRID    = "#f0eeea"
LEADER  = "#d6d3cc"

TITLE, LABEL, TICK, ANNOT, FOOT = 12.0, 9.5, 9.0, 9.0, 8.0

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": ANNOT,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": LABEL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

def frame(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)

def title(fig, ax, text):
    ax.set_title(text, fontsize=TITLE, color=INK_HI, fontweight="bold",
                 loc="left", pad=18)

def footnote(fig, text, y=0.035):
    fig.text(0.10, y, text, ha="left", va="top", fontsize=FOOT,
             color=MUTED, linespacing=1.7, transform=fig.transFigure)

def leader(ax, xy, xytext, **kw):
    ax.annotate("", xy=xy, xytext=xytext, textcoords=kw.pop("tc", "data"),
                arrowprops=dict(arrowstyle="-", color=LEADER, lw=0.9,
                                shrinkA=2, shrinkB=6))
