#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The published comparison figure, drawn from the large corpus.

Reads `results/large_synthetic/summary.json`, so it regenerates from a committed
artefact in a second rather than from a two-hour benchmark run. The values are
the ones in that file; nothing is recomputed here.

The corpus 1 version of this comparison is drawn by omrbench.py. This one is
preferred for publication because the false positive rates behind it rest on
3,840 error-free sheets rather than 12 per generator.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# name in summary.json -> (label, which side of the point to put it on)
DETECTORS = {
    "noop":       ("No correction", "right"),
    "gated":      ("Gated pair HMM", "right"),
    "bruteforce": ("Global displacement scan", "right"),
    "fixedcost":  ("Fixed-cost affine alignment", "left"),
    "lcs":        ("Longest common subsequence", "left"),
}


def main() -> None:
    import matplotlib.pyplot as plt
    from figstyle import (SURFACE, INK, ACCENT, CONTEXT, ANNOT, frame, title,
                          footnote)

    src = os.path.join(_ROOT, "results", "large_synthetic", "summary.json")
    with open(src) as fh:
        data = json.load(fh)
    overall = data["overall"]

    pts = []
    for key, (label, side) in DETECTORS.items():
        if key not in overall:
            continue
        d = overall[key]
        pts.append((label, 100.0 * d["recovery"], d["fpr"], side,
                    key == "gated"))

    n = overall["gated"]["n"]
    clean = overall["gated"]["clean_sheets"]

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.20)
    frame(ax)

    for i, (label, x, y, side, hl) in enumerate(pts):
        ax.scatter([x], [y], s=190 if hl else 78,
                   color=ACCENT if hl else CONTEXT,
                   edgecolor=SURFACE, linewidth=1.8, zorder=4)
        near = [j for j, o in enumerate(pts)
                if j != i and abs(o[2] - y) < 0.05 and abs(o[1] - x) < 45]
        dy = 0
        if near:
            dy = 11 if x <= min(pts[j][1] for j in near) else -11
        dx = 13 if side == "right" else -13
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center",
                    fontsize=ANNOT, color=ACCENT if hl else INK)

    ax.set_xlabel("marks returned to wrongly-scored candidates  (%)", labelpad=9)
    ax.set_ylabel("false-positive rate\n(clean sheets wrongly corrected)", labelpad=9)
    title(fig, ax, "Recovery against false-positive rate")
    ax.set_xlim(-9, 116)
    ax.set_ylim(-0.09, 0.95)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 0.25, 0.50, 0.75])
    footnote(fig, f"{n:,} synthetic sheets · eleven candidate models · "
                  f"eighteen error mechanisms · five detectors on identical data\n"
                  f"False-positive rate is over {clean:,} error-free sheets; "
                  f"an observed 0.000 bounds the true rate below "
                  f"{100 * overall['gated']['fpr_ci_upper']:.2f}% (95%)")
    out = os.path.join(_ROOT, "results", "figures", "recovery_vs_fpr_large.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.22)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
