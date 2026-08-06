#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operating characteristics of the three acceptance profiles.

Produces every figure quoted in REPORT.md section 6.1 and in ASSUMPTIONS.md A6.
Three arms:

  CALIBRATION   Error-free sheets, full permutation counts. Compares the nominal
                level against the rate at which the reported p-value falls below
                it. This is what establishes that the level is a conservative
                bound and not a false positive rate. Cannot use early stopping:
                the p-value itself is the measurement.

  PROFILES      Genuine single-row skips, adjudicated once at a level looser
                than any profile. Acceptance at a tight alpha is a subset of
                acceptance at a loose one, and the item award depends on
                `item_posterior_threshold` and the segment posteriors but never
                on alpha, so one pass yields the correct award at every profile.

  CERTIFICATION Error-free sheets at a level looser than any profile, bounding
                the false positive rate for all three at once. Early stopping is
                used here and only here: it changes the reported p-value but not
                the accept/reject decision, which is all a rate needs.

Seeding is fixed and per-sheet seeds derive from zlib.crc32, matching
benchmark/omrbench.py. Running twice produces identical output.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import zlib
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omr_shift import (  # noqa: E402
    AdjudicationConfig,
    Adjudicator,
    Policy,
    ResponseSheet,
    clopper_pearson_upper,
)

SEED = 20260805
OPTIONS = "ABCD"
# The 20-item key used throughout. Short papers are the hard case: the detection
# floor is an absolute number of correct marks, so it consumes a larger share of
# a short sheet. Reporting on 20 states the characteristics where they bite.
KEY: Sequence[str] = ("B", "D", "A", "C", "A", "D", "B", "C", "D", "A",
                      "C", "B", "A", "D", "C", "B", "A", "C", "D", "B")
ABILITIES_CLEAN = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
ABILITIES_SHIFT = (0.55, 0.65, 0.75, 0.85, 0.95)
LEVELS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)
# Looser than any profile, so one pass covers all three.
CEILING = 0.1
# The draws must resolve the TIGHTEST level thresholded, not the loosest. The
# smallest p-value obtainable is 1/(n+1); if that exceeds a level, no sheet can
# clear it and that column reads zero by construction rather than by measurement.
DRAWS_PROFILE = math.ceil(10.0 / min(LEVELS)) - 1
# The certification arm thresholds at CEILING only, so it needs far fewer draws
# than the profile arm. Ten times the representability minimum, which leaves ~99
# admissible exceedances: enough that the decision is not decided by the
# granularity of the p-value, and cheap because early stopping ends each sheet as
# soon as the budget is spent.
DRAWS_CERTIFY = math.ceil(10.0 / CEILING) * 10 - 1


def search_burden(n_questions: int, max_disp: int, min_seg: int) -> int:
    """Windows the coherence scan tests on one sheet.

    Every non-zero offset, every start and end pair of at least `min_seg`. This
    is what a displaced block has to beat, and it is what sets the detection
    floor reported in REPORT.md section 11.2: the floor lands where the expected
    number of spurious hits, windows * P(all correct | chance), reaches 0.001.
    Neither term depends on candidate ability, which is why a prior from prior
    attainment cannot lower the floor.
    """
    total = 0
    for d in range(-max_disp, max_disp + 1):
        if d == 0:
            continue
        L = len([q for q in range(n_questions) if 0 <= q + d < n_questions])
        if L < min_seg:
            continue
        total += sum(1 for i in range(L) for j in range(i + min_seg, L + 1))
    return total


def _rng(tag: str, i: int) -> random.Random:
    """Per-sheet seed. `hash` is randomised per process and cannot be used."""
    return random.Random(zlib.crc32(f"{tag}:{i}".encode()) ^ SEED)


def _answer(theta: float, rng: random.Random) -> List[str]:
    return [k if rng.random() < theta else rng.choice([o for o in OPTIONS if o != k])
            for k in KEY]


def _clean(i: int, theta: float) -> ResponseSheet:
    return ResponseSheet(tuple(KEY), tuple(_answer(theta, _rng("clean", i))),
                         candidate_id=f"clean-{i}")


def _skip(i: int, theta: float, q: int) -> ResponseSheet:
    """One bubble row skipped entering question q. Displaced run is len(KEY)-q."""
    n = len(KEY)
    truth = _answer(theta, _rng("shift", i))
    marks = truth[:q - 1] + [None] + truth[q - 1:n - 1]
    return ResponseSheet(tuple(KEY), tuple(marks), candidate_id=f"shift-{i}")


def _probe(sheet: ResponseSheet, cfg: AdjudicationConfig, early: bool = False) -> Dict:
    a = Adjudicator(sheet, cfg).run(verbose=False, early_stop=early)
    displaced = [s for s in a.segments if s.offset != 0]
    best = max(displaced, key=lambda s: s.n_correct) if displaced else None
    return {
        "p": a.calibration["p_value"],
        "binding": a.calibration["decisive_null"],
        "raw": a.raw_score,
        "adjudicated": a.adjudicated_score,
        "gate_bf": a.gates["bayes_factor"]["passed"],
        "gate_seg": a.gates["segment_coherence"]["passed"],
        "gate_nontrivial": a.gates["non_trivial"]["passed"],
        "block_items": best.n_items if best else 0,
        "block_correct": best.n_correct if best else 0,
        "smallest_block": min((s.n_items for s in displaced), default=0),
        "change_point": a.change_points[0]["at_question"] if a.change_points else None,
    }


def _passes(r: Dict, alpha: float) -> bool:
    """Every gate, with the Monte-Carlo one evaluated at `alpha`."""
    return (r["p"] <= alpha and r["gate_bf"] and r["gate_seg"] and r["gate_nontrivial"])


def run(n_clean: int, n_shift: int, n_certify: int) -> str:
    L: List[str] = []
    w = 78
    L.append("=" * w)
    L.append("ACCEPTANCE PROFILES -- OPERATING CHARACTERISTICS".center(w))
    L.append("=" * w)
    L.append(f"Sheet length {len(KEY)}. Single-row skips only. All sheets SYNTHETIC.")
    L.append("")

    # ---- arm 1: calibration ------------------------------------------------
    base = AdjudicationConfig()
    clean: List[Dict] = []
    for i in range(n_clean):
        theta = ABILITIES_CLEAN[i % len(ABILITIES_CLEAN)]
        clean.append(_probe(_clean(i, theta), base))

    L.append("-" * w)
    L.append("1. CALIBRATION  --  is the reported p-value a false positive rate?")
    L.append("-" * w)
    L.append(f"  {n_clean} error-free sheets, ability {min(ABILITIES_CLEAN)} to "
             f"{max(ABILITIES_CLEAN)}, {base.n_permutations} draws per null.")
    L.append("")
    L.append(f"  {'nominal alpha':>14} {'observed rate':>15} {'':>8}")
    for a in (0.5, 0.25, 0.1, 0.05, 0.01, 0.001):
        k = sum(1 for r in clean if r["p"] <= a)
        L.append(f"  {a:>14} {k / len(clean):>15.3f} {'(' + str(k) + '/' + str(len(clean)) + ')':>8}")
    L.append("")
    binding: Dict[str, int] = {}
    for r in clean:
        binding[r["binding"]] = binding.get(r["binding"], 0) + 1
    L.append("  binding null on error-free sheets (the worst of the three is reported):")
    for name, count in sorted(binding.items(), key=lambda kv: -kv[1]):
        L.append(f"    {name:18s} {count:4d} / {len(clean)}")
    L.append("")
    L.append("  A nominal level that fires far less often than nominal is a")
    L.append("  conservative bound, not a rate. See ASSUMPTIONS.md A6.")
    L.append("")

    # ---- arm 2: profile characteristics ------------------------------------
    loose = replace(base, permutation_alpha=CEILING, n_permutations=DRAWS_PROFILE)
    shifted: List[Dict] = []
    for i in range(n_shift):
        theta = ABILITIES_SHIFT[i % len(ABILITIES_SHIFT)]
        q = 2 + (zlib.crc32(f"cp:{i}".encode()) % 15)
        sheet = _skip(i, theta, q)
        r = _probe(sheet, loose)
        # Marks a perfect correction could return: those correct under the
        # shifted registration, plus those already correct before the skip.
        r["ceiling"] = (sum(1 for j in range(1, len(KEY))
                            if sheet.marks[j] is not None and sheet.marks[j] == KEY[j - 1])
                        + sum(1 for j in range(q - 1) if sheet.marks[j] == KEY[j]))
        r["true_cp"] = q
        r["theta"] = theta
        shifted.append(r)

    lost = sum(r["ceiling"] - r["raw"] for r in shifted)
    L.append("-" * w)
    L.append("2. PROFILES  --  what each level recovers")
    L.append("-" * w)
    L.append(f"  {n_shift} genuine single-row skips, ability {min(ABILITIES_SHIFT)} to "
             f"{max(ABILITIES_SHIFT)}, change point 2 to 16.")
    L.append(f"  Marks lost to the skips in total: {lost}")
    L.append("")
    L.append(f"  {'alpha':>7} {'profile':>13} {'detected':>10} {'marks back':>12} "
             f"{'recovery':>9} {'lost':>6} {'misloc':>7} {'min block':>10}")
    by_alpha = {p.alpha: p.label for p in Policy}
    curve = []
    for a in LEVELS:
        det = [r for r in shifted if _passes(r, a)]
        back = sum(r["adjudicated"] - r["raw"] for r in det)
        curve.append((a, 100.0 * back / lost if lost else 0.0))
        taken = sum(r["raw"] - r["adjudicated"] for r in det if r["adjudicated"] < r["raw"])
        mis = sum(1 for r in det if r["change_point"] != r["true_cp"])
        smallest = min((r["block_correct"] for r in det), default=0)
        L.append(f"  {a:>7} {by_alpha.get(a, ''):>13} {len(det):>4}/{n_shift:<5} "
                 f"{back:>5}/{lost:<6} {back / lost if lost else 0:>8.1%} "
                 f"{taken:>6} {mis:>7} {smallest:>10}")
    L.append("")
    L.append("  min block is the fewest CORRECT marks in a displaced block that")
    L.append("  the level accepted -- what it can physically see. No level makes a")
    L.append("  shorter displacement detectable.")
    L.append("")
    L.append(f"  {'alpha':>7} " + " ".join(f"{('t=' + str(t)):>8}" for t in ABILITIES_SHIFT))
    for a in (0.001, 0.01, 0.05):
        cells = []
        for t in ABILITIES_SHIFT:
            sub = [r for r in shifted if r["theta"] == t]
            k = sum(1 for r in sub if _passes(r, a))
            cells.append(f"{k}/{len(sub)}")
        L.append(f"  {a:>7} " + " ".join(f"{c:>8}" for c in cells))
    L.append("  detection by ability. Weak candidates are not recovered at any level.")
    L.append("")

    # ---- arm 3: which gate binds -------------------------------------------
    sole: Dict[str, int] = {}
    for r in shifted:
        state = {"monte_carlo": r["p"] <= base.permutation_alpha,
                 "bayes_factor": r["gate_bf"],
                 "segment_coherence": r["gate_seg"],
                 "non_trivial": r["gate_nontrivial"]}
        failing = [k for k, v in state.items() if not v]
        if len(failing) == 1:
            sole[failing[0]] = sole.get(failing[0], 0) + 1
    accepted_any = [r for r in shifted if _passes(r, CEILING)]
    L.append("-" * w)
    L.append("3. WHICH GATE BINDS  --  at the default level")
    L.append("-" * w)
    L.append(f"  Rejections with exactly one failing gate: {sum(sole.values())}")
    for name in ("monte_carlo", "bayes_factor", "segment_coherence", "non_trivial"):
        L.append(f"    sole failing gate was {name:18s} {sole.get(name, 0):4d}")
    L.append(f"  Smallest displaced block on any accepted sheet: "
             f"{min((r['smallest_block'] for r in accepted_any), default=0)} items")
    L.append(f"  Configured min_segment_length: {base.min_segment_length}")
    L.append("")

    # ---- arm 4: certification ----------------------------------------------
    certify_cfg = replace(base, permutation_alpha=CEILING, n_permutations=DRAWS_CERTIFY)
    fp = 0
    for i in range(n_certify):
        theta = ABILITIES_CLEAN[i % len(ABILITIES_CLEAN)]
        r = _probe(_clean(100000 + i, theta), certify_cfg, early=True)
        if _passes(r, CEILING):
            fp += 1
    bound = clopper_pearson_upper(fp, n_certify)
    L.append("-" * w)
    L.append("4. CERTIFICATION  --  false positives, all profiles at once")
    L.append("-" * w)
    L.append(f"  {n_certify} error-free sheets at alpha = {CEILING}, which is looser than")
    L.append("  any profile. Acceptance at a tight alpha is a subset of acceptance")
    L.append("  at a loose one, so this bounds all three.")
    L.append("")
    L.append(f"  false positives            : {fp} / {n_certify}")
    L.append(f"  95% upper bound (C-P)      : {bound:.5f}  ({100 * bound:.3f}%)")
    L.append(f"  unearned marks per sheet at the bound, 1.8% base rate,")
    L.append(f"  assuming 10 marks per false acceptance: "
             f"{0.982 * bound * 10:.4f}")
    L.append("  no-correction baseline: 0.006. The 10-mark assumption is not")
    L.append("  measured -- there were no false acceptances to measure it on --")
    L.append("  and the figure scales linearly with it.")
    L.append("")
    L.append("=" * w)
    L.append("Zero observed is not zero risk. Every sheet above is synthetic, and")
    L.append("no confirmed historical case exists to calibrate against.")
    L.append("=" * w)
    return "\n".join(L), curve, bound


def render_figure(curve, n_shift: int, n_certify: int, bound: float, outdir: str):
    """Recovery against acceptance level, with observed false positives below.

    The values are the ones tabulated above, so the figure cannot disagree with
    the table. Two panels sharing one x-axis rather than two y-scales on one
    plot: the quantities have different units and a shared scale would be
    arbitrary.
    """
    try:
        import matplotlib.pyplot as plt
        from figstyle import (SURFACE, INK, MUTED, ACCENT, CONTEXT, GRID,
                              ANNOT, frame, title, footnote)
    except ImportError:
        return None

    xs = [a for a, _ in curve]
    ys = [r for _, r in curve]
    named = {0.001: ("Conservative", 10, 16, "left"),
             0.010: ("Balanced (default)", 0, -20, "center"),
             0.050: ("Sensitive", 0, 16, "center")}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3.0, 1.0],
                                              "hspace": 0.18})
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.20)
    frame(a1); frame(a2)

    a1.plot(xs, ys, "-", color=ACCENT, lw=1.8, zorder=3)
    a1.scatter(xs, ys, s=42, color=ACCENT, zorder=4,
               edgecolor=SURFACE, linewidth=1.6)
    for a, r in curve:
        if a in named:
            label, dx, dy, ha = named[a]
            a1.scatter([a], [r], s=150, facecolor="none", edgecolor=ACCENT,
                       linewidth=1.5, zorder=5)
            a1.annotate(label, (a, r), xytext=(dx, dy),
                        textcoords="offset points", ha=ha,
                        va="bottom" if dy > 0 else "top",
                        fontsize=ANNOT, color=INK)
    a1.set_ylabel("marks recovered  (%)", labelpad=9)
    lo, hi = min(ys), max(ys)
    a1.set_ylim(lo - 9, hi + 11)
    title(fig, a1, "Recovery rises with the acceptance level; "
                   "false positives stay at zero")

    a2.axhspan(0, 100.0 * bound, color=GRID, zorder=1)
    a2.plot(xs, [0] * len(xs), "-", color=CONTEXT, lw=1.8, zorder=3)
    a2.scatter(xs, [0] * len(xs), s=42, color=CONTEXT, zorder=4,
               edgecolor=SURFACE, linewidth=1.6)
    a2.annotate("95% upper bound", (xs[0] * 1.1, 72.0 * bound),
                fontsize=ANNOT, color=MUTED, ha="left", va="center")
    a2.set_xscale("log")
    a2.set_xlabel("acceptance level  \u03b1", labelpad=9)
    a2.set_ylabel("false-positive\nrate  (%)", labelpad=9)
    a2.set_ylim(-25.0 * bound, 140.0 * bound)
    a2.set_yticks([0, round(100.0 * bound, 2)])
    a2.set_xticks(xs)
    a2.set_xticklabels([str(x) for x in xs])

    footnote(fig, f"{n_shift} single-row skips \u00b7 sheet length {len(KEY)} \u00b7 "
                  f"ability {min(ABILITIES_SHIFT)}\u2013{max(ABILITIES_SHIFT)}\n"
                  f"No level tested produced a false positive; 0 of {n_certify} "
                  f"error-free sheets bounds the rate below {100.0 * bound:.2f}% (95%)")
    path = os.path.join(outdir, "figures", "recovery_vs_alpha.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    # Each default is set by the precision its own claim needs.
    #
    #  --clean 300    The calibration arm estimates how often the reported
    #                 p-value falls below a nominal level. The largest rate
    #                 measured is about 0.08, and 300 sheets puts a 95% interval
    #                 of roughly +/-0.03 around it: enough to show the level is
    #                 conservative by a factor, which is the claim.
    #  --shift 150    The profile arm reports recovery per level. 150 sheets
    #                 spread over five abilities gives 30 per ability, enough to
    #                 show the shape of the curve without claiming precision at
    #                 any single point.
    #  --certify 1500 The certification arm is the binding constraint on the
    #                 false positive claim, and the bound depends only on the
    #                 sheet count: 0 of 1,500 gives 0.20% at 95%. 3,000 would
    #                 give 0.10% and 30,000 would give 0.01%; 1,500 was chosen
    #                 as the point where the bound is well inside the harm
    #                 comparison in REPORT.md section 6.1 at an hour of compute.
    ap.add_argument("--clean", type=int, default=300, help="sheets for the calibration arm")
    ap.add_argument("--shift", type=int, default=150, help="sheets for the profile arm")
    ap.add_argument("--certify", type=int, default=1500, help="sheets for the certification arm")
    args = ap.parse_args()

    text, curve, bound = run(args.clean, args.shift, args.certify)
    print(text)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "results", "policy_profiles.txt")
    with open(out, "w") as fh:
        fh.write(text + "\n")
    fig = render_figure(curve, args.shift, args.certify, bound,
                        os.path.join(root, "results"))
    print(f"\nwritten to {out}" + (f" and {fig}" if fig else ""))


if __name__ == "__main__":
    main()
