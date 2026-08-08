#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-cohort screening simulation for displacement detection.

WHY THIS EXISTS
Per-sheet false alarm rates accumulate when processing thousands of submissions.
This script tests full-sitting screening using synthetic data with known ground truth
to verify that false discovery controls function as intended.

HOW IT WORKS
- Generates a synthetic examination cohort matching historical base rates.
- Applies `CohortScreen` using Benjamini-Hochberg false discovery rate adjustments.
- Compares expected false discovery counts against actual false discoveries.
- Measures pipeline mechanics rather than calibrating for specific live exams.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import zlib
from dataclasses import replace
from typing import List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance  # noqa: E402
from omr_shift import (  # noqa: E402
    AdjudicationConfig, Adjudicator, CohortScreen, ResponseSheet,
    clopper_pearson_upper,
)

SEED = 20260808
OPTIONS = "ABCD"
# Paper length matters more here than anywhere else in the repository, so it is
# a parameter rather than a constant. The evidence a displaced sheet can produce
# is bounded by how many questions follow the slip, and the cohort threshold it
# must clear is set by the size of the sitting. On a 20-question paper the
# detection floor already asks for 8 to 11 correct marks -- half the sheet --
# and the p-value bottoms out around 1e-3, far above what a large cohort needs.
N_DEFAULT = 46
ABILITIES = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def _sheet(i: int, base_rate: float, n: int = N_DEFAULT) -> Tuple[ResponseSheet, bool]:
    """One candidate. Carries a genuine skip with probability `base_rate`."""
    rng = random.Random(zlib.crc32(f"cohort:{i}".encode()) ^ SEED)
    key = [rng.choice(OPTIONS) for _ in range(n)]
    theta = ABILITIES[i % len(ABILITIES)]
    truth = [k if rng.random() < theta else rng.choice([o for o in OPTIONS if o != k])
             for k in key]
    if rng.random() < base_rate:
        q = rng.randint(2, n - 5)
        marks = truth[:q - 1] + [None] + truth[q - 1:n - 1]
        return ResponseSheet(tuple(key), tuple(marks), candidate_id=f"c{i}"), True
    return ResponseSheet(tuple(key), tuple(truth), candidate_id=f"c{i}"), False


def run(n_sheets: int, q: float, base_rate: float,
        n_items: int = N_DEFAULT) -> str:
    screen = CohortScreen(q=q, base_rate=base_rate)
    # The draw count is set by the cohort, not by the sheet. Below the required
    # number no sheet can clear the step-up threshold and the screen reports
    # zero flagged whatever the sheets contain, so it is derived rather than
    # chosen. Early stopping keeps this affordable: an error-free sheet is
    # abandoned once its exceedance budget is spent, so the full count is paid
    # only on sheets that look displaced.
    draws = screen.draws_required(n_sheets, q)
    cfg = replace(AdjudicationConfig(), n_permutations=draws)
    screen.check_resolution(n_sheets, cfg.n_permutations)
    results: List[Tuple[str, float, bool]] = []
    truth: dict = {}
    for i in range(n_sheets):
        sheet, has_shift = _sheet(i, base_rate, n_items)
        truth[sheet.candidate_id] = has_shift
        adj = Adjudicator(sheet, cfg).run(n_permutations=draws,
                                          verbose=False, early_stop=True)
        results.append((sheet.candidate_id, adj.calibration["p_value"], adj.accepted))

    report = screen.screen(results)

    per_sheet = sum(1 for _, _, gate in results if gate)
    flagged = [d for d in report.decisions if d.accepted_in_cohort]
    false_disc = sum(1 for d in flagged if not truth[d.sheet_id])
    n_with_shift = sum(1 for v in truth.values() if v)
    n_clean = n_sheets - n_with_shift
    fp_alone = sum(1 for sid, _, gate in results if gate and not truth[sid])

    L = [report.text(), ""]
    L.append(f"  permutation draws per sheet  : {draws:,}  (derived from the cohort size)")
    L.append(f"  questions per paper          : {n_items}")
    w = 74
    L.append("-" * w)
    L.append("AGAINST THE KNOWN TRUTH  --  available here, not on a real sitting")
    L.append("-" * w)
    L.append(f"  sheets with a genuine skip   : {n_with_shift}  "
             f"({n_with_shift / n_sheets:.4f} of the cohort)")
    L.append(f"  error-free sheets            : {n_clean}")
    L.append("")
    L.append(f"  flagged by the per-sheet gate alone : {per_sheet}"
             f"   of which wrong: {fp_alone}")
    L.append(f"  flagged after the cohort screen     : {report.n_flagged}"
             f"   of which wrong: {false_disc}")
    L.append("")
    obs = false_disc / report.n_flagged if report.n_flagged else 0.0
    L.append(f"  observed false discovery rate : {obs:.4f}")
    L.append(f"  bound published in advance    : {q:.4f}")
    L.append(f"  observed within the bound     : {'yes' if obs <= q else 'NO'}")
    L.append("")
    L.append(f"  false positives on error-free sheets : {fp_alone} of {n_clean}")
    L.append(f"  95% upper bound (Clopper-Pearson)    : "
             f"{clopper_pearson_upper(fp_alone, n_clean):.5f}")
    L.append("")
    L.append("  The cohort layer can only remove sheets, never add them, so")
    L.append("  screening a sitting is never more permissive than adjudicating")
    L.append("  one sheet at a time.")
    L.append("=" * w)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=4000,
                    help="cohort size; the false discovery bound is what it buys")
    ap.add_argument("--q", type=float, default=0.05, help="target false discovery rate")
    ap.add_argument("--base-rate", type=float, default=0.018,
                    help="share of sheets carrying a genuine skip")
    ap.add_argument("--length", type=int, default=N_DEFAULT,
                    help="questions per paper; the evidence a slip can produce "
                         "is bounded by how many questions follow it")
    args = ap.parse_args()
    text = run(args.sheets, args.q, args.base_rate, args.length)
    print(text)
    out = os.path.join(_ROOT, "results", "cohort_screen.txt")
    provenance.write_text(out, text, sheets=args.sheets, q=args.q,
                          base_rate=args.base_rate, length=args.length)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
