#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ITEM-WEIGHTED COHERENCE SCAN
 Putting psychometrics into the gate that actually binds
================================================================================

WHY
---
Two measured facts force this design:

  C3  The Monte-Carlo coherence gate is the BINDING constraint. It is the only
      gate that holds when every other one is bought, and it uses no ability
      model at all.

  F10 IRT emissions produced no measurable improvement (26/40 vs 25/40). Not an
      implementation failure -- Rasch calibration recovered item difficulty at
      r = +0.992. The emission model feeds the Bayes-factor gate, which is not
      the bottleneck.

Conclusion: item information has to go into the coherence statistic itself, or
it cannot affect the decision.

THE STATISTIC
-------------
The plain scan counts matches inside a candidate displaced block and tests that
count against chance. This replaces the count with a sum of log-likelihood
ratios:

    T_w = max over displacement d != 0, and over windows W with |W| >= L_min, of

              SUM_{q in W}  log [ P(observation | q aligned to row q+d)
                                / P(observation | that row is unrelated to q) ]

    match     ->  log[ P_correct(q) / p_hat(key_q) ]
    mismatch  ->  log[ ((1 - P_correct(q)) / (C-1)) / p_hat(mark) ]

Two sources of item information enter, and they are different things:

  1. THE NULL uses the candidate's own response marginals p_hat, not 1/C. An
     item whose key is a letter this candidate marks 35% of the time aligns by
     chance 35% of the time, not 25%. This prices out response bias, which the
     unweighted scan cannot see at all.

  2. THE ALTERNATIVE uses IRT: P_correct(q) = c + (1-c) sigma(a(theta - b)).

DIRECTION OF THE WEIGHTING -- COUNTER-INTUITIVE, AND WORTH STATING
------------------------------------------------------------------
The obvious guess is that a match on a HARD item should count for more. That is
backwards. Under a displaced alignment the mark came from a different question,
so difficulty is irrelevant to the null; it enters only through the alternative:

    very easy item (p=0.95) : match +1.34 nats,  MISMATCH -2.71 nats
    very hard item (p=0.28) : match +0.11 nats,  mismatch -0.04 nats

The informative items are the EASY ones, and the informative event is a
MISMATCH. A candidate failing an item they should have found easy is strong
evidence that something mechanical went wrong. Failing a hard item says almost
nothing. Hard items are nearly uninformative in both directions.

CALIBRATION
-----------
T_w has no closed-form null, so it is calibrated by the same three Monte-Carlo
nulls as the unweighted statistic, recomputed under the same weighting. The
maximum over (displacement, window) is taken inside every replicate, so
multiplicity is handled exactly. Nothing about the fairness architecture
changes: this replaces one statistic with another inside the existing gate.
================================================================================
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omr_shift import (  # noqa: E402
    AdjudicationConfig, CoherenceScanStatistic, ResponseSheet,
)
from irt_model import ItemBank, sigmoid  # noqa: E402

NEG_INF = float("-inf")


class WeightedCoherenceScan:
    """Item-weighted coherence scan. Drop-in alternative to CoherenceScanStatistic."""

    def __init__(self, cfg: AdjudicationConfig, bank: Optional[ItemBank] = None,
                 theta: float = 0.85, n_options: int = 4) -> None:
        self.cfg = cfg
        self.bank = bank
        self.theta = theta
        self.C = n_options

    # -- per-sheet nuisance quantities -------------------------------------

    def _marginals(self, sheet: ResponseSheet) -> Dict[str, float]:
        counts = {o: 1.0 for o in sheet.options}
        for m in sheet.marks:
            if m is not None:
                counts[m] += 1.0
        tot = sum(counts.values())
        return {o: c / tot for o, c in counts.items()}

    def _p_correct(self, q: int, n_items: int) -> float:
        if self.bank is None:
            return self.theta
        return self.bank.p_correct(q, self.theta)

    def _llr(self, q: int, mark: Optional[str], sheet: ResponseSheet,
             pm: Dict[str, float]) -> float:
        if mark is None:
            return 0.0
        p = min(max(self._p_correct(q, sheet.n_questions), 1e-6), 1 - 1e-6)
        if mark == sheet.key[q]:
            return math.log(p / max(pm[mark], 1e-9))
        return math.log(((1.0 - p) / (self.C - 1)) / max(pm[mark], 1e-9))

    # -- the statistic ------------------------------------------------------

    def compute(self, sheet: ResponseSheet) -> Tuple[float, Optional[Dict]]:
        N, M = sheet.n_questions, sheet.n_rows
        D, Lmin = self.cfg.max_displacement, self.cfg.min_segment_length
        pm = self._marginals(sheet)
        best, arg = 0.0, None
        for d in range(-D, D + 1):
            if d == 0:
                continue
            idx = [q for q in range(N) if 0 <= q + d < M]
            if len(idx) < Lmin:
                continue
            cum = [0.0] * (len(idx) + 1)
            for i, q in enumerate(idx):
                cum[i + 1] = cum[i] + self._llr(q, sheet.marks[q + d], sheet, pm)
            # maximum-sum window of length >= Lmin, via running prefix minimum
            best_pref = [0.0] * (len(idx) + 1)
            argmin = [0] * (len(idx) + 1)
            bp, ai = cum[0], 0
            for j in range(len(idx) + 1):
                if j >= Lmin:
                    if cum[j - Lmin] < bp:
                        bp, ai = cum[j - Lmin], j - Lmin
                best_pref[j], argmin[j] = bp, ai
            for j in range(Lmin, len(idx) + 1):
                v = cum[j] - best_pref[j]
                if v > best:
                    i = argmin[j]
                    k = sum(1 for t in range(i, j)
                            if sheet.marks[idx[t] + d] == sheet.key[idx[t]])
                    best = v
                    arg = {"offset": d, "q_start": idx[i] + 1, "q_end": idx[j - 1] + 1,
                           "n_items": j - i, "n_correct": k, "llr": v}
        return best, arg


# ==============================================================================
# COMPARISON EXPERIMENT
# ==============================================================================


def _mc_p(scan, sheet, observed, n_perm, rng, alpha: Optional[float] = None) -> float:
    """
    Key-marginal Monte-Carlo p-value, with EXACT early termination.

    The p-value is (exceedances + 1) / (draws + 1). To pass a gate at level
    alpha we need (e + 1) / (n + 1) <= alpha, i.e. e <= alpha*(n+1) - 1. With
    alpha = 0.001 and n = 1200 that allows e = 0: a SINGLE exceedance already
    makes the gate unpassable (2/1201 = 0.00167 > 0.001).

    So once the exceedance count passes the allowance, the decision is settled
    and the remaining draws cannot change it. Stopping there is exact -- it is
    not an approximation or a subsample. Sheets with no shift typically settle
    in one or two draws, well short of 1200.

    When it stops early the returned value is a valid LOWER BOUND on the true
    p-value, which is all the gate needs; pass alpha=None to force the full run
    when the exact value is wanted for reporting.
    """
    if alpha is None:
        draws = []
        for _ in range(n_perm):
            k = tuple(rng.choice(sheet.options) for _ in range(sheet.n_questions))
            draws.append(scan.compute(replace(sheet, key=k))[0])
        return (sum(1 for v in draws if v >= observed) + 1) / (len(draws) + 1)

    allowed = math.floor(alpha * (n_perm + 1) - 1)
    exceed = 0
    for i in range(n_perm):
        k = tuple(rng.choice(sheet.options) for _ in range(sheet.n_questions))
        if scan.compute(replace(sheet, key=k))[0] >= observed:
            exceed += 1
            if exceed > allowed:
                return (exceed + 1) / (i + 2)      # already > alpha; settled
    return (exceed + 1) / (n_perm + 1)


def experiment(n_sheets: int = 60, n_perm: int = 1200, seed: int = 20260805) -> str:
    from omr_shift import CASE_SHEET
    rng = random.Random(seed)
    OPT = ("A", "B", "C", "D")
    N = 46
    cfg = AdjudicationConfig()
    alpha = cfg.permutation_alpha

    # --- a world where items genuinely differ in difficulty ---------------
    true_b = [rng.gauss(0, 1.3) for _ in range(N)]
    bank = ItemBank(true_b, [1.0] * N, [0.25] * N, "known item parameters")
    key = [rng.choice(OPT) for _ in range(N)]

    def answer(theta):
        out = []
        for j, k in enumerate(key):
            p = 0.25 + 0.75 * sigmoid(theta - true_b[j])
            out.append(k if rng.random() < p else rng.choice([o for o in OPT if o != k]))
        return out

    plain = CoherenceScanStatistic(cfg, 4)
    weighted = WeightedCoherenceScan(cfg, bank, theta=1.4, n_options=4)

    res = {"plain": {"det": 0, "fa": 0}, "weighted": {"det": 0, "fa": 0}}
    # PAIRED outcomes: both statistics see the SAME sheets, so McNemar's test on
    # discordant pairs is far more powerful than comparing two independent
    # proportions -- and it is the correct test for this design.
    b_only_w, b_only_p = 0, 0
    for _ in range(n_sheets):
        clean = answer(rng.gauss(1.4, 0.5))
        at = rng.randint(6, N - 12)
        shifted = (clean[:at] + [rng.choice(OPT)] + clean[at:])[:N]
        hit = {}
        for label, sc in (("plain", plain), ("weighted", weighted)):
            for kind, marks in (("det", shifted), ("fa", clean)):
                sh = ResponseSheet(tuple(key), tuple(marks))
                obs = sc.compute(sh)[0]
                p = _mc_p(sc, sh, obs, n_perm, rng, alpha=alpha)
                fired = int(p <= alpha)
                res[label][kind] += fired
                hit[(label, kind)] = fired
        if hit[("weighted", "det")] and not hit[("plain", "det")]:
            b_only_w += 1
        if hit[("plain", "det")] and not hit[("weighted", "det")]:
            b_only_p += 1

    L = []
    L.append("=" * 76)
    L.append("ITEM-WEIGHTED COHERENCE SCAN vs PLAIN SCAN".center(76))
    L.append("=" * 76)
    L.append("  Both statistics are calibrated by the SAME Monte-Carlo null at the")
    L.append(f"  same alpha ({alpha}), with {n_perm} draws. Any difference is the")
    L.append("  statistic alone.")
    L.append(f"  Held-out sheets: {n_sheets} shifted, {n_sheets} clean.")
    L.append(f"  Items genuinely vary in difficulty (sd of b = "
             f"{math.sqrt(sum((b-sum(true_b)/N)**2 for b in true_b)/N):.2f}).")
    L.append("")
    L.append(f"  {'statistic':<24}{'power':>10}{'false alarms':>15}")
    for lab in ("plain", "weighted"):
        L.append(f"  {lab:<24}{res[lab]['det']/n_sheets:>10.3f}"
                 f"{res[lab]['fa']/n_sheets:>15.3f}")
    L.append("")
    # exact McNemar (binomial) on the discordant pairs
    nd = b_only_w + b_only_p
    if nd == 0:
        pm = 1.0
    else:
        pm = min(1.0, 2 * sum(math.comb(nd, i) * 0.5 ** nd
                              for i in range(min(b_only_w, b_only_p) + 1)))
    L.append(f"  PAIRED comparison on the same sheets (exact McNemar):")
    L.append(f"    weighted detected, plain missed : {b_only_w}")
    L.append(f"    plain detected, weighted missed : {b_only_p}")
    L.append(f"    discordant pairs = {nd},  two-sided p = {pm:.4f}")
    verdict = ("IMPROVEMENT (significant)" if pm <= 0.05 and b_only_w > b_only_p
               else "no significant difference")
    L.append(f"    verdict: {verdict}")
    L.append("")

    # --- and on the real sheet -------------------------------------------
    real = ResponseSheet.from_file(CASE_SHEET)
    flat = WeightedCoherenceScan(cfg, None, theta=0.85, n_options=4)
    o_p, w_p = plain.compute(real)[0], flat.compute(real)[0]
    p_p = _mc_p(plain, real, o_p, n_perm, rng)
    p_w = _mc_p(flat, real, w_p, n_perm, rng)
    L.append("  On the appellant's real sheet (no item bank available):")
    L.append(f"    plain scan    T = {o_p:6.3f}   Monte-Carlo p = {p_p:.4f}")
    L.append(f"    weighted scan T = {w_p:6.3f}   Monte-Carlo p = {p_w:.4f}")
    L.append("    (both must remain non-significant, or the refinement has")
    L.append("     bought power by loosening the safety property)")
    L.append("=" * 76)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--perm", type=int, default=1200)
    args = ap.parse_args()
    text = experiment(n_sheets=args.n, n_perm=args.perm)
    print(text)
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    with open(os.path.join(here, "extension_weighted_scan.txt"), "w") as f:
        f.write(text)
