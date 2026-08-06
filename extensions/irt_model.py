#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ITEM RESPONSE THEORY EMISSIONS FOR THE PAIR-HMM
================================================================================

The reference model assumes a single ability theta and treats every item as
equally hard:

    P(mark = key | aligned) = theta,  for all items.

That is a Rasch model with all item difficulties tied to zero -- psychometrically
the weakest assumption in the whole system, and the one flagged as the highest
-value extension in REPORT.md section 11. Cook (2013), cited in REPORT.md,
applies item response theory to this same problem, so an IRT emission inside a
probabilistic alignment model is the obvious next thing to try. It was tried.
The result is in ASSUMPTIONS.md A4: no measurable improvement.

That is not a comparison against Cook's method. This varies the emission model
inside the alignment; it does not implement a rival IRT detector.

THE MODEL
---------
Three-parameter logistic, per item j:

    P(correct on item j | theta) = c_j + (1 - c_j) * sigma( a_j * (theta - b_j) )

    b_j  difficulty       -- where on the ability scale the item bites
    a_j  discrimination   -- how sharply it separates candidates
    c_j  pseudo-guessing  -- the floor; defaults to 1/C, which is exactly the
                             chance baseline the coherence scan already uses,
                             so the two components stay consistent

WHY THIS MATTERS FOR SHIFT DETECTION SPECIFICALLY
-------------------------------------------------
It changes what counts as EVIDENCE of misalignment.

Under the constant-theta model, every wrong answer is equally surprising. Under
IRT, a candidate getting an EASY item wrong is strong evidence that something
mechanical went wrong; getting a HARD item wrong is barely evidence at all. The
log-likelihood ratio becomes item-weighted, so the detector concentrates on the
items that actually discriminate.

Symmetrically, a displaced alignment that "repairs" a run of easy items is worth
much less than one that repairs a run of hard items -- which is precisely the
right scepticism, because easy items have common answers and align by chance
more often.

CALIBRATION
-----------
Item parameters come from the COHORT, never from the candidate under appeal.
`RaschCalibrator` implements joint maximum likelihood from scratch (no external
libraries): alternating Newton updates on item difficulties and person
abilities, with the usual centring constraint for identifiability.

With no cohort available, `ItemBank.uninformative()` reproduces the constant-
theta model exactly, so the IRT path is a strict generalisation -- it can only
add information, never silently change behaviour when there is none to add.
================================================================================
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omr_shift import (  # noqa: E402
    AdjudicationConfig, Adjudicator, BandedPairHMM, ResponseSheet, ScoringModel,
)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ==============================================================================
# 1. ITEM BANK
# ==============================================================================


@dataclass
class ItemBank:
    """Calibrated item parameters. One entry per question.

    `constant=True` bypasses the logistic entirely and returns theta itself, so
    the bank reproduces the constant-ability model. Setting difficulty 0,
    discrimination 1 and guessing 0 does NOT achieve this: it gives
    sigmoid(theta), which equals 0.70 at theta = 0.85, not 0.85. An earlier
    version made that mistake and documented the result as a strict
    generalisation; it was not.
    """

    difficulty: List[float]
    discrimination: List[float]
    guessing: List[float]
    source: str = "uninformative"
    constant: bool = False

    @classmethod
    def uninformative(cls, n_items: int, n_options: int = 4) -> "ItemBank":
        """Reproduces the constant-ability model exactly, so the IRT path is a
        strict generalisation: with no cohort information it changes nothing."""
        return cls([0.0] * n_items, [1.0] * n_items, [0.0] * n_items,
                   "uninformative (equivalent to constant ability)", constant=True)

    @classmethod
    def with_guessing_floor(cls, n_items: int, n_options: int = 4) -> "ItemBank":
        c = 1.0 / n_options
        return cls([0.0] * n_items, [1.0] * n_items, [c] * n_items,
                   f"guessing floor c={c:.2f}, difficulties uncalibrated")

    def p_correct(self, j: int, theta: float) -> float:
        if self.constant:
            return theta
        c, a, b = self.guessing[j], self.discrimination[j], self.difficulty[j]
        return c + (1.0 - c) * sigmoid(a * (theta - b))

    def summary(self) -> Dict:
        n = len(self.difficulty)
        return {
            "n_items": n,
            "source": self.source,
            "difficulty_mean": sum(self.difficulty) / n,
            "difficulty_sd": math.sqrt(sum((b - sum(self.difficulty) / n) ** 2
                                           for b in self.difficulty) / n),
            "difficulty_range": (min(self.difficulty), max(self.difficulty)),
            "discrimination_mean": sum(self.discrimination) / n,
        }


# ==============================================================================
# 2. CALIBRATION FROM A COHORT  (joint maximum likelihood, from scratch)
# ==============================================================================


class RaschCalibrator:
    """
    Joint maximum-likelihood calibration of the Rasch (1PL) model.

    Input is a cohort response matrix: rows are candidates, columns are items,
    entries are 1 (correct) or 0. Alternating Newton updates on item
    difficulties and person abilities; item difficulties are centred each
    iteration for identifiability.

    Candidates and items with perfect or zero scores carry no information about
    the parameters and are excluded from the corresponding updates, which is the
    standard JMLE treatment.

    NOTE ON GOVERNANCE: the appellant's own responses must be EXCLUDED from
    calibration. Otherwise the item parameters adapt to the very sheet under
    dispute, and the evidence becomes partly self-generated.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6) -> None:
        self.max_iter, self.tol = max_iter, tol

    def fit(self, responses: Sequence[Sequence[int]],
            n_options: int = 4, guessing_floor: bool = True) -> ItemBank:
        n_p, n_i = len(responses), len(responses[0])
        item_tot = [sum(responses[p][j] for p in range(n_p)) for j in range(n_i)]
        pers_tot = [sum(responses[p]) for p in range(n_p)]

        b = [0.0] * n_i
        th = [0.0] * n_p
        # sensible starts from classical proportions
        for j in range(n_i):
            pr = min(max(item_tot[j] / n_p, 1e-3), 1 - 1e-3)
            b[j] = -math.log(pr / (1 - pr))
        for p in range(n_p):
            pr = min(max(pers_tot[p] / n_i, 1e-3), 1 - 1e-3)
            th[p] = math.log(pr / (1 - pr))

        live_p = [p for p in range(n_p) if 0 < pers_tot[p] < n_i]
        live_i = [j for j in range(n_i) if 0 < item_tot[j] < n_p]

        for _ in range(self.max_iter):
            delta = 0.0
            for p in live_p:                       # person update
                num = den = 0.0
                for j in range(n_i):
                    pr = sigmoid(th[p] - b[j])
                    num += responses[p][j] - pr
                    den += pr * (1 - pr)
                if den > 1e-9:
                    step = max(-1.0, min(1.0, num / den))
                    th[p] += step
                    delta = max(delta, abs(step))
            for j in live_i:                       # item update
                num = den = 0.0
                for p in range(n_p):
                    pr = sigmoid(th[p] - b[j])
                    num += responses[p][j] - pr
                    den += pr * (1 - pr)
                if den > 1e-9:
                    step = max(-1.0, min(1.0, -num / den))
                    b[j] += step
                    delta = max(delta, abs(step))
            m = sum(b) / n_i                       # identifiability constraint
            b = [x - m for x in b]
            if delta < self.tol:
                break

        c = 1.0 / n_options if guessing_floor else 0.0
        return ItemBank(b, [1.0] * n_i, [c] * n_i,
                        f"Rasch JMLE on {n_p} cohort sheets")


# ==============================================================================
# 3. IRT SCORING MODEL  -- drop-in replacement for ScoringModel
# ==============================================================================


class IRTScoringModel(ScoringModel):
    """
    Identical to ScoringModel except that the aligned-pair emission is
    item-specific.

    Everything else -- the orphan-row marginal, the gap hazards, and critically
    the FAIRNESS CLAMP on unmatched questions (axiom A5) -- is inherited
    unchanged. The clamp is re-derived per item here, since the wrong-answer
    probability now varies by item; without that, an item the candidate was
    almost certain to get right would make unmatching profitable again.
    """

    def __init__(self, sheet: ResponseSheet, cfg: AdjudicationConfig,
                 bank: Optional[ItemBank] = None) -> None:
        super().__init__(sheet, cfg)
        self.bank = bank or ItemBank.uninformative(sheet.n_questions, self.n_opts)

    def match_score(self, q_idx: int, r_idx: int, theta: float) -> float:
        mark = self.sheet.marks[r_idx]
        if mark is None:
            return self.log_blank_open(theta)
        p = min(max(self.bank.p_correct(q_idx, theta), 1e-6), 1 - 1e-6)
        if mark == self.sheet.key[q_idx]:
            return math.log(p)
        return math.log((1.0 - p) / (self.n_opts - 1))

    def log_blank_open(self, theta: float, q_idx: Optional[int] = None) -> float:
        """Fairness clamp, now item-aware.

        The clamp must bind against the EASIEST item on the paper -- the one
        with the smallest wrong-answer probability -- otherwise unmatching that
        item would be cheaper than admitting it was answered wrongly.
        """
        prior = math.log(self.cfg.blank_rate)
        if q_idx is None:
            worst = min(1.0 - self.bank.p_correct(j, theta)
                        for j in range(self.sheet.n_questions))
        else:
            worst = 1.0 - self.bank.p_correct(q_idx, theta)
        worst = max(worst, 1e-9)
        ceiling = math.log(worst / (self.n_opts - 1)) + math.log(self.cfg.blank_safety)
        return min(prior, ceiling)

    def log_blank_extend(self, theta: float) -> float:
        prior = math.log(self.cfg.blank_rate) + math.log(self.cfg.blank_extend)
        return min(prior, self.log_blank_open(theta))

    def validate_no_free_unmatch(self) -> List[str]:
        issues = []
        for t, _ in self.cfg.theta_grid(self.n_opts):
            for j in range(self.sheet.n_questions):
                wrong = math.log(max(1e-9, 1.0 - self.bank.p_correct(j, t))
                                 / (self.n_opts - 1))
                if self.log_blank_open(t) >= wrong + 1e-12:
                    issues.append(
                        f"axiom A5 violated at theta={t:.3f}, item {j+1}: "
                        f"unmatching is not costlier than a wrong match.")
                    break
        return issues

    def break_even_questions(self, theta: float) -> float:
        """Now item-dependent; reported as the average over the paper."""
        tot, n = 0.0, 0
        for j in range(self.sheet.n_questions):
            p = min(max(self.bank.p_correct(j, theta), 1e-6), 1 - 1e-6)
            gain = math.log(p) - math.log((1 - p) / (self.n_opts - 1))
            if gain > 1e-9:
                tot += -self.log_row_skip_open / gain
                n += 1
        return tot / n if n else float("inf")


class IRTAdjudicator(Adjudicator):
    """Adjudicator using IRT emissions. Identical gates and award rule."""

    def __init__(self, sheet: ResponseSheet, cfg: Optional[AdjudicationConfig] = None,
                 bank: Optional[ItemBank] = None) -> None:
        super().__init__(sheet, cfg)
        self.model = IRTScoringModel(sheet, self.cfg, bank)


# ==============================================================================
# 4. DOES IT ACTUALLY HELP?  -- the experiment that justifies the extension
# ==============================================================================


def experiment(n_sheets: int = 120, seed: int = 20260804) -> str:
    """
    Generate cohorts whose items genuinely differ in difficulty, calibrate an
    item bank from the cohort, then compare constant-theta against IRT emissions
    on held-out sheets containing a planted shift.

    The appellant sheet is NEVER part of the calibration cohort.
    """
    from omr_shift import CoherenceScanStatistic
    rng = random.Random(seed)
    OPT = ("A", "B", "C", "D")
    N = 46
    cfg = replace(AdjudicationConfig(), external_ability=0.85)

    true_b = [rng.gauss(0, 1.3) for _ in range(N)]
    key = [rng.choice(OPT) for _ in range(N)]

    def answer(theta):
        out = []
        for j, k in enumerate(key):
            p = 0.25 + 0.75 * sigmoid(theta - true_b[j])
            out.append(k if rng.random() < p else rng.choice([o for o in OPT if o != k]))
        return out

    # ---- cohort, excluding the appellants -------------------------------
    cohort = []
    for _ in range(400):
        th = rng.gauss(1.2, 1.0)
        cohort.append([1 if m == k else 0 for m, k in zip(answer(th), key)])
    bank = RaschCalibrator().fit(cohort, n_options=4)
    s = bank.summary()

    L = []
    L.append("=" * 76)
    L.append("IRT EMISSION MODEL -- DOES IT HELP?".center(76))
    L.append("=" * 76)
    L.append(f"  cohort for calibration : 400 sheets (appellants excluded)")
    L.append(f"  calibration            : {s['source']}")
    L.append(f"  recovered difficulty   : mean {s['difficulty_mean']:+.3f}, "
             f"sd {s['difficulty_sd']:.3f}, range "
             f"[{s['difficulty_range'][0]:+.2f}, {s['difficulty_range'][1]:+.2f}]")
    true_sd = math.sqrt(sum((b - sum(true_b)/N)**2 for b in true_b)/N)
    corr_num = sum((bank.difficulty[j] - s['difficulty_mean']) * (true_b[j] - sum(true_b)/N)
                   for j in range(N))
    corr = corr_num / (N * s['difficulty_sd'] * true_sd) if s['difficulty_sd'] > 0 else 0
    L.append(f"  true difficulty sd     : {true_sd:.3f}")
    L.append(f"  correlation(recovered, true) : {corr:+.3f}")
    L.append("")

    # ---- held-out appellants with a planted shift ------------------------
    flat = ItemBank.with_guessing_floor(N, 4)
    res = {"constant-theta": {"det": 0, "fa": 0}, "IRT": {"det": 0, "fa": 0}}
    for _ in range(n_sheets):
        th = rng.gauss(1.4, 0.5)
        clean = answer(th)
        at = rng.randint(6, N - 12)
        shifted = (clean[:at] + [rng.choice(OPT)] + clean[at:])[:N]
        sheet_shift = ResponseSheet(tuple(key), tuple(shifted))
        sheet_clean = ResponseSheet(tuple(key), tuple(clean))
        for label, bnk in (("constant-theta", flat), ("IRT", bank)):
            # n_permutations must exceed 1/alpha, else the Monte-Carlo gate can
            # never pass and every measured power is spuriously zero.
            npm = max(1500, int(2.0 / cfg.permutation_alpha))
            a = IRTAdjudicator(sheet_shift, cfg, bnk).run(n_permutations=npm, verbose=False)
            res[label]["det"] += bool(a.accepted)
            a0 = IRTAdjudicator(sheet_clean, cfg, bnk).run(n_permutations=npm, verbose=False)
            res[label]["fa"] += bool(a0.accepted)

    L.append(f"  held-out appellant sheets: {n_sheets} shifted, {n_sheets} clean")
    L.append(f"  Monte-Carlo draws per sheet: {max(1500, int(2.0/cfg.permutation_alpha))} "
             f"(must exceed 1/alpha = {int(1/cfg.permutation_alpha)})")
    L.append("")
    L.append(f"  {'emission model':<20}{'power':>10}{'false alarms':>15}")
    for label in ("constant-theta", "IRT"):
        L.append(f"  {label:<20}{res[label]['det']/n_sheets:>10.3f}"
                 f"{res[label]['fa']/n_sheets:>15.3f}")
    L.append("")
    L.append("  Both emission models inherit the same gates, the same fairness")
    L.append("  clamp and the same Monte-Carlo calibration, so any difference is")
    L.append("  attributable to the emission model alone.")
    L.append("=" * 76)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    text = experiment(n_sheets=args.n)
    print(text)
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    with open(os.path.join(here, "extension_irt.txt"), "w") as f:
        f.write(text)
