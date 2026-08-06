#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 REALISTIC ERROR MECHANISMS
 Injectors derived from how candidates actually describe going wrong
================================================================================

The injectors in `omrbench.py` were idealised: a shift begins at a uniformly
random position and continues to the end of the paper. Candidates describing
their own mistakes report something considerably messier. This module models
what they actually describe, and each mechanism is worked out PHYSICALLY --
what the sheet can really look like afterwards.

That discipline matters. Working the physics out changed the design in both
directions: it revealed a genuine (if narrow) limit of the reference method, and
it falsified two limitations that had previously been asserted on reasoning
alone. Both corrections are recorded inline below, next to the measurement.

  1. SELF-CORRECTED SHIFT  ("I realised after 20+ questions and fixed it")
     Physically subtle. Rows are filled in order, so a candidate who has been
     one row ahead since question a has no way to re-align at question b: row b
     is already occupied by question b-1's answer. Re-alignment without erasing
     is only possible if they LOSE one question in the process. The observable
     sheet is therefore: row a orphaned, questions a..b-1 displaced by +1,
     question b UNANSWERED, and questions b+1 onward correctly aligned.
     This is exactly the question-gap the pair-HMM's X state exists to model.

  2. DEFERRED QUESTION  ("I skipped it to come back, then filled it in later")
     The candidate skips question a on the paper WITHOUT skipping its row, so
     every later answer moves up one row. At the end they enter question a's
     answer in the one remaining blank row -- the last one.

     The FULL truth is non-monotone: question a maps to the LAST row while
     question a+1 maps to row a, so the question->row map decreases and axiom A2
     forbids it.

     MEASURED CORRECTION: this was first written up as a total structural blind
     spot. The benchmark falsified that. The method does not need the full
     truth -- it can take the best MONOTONE SUB-MAP, treating the deferred
     question as unanswered and the final row as an orphan. That sub-map IS
     strictly increasing, so everything except the one deferred answer is
     recoverable. Measured: 4.74 marks withheld against the no-op's 12.36, i.e.
     about 67% recovered, at a cost of exactly 1 mark (the deferred question,
     which can never be awarded because matching it would break monotonicity).
     The blind spot is one mark wide, not the whole mechanism.

  3. ANXIETY / TIME PRESSURE
     The moment of losing one's place is not clean. Ability degrades in a window
     around the slip. Every other injector assumes the candidate answers equally
     well either side of the error, which flatters every detector.

  4. ISOLATED MISPLACEMENT  ("rushed and put one answer in the wrong row")
     A single answer displaced and immediately recovered. Worth one mark. No
     detector should "correct" this -- it is the case where firing is worse than
     not firing, and it belongs in the benchmark as a trap.

  5. BOUNDARY SLIP
     Real slips cluster at column, page and block boundaries. On the sheet in
     this case that is row 40/41 (end of column 1) and rows 20/21. A uniform
     random start position understates how often the anchor region before the
     slip is short.

  6. EARLY FULL SHIFT  ("lost 40+ points -- every answer one row below")
     The slip begins in the first few questions and runs to the end, leaving
     almost no correctly-aligned anchor region beforehand.

     MEASURED CORRECTION: REPORT.md section 11 claimed this case was effectively
     undetectable for want of an anchor. The benchmark falsified that too --
     power 0.886, localisation error 0, and 3.01 marks withheld against the
     no-op's 20.93. The coherence scan does not need a pre-shift anchor; it
     needs a long coherent displaced block, and an early full shift supplies
     40+ questions of one. This is the single largest harm case in the suite
     (a candidate loses ~21 marks) and the method handles it well.

None of this is evidence about any individual candidate. It is a library of
mechanisms the benchmark should cover.
================================================================================
"""

from __future__ import annotations

import os
import random
import sys
import zlib
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omrbench import (  # noqa: E402
    OPTIONS, Detector, ErrorInjector, GroundTruth, GatedPairHMMDetector,
    FixedCostDPDetector, BruteForceShiftDetector, LCSDetector, NoOpDetector,
    CleanGenerator, TopicClusteredGenerator, NonStationaryGenerator,
    TwoRegimeGenerator, AttractiveDistractorGenerator,
)


def _finish(obs: List[str], key: Sequence[str], align: Dict[int, int],
            events: List[Tuple[int, int]], name: str,
            monotone: bool = True) -> Tuple[List[str], GroundTruth]:
    n = len(key)
    align = {q: r for q, r in align.items() if 0 <= r < len(obs)}
    true_s = sum(1 for q, r in align.items() if obs[r] == key[q])
    obs_s = sum(1 for q in range(min(n, len(obs))) if obs[q] == key[q])
    gt = GroundTruth(True, events, align, true_s, obs_s, "", name)
    gt.monotone = monotone          # attached for reporting
    return obs, gt


class SelfCorrectedShiftInjector(ErrorInjector):
    """'I realised after 20+ questions and corrected it.'

    Physically: row `a` orphaned, questions a..b-1 displaced +1, question b lost
    (it is the price of re-aligning without erasing), tail correctly aligned.
    """
    name = "self_corrected_shift"
    description = "slip noticed after ~20 questions; realignment costs one question"

    def inject(self, rng, key, marks):
        n = len(key)
        a = rng.randint(4, n - 26)
        b = a + rng.randint(15, 24)
        obs = list(marks[:a]) + [rng.choice(OPTIONS)] + list(marks[a:b]) + list(marks[b + 1:])
        obs = obs[:n]
        align = {}
        for q in range(n):
            if q < a:
                align[q] = q
            elif q < b:
                align[q] = q + 1
            elif q > b:
                align[q] = q
            # q == b : unanswered
        return _finish(obs, key, align, [(a + 1, 1), (b + 1, -1)], self.name)


class DeferredQuestionInjector(ErrorInjector):
    """'I skipped it to come back to, then filled it in at the end.'

    The full alignment is NON-MONOTONE, so axiom A2 forbids representing it
    exactly. But the best monotone SUB-map recovers everything except the single
    deferred answer -- see the module docstring. Measured recovery ~67%.
    """
    name = "deferred_question"
    description = "skipped a question without skipping its row; answered it last"

    def inject(self, rng, key, marks):
        n = len(key)
        a = rng.randint(3, n - 12)
        obs = list(marks[:a]) + list(marks[a + 1:]) + [marks[a]]
        obs = obs[:n]
        align = {}
        for q in range(n):
            if q < a:
                align[q] = q
            elif q == a:
                align[q] = n - 1              # answered last -> NON-MONOTONE
            else:
                align[q] = q - 1
        return _finish(obs, key, align, [(a + 1, -1)], self.name, monotone=False)


class AnxietyShiftInjector(ErrorInjector):
    """A slip, plus degraded performance in a window around it."""
    name = "anxiety_shift"
    description = "slip accompanied by degraded answers either side of the change point"

    def inject(self, rng, key, marks):
        n = len(key)
        a = rng.randint(6, n - 14)
        m = list(marks)
        for i in range(max(0, a - 4), min(n, a + 6)):     # panic window
            if rng.random() < 0.55:
                m[i] = rng.choice([o for o in OPTIONS if o != key[i]])
        obs = m[:a] + [rng.choice(OPTIONS)] + m[a:]
        obs = obs[:n]
        align = {q: (q if q < a else q + 1) for q in range(n)}
        return _finish(obs, key, align, [(a + 1, 1)], self.name)


class IsolatedMisplacementInjector(ErrorInjector):
    """One answer in the wrong row, immediately recovered. A TRAP: worth one
    mark, and any detector that 'corrects' it is doing harm."""
    name = "isolated_misplacement"
    description = "single answer rushed into the wrong row, then recovered"

    def inject(self, rng, key, marks):
        n = len(key)
        a = rng.randint(4, n - 4)
        obs = list(marks)
        obs[a], obs[a + 1] = obs[a + 1], obs[a]
        align = {q: q for q in range(n)}
        align[a], align[a + 1] = a + 1, a
        return _finish(obs, key, align, [(a + 1, 0)], self.name, monotone=False)


class BoundarySlipInjector(ErrorInjector):
    """Slip beginning exactly at a column / block boundary."""
    name = "boundary_slip"
    description = "slip starting at a column or block boundary (rows 20/21, 40/41)"

    def __init__(self, boundaries: Sequence[int] = (20, 40)):
        self.boundaries = boundaries

    def inject(self, rng, key, marks):
        n = len(key)
        cands = [b for b in self.boundaries if 4 < b < n - 6]
        a = rng.choice(cands) if cands else n // 2
        obs = list(marks[:a]) + [rng.choice(OPTIONS)] + list(marks[a:])
        obs = obs[:n]
        align = {q: (q if q < a else q + 1) for q in range(n)}
        return _finish(obs, key, align, [(a + 1, 1)], self.name)


class EarlyFullShiftInjector(ErrorInjector):
    """'Every answer one row below after an early mistake.' Almost no anchor
    region before the slip -- the hardest condition for change-point logic."""
    name = "early_full_shift"
    description = "slip in the first few questions, running to the end of the paper"

    def inject(self, rng, key, marks):
        n = len(key)
        a = rng.randint(1, 4)
        obs = list(marks[:a]) + [rng.choice(OPTIONS)] + list(marks[a:])
        obs = obs[:n]
        align = {q: (q if q < a else q + 1) for q in range(n)}
        return _finish(obs, key, align, [(a + 1, 1)], self.name)


REALISTIC_INJECTORS = [
    ErrorInjector(),
    SelfCorrectedShiftInjector(),
    DeferredQuestionInjector(),
    AnxietyShiftInjector(),
    IsolatedMisplacementInjector(),
    BoundarySlipInjector(),
    EarlyFullShiftInjector(),
]


# ==============================================================================


def run(n_per_cell: int = 20, seed: int = 20260804) -> Dict:
    gens = [CleanGenerator(), TopicClusteredGenerator(), NonStationaryGenerator(),
            AttractiveDistractorGenerator(), TwoRegimeGenerator()]
    dets = [NoOpDetector(), BruteForceShiftDetector(), LCSDetector(),
            FixedCostDPDetector(), GatedPairHMMDetector()]
    out: Dict[Tuple[str, str], Dict] = {}
    for inj in REALISTIC_INJECTORS:
        cases = []
        for g in gens:
            # deterministic across processes; see omrbench.py
            rng = random.Random(seed + zlib.crc32((g.name + inj.name).encode()) % 100000)
            for _ in range(n_per_cell):
                k, m = g.generate(rng)
                obs, gt = inj.inject(rng, k, m)
                cases.append((k, obs, gt))
        print(f"  injector {inj.name:<24} ({len(cases)} sheets) ...", flush=True)
        for d in dets:
            fired = 0
            recovered = 0.0
            awarded = 0.0
            loc = []
            for k, obs, gt in cases:
                dec = d.decide(k, obs)
                fired += bool(dec.accepted)
                delta = dec.awarded_score - gt.true_score
                awarded += max(0, delta)
                recovered += max(0, -delta)
                if dec.accepted and dec.shift_locations and gt.events:
                    loc.append(min(abs(l.get("at_question", 0) - gt.events[0][0])
                                   for l in dec.shift_locations))
            n = len(cases)
            out[(inj.name, d.name)] = {
                "fire_rate": fired / n,
                "marks_wrongly_awarded": awarded / n,
                "marks_wrongly_withheld": recovered / n,
                "median_loc": (sorted(loc)[len(loc) // 2] if loc else None),
                "n": n,
            }
    return out


def render(res: Dict, n_per_cell: int) -> str:
    dets = ["no-op (never correct)", "brute-force shift", "LCS (maximally generous)",
            "fixed-cost DP alignment", "gated pair-HMM (reference)"]
    L, w = [], 96
    L.append("=" * w)
    L.append("REALISTIC ERROR MECHANISMS -- PER-INJECTOR RESULTS".center(w))
    L.append("=" * w)
    L.append(f"{n_per_cell} sheets per (generator x injector), 5 generators = "
             f"{n_per_cell*5} sheets per injector.")
    L.append("")
    L.append("  FIRE   how often the detector accepted a correction")
    L.append("  AWARD  mean marks wrongly AWARDED per sheet   (harms integrity)")
    L.append("  HOLD   mean marks wrongly WITHHELD per sheet  (harms the candidate)")
    L.append("  LOC    median localisation error, questions")
    L.append("")
    for inj in REALISTIC_INJECTORS:
        meta = inj
        L.append("-" * w)
        L.append(f"MECHANISM: {inj.name}")
        L.append(f"  {getattr(meta, 'description', '')}")
        if inj.name == "deferred_question":
            L.append("  >>> NON-MONOTONE: outside the reference method's feasible set. <<<")
        if inj.name == "isolated_misplacement":
            L.append("  >>> TRAP: worth 1 mark. Firing here is worse than not firing. <<<")
        if inj.name == "none":
            L.append("  (control: no error at all -- FIRE is the false-alarm rate)")
        L.append("-" * w)
        L.append(f"  {'detector':<28}{'FIRE':>8}{'AWARD':>9}{'HOLD':>8}{'LOC':>7}")
        for d in dets:
            r = res.get((inj.name, d))
            if not r:
                continue
            loc = "-" if r["median_loc"] is None else f"{r['median_loc']:.0f}"
            L.append(f"  {d:<28}{r['fire_rate']:>8.3f}{r['marks_wrongly_awarded']:>9.2f}"
                     f"{r['marks_wrongly_withheld']:>8.2f}{loc:>7}")
        L.append("")
    L.append("=" * w)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    print("Running realistic-error benchmark ...")
    res = run(n_per_cell=args.n)
    text = render(res, args.n)
    print()
    print(text)
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    with open(os.path.join(here, "benchmark_mechanisms.txt"), "w") as f:
        f.write(text)
    print("wrote report_realistic_errors.txt")
