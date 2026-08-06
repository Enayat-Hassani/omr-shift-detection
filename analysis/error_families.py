#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MECHANICAL ERROR TAXONOMY
 Testing every structured failure mode, displacement included
================================================================================

`omr_shift.py` answers one question: "is this sheet DISPLACED?"
That is a single member of a much larger family. This module asks the general
question: **"is there ANY low-complexity mechanical transformation that explains
this sheet better than 'the answers were wrong'?"**

THE UNIFYING IDEA
-----------------
Every mechanical error is a structured, low-description-length transformation of
one of exactly two things:

    POSITION errors  -- WHERE a mark was recorded, not WHAT was marked.
        displacement / row skip .......... omr_shift.py
        wrong shuffle map (booklet form) . a permutation of question order
        section misassignment ............ answers written in another
                                           subject's block of the sheet
        block or column transposition .... 4-column sheets invite this
        reversal ......................... sheet scanned or filled upside-down
        circular rotation ................ wrap-around

    SYMBOL errors    -- WHAT was recorded, not where.
        bubble-column mis-registration ... every A read as B, B as C, ...
        mirrored bubble row .............. A<->D, B<->C
        wrong key legend ................. the key's letters relabelled

THE KEY THEOREM, AND WHY IT MATTERS
-----------------------------------
    **Every POSITION error preserves the multiset of marks.**

Shifting, shuffling, reversing, swapping blocks, or reading the wrong section
changes only WHERE each mark lands, never WHICH marks exist. Therefore the
counts of A, B, C and D on the sheet are INVARIANT across the entire position-
error family.

So a single test disposes of the whole family at once:

    If these marks are a re-ordering of a competent candidate's answers, then
    the mark counts must match what a competent candidate's answers would
    contain -- because re-ordering cannot create or destroy an 'A'.

This is far stronger than testing each transformation separately. It requires no
search, has no multiplicity to correct for, and cannot be defeated by a
transformation nobody thought to enumerate. It is the first thing that should be
run on any suspected mis-registration, and it costs microseconds.

Its limitation is equally clear: it says nothing about SYMBOL errors, which
change the multiset by construction. Those are enumerated exhaustively instead
(there are only 4! = 24 of them).

Author: companion to omr_shift.py.
================================================================================
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ==============================================================================
# 1. RESULT TYPES
# ==============================================================================


@dataclass
class HypothesisResult:
    """One mechanical-error family, tested."""

    family: str
    description: str
    best_transform: str
    n_correct: int
    n_items: int
    raw_p: float
    n_tests: int                 # size of the search space, for multiplicity
    corrected_p: float
    verdict: str

    @property
    def rate(self) -> float:
        return self.n_correct / self.n_items if self.n_items else 0.0


# ==============================================================================
# 2. STATISTICS
# ==============================================================================


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p). Exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def sidak(p: float, m: int) -> float:
    """Sidak correction -- exact under independence, never exceeds 1.

    Preferred over Bonferroni here because m runs to thousands, where
    Bonferroni's p*m saturates at 1 and destroys the ability to compare
    families against each other.
    """
    if p >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p) ** m


# ==============================================================================
# 3. THE TAXONOMY
# ==============================================================================


class MechanicalErrorTaxonomy:
    """
    Exhaustive test of structured mechanical explanations for a sheet.

    Usage:
        tax = MechanicalErrorTaxonomy(key, marks)
        for r in tax.run_all():
            print(r)
        print(tax.invariance_report())
    """

    def __init__(self, key: Sequence[str], marks: Sequence[Optional[str]],
                 options: Sequence[str] = ("A", "B", "C", "D"), seed: int = 20260804) -> None:
        self.key = list(key)
        self.marks = list(marks)
        self.opts = tuple(options)
        self.N = len(self.key)
        self.C = len(self.opts)
        self.chance = 1.0 / self.C
        self.rng = random.Random(seed)

    # -- helpers -----------------------------------------------------------

    def _score_aligned(self, seq: Sequence[Optional[str]]) -> int:
        return sum(1 for i in range(self.N) if i < len(seq) and seq[i] == self.key[i])

    def _score_offset(self, seq: Sequence[Optional[str]], d: int) -> Tuple[int, int]:
        idx = [q for q in range(self.N) if 0 <= q + d < len(seq)]
        return sum(1 for q in idx if seq[q + d] == self.key[q]), len(idx)

    def _verdict(self, corrected_p: float) -> str:
        if corrected_p <= 0.001:
            return "SUPPORTED (decisive)"
        if corrected_p <= 0.01:
            return "supported (strong)"
        if corrected_p <= 0.05:
            return "weak support"
        return "not supported"

    def _result(self, family, desc, transform, k, n, n_tests) -> HypothesisResult:
        raw = binom_sf(k, n, self.chance)
        cp = sidak(raw, n_tests)
        return HypothesisResult(family, desc, transform, k, n, raw, n_tests, cp,
                                self._verdict(cp))

    # -- POSITION errors ---------------------------------------------------

    def test_displacement(self, min_overlap: int = 8) -> HypothesisResult:
        """Row skip / added row, at ANY magnitude, small and large."""
        best, arg, tests = (0, 0), None, 0
        for d in range(-(self.N - min_overlap), self.N - min_overlap + 1):
            k, n = self._score_offset(self.marks, d)
            if n < min_overlap:
                continue
            tests += 1
            if binom_sf(k, n, self.chance) < binom_sf(best[0], best[1], self.chance) or arg is None:
                best, arg = (k, n), d
        return self._result("POSITION: displacement",
                            "sheet read from the wrong row; any magnitude",
                            f"offset {arg:+d}", best[0], best[1], tests)

    def test_reversal(self) -> HypothesisResult:
        """Sheet filled bottom-to-top, or scanned upside-down."""
        k = self._score_aligned(self.marks[::-1])
        return self._result("POSITION: reversal",
                            "sheet scanned upside-down or filled bottom-up",
                            "reverse", k, self.N, 1)

    def test_rotation(self) -> HypothesisResult:
        """Circular wrap-around of the mark sequence."""
        best, arg = 0, 0
        for r in range(1, self.N):
            k = self._score_aligned(self.marks[r:] + self.marks[:r])
            if k > best:
                best, arg = k, r
        return self._result("POSITION: rotation",
                            "mark sequence wrapped around the block",
                            f"rotate {arg}", best, self.N, self.N - 1)

    def test_block_permutation(self, block_sizes: Sequence[int] = (10, 20, 23)) -> HypothesisResult:
        """
        Column or block transposition.

        Multi-column answer sheets (this one is 4 columns of 40, each split into
        two sub-blocks of 20) invite exactly this: a candidate or a scanner
        associating a contiguous run of answers with the wrong block.
        """
        best, arg, tests = 0, None, 0
        for size in block_sizes:
            nb = math.ceil(self.N / size)
            if nb > 6:
                continue
            blocks = [self.marks[i * size:(i + 1) * size] for i in range(nb)]
            for perm in itertools.permutations(range(nb)):
                tests += 1
                seq = [x for b in perm for x in blocks[b]]
                k = self._score_aligned(seq)
                if k > best:
                    best, arg = k, f"blocks of {size}, order {perm}"
        return self._result("POSITION: block/column swap",
                            "answers associated with the wrong column or sub-block",
                            arg or "-", best, self.N, max(tests, 1))

    # -- SYMBOL errors -----------------------------------------------------

    def test_option_permutation(self) -> HypothesisResult:
        """
        Bubble-column mis-registration or a mis-built key legend.

        NOT covered by the multiset-invariance theorem, because relabelling
        options changes the mark counts by construction. Enumerated exhaustively
        -- there are only C! = 24 possibilities.
        """
        best, arg = 0, None
        for p in itertools.permutations(self.opts):
            m = dict(zip(self.opts, p))
            k = self._score_aligned([m[s] if s else None for s in self.marks])
            if k > best:
                best, arg = k, "".join(p)
        return self._result("SYMBOL: option relabelling",
                            "bubble columns mis-registered, or key legend wrong",
                            f"ABCD -> {arg}", best, self.N, math.factorial(self.C))

    def test_symbol_and_displacement(self, min_overlap: int = 10) -> HypothesisResult:
        """Both failures at once -- the largest search space, so the harshest
        multiplicity correction. Included for completeness."""
        best, arg, tests = (0, 1), None, 0
        for p in itertools.permutations(self.opts):
            m = dict(zip(self.opts, p))
            mapped = [m[s] if s else None for s in self.marks]
            for d in range(-(self.N - min_overlap), self.N - min_overlap + 1):
                k, n = self._score_offset(mapped, d)
                if n < min_overlap:
                    continue
                tests += 1
                if binom_sf(k, n, self.chance) < binom_sf(best[0], best[1], self.chance):
                    best, arg = (k, n), f"ABCD -> {''.join(p)}, offset {d:+d}"
        return self._result("COMBINED: symbol + displacement",
                            "two independent mechanical failures on one sheet",
                            arg or "-", best[0], best[1], max(tests, 1))

    def run_all(self) -> List[HypothesisResult]:
        out = [
            self.test_displacement(),
            self.test_reversal(),
            self.test_rotation(),
            self.test_block_permutation(),
            self.test_option_permutation(),
            self.test_symbol_and_displacement(),
        ]
        return sorted(out, key=lambda r: r.corrected_p)

    # ==========================================================================
    # 4. THE INVARIANCE TEST -- the whole position family, in one shot
    # ==========================================================================

    def invariance_test(self, abilities: Sequence[float] = (0.95, 0.85, 0.75, 0.65, 0.55),
                        n_mc: int = 20000) -> List[Dict]:
        """
        THE MOST IMPORTANT TEST IN THIS MODULE.

        Re-ordering marks cannot create or destroy an 'A'. So if these marks are
        a re-ordering of a competent candidate's answers -- by ANY position
        mechanism, including ones nobody has thought of -- their COUNTS must
        look like a competent candidate's answer counts.

        For a candidate of ability theta answering this key:

            E[count of option o] = theta*key_count(o)
                                 + (1-theta)*(N - key_count(o))/(C-1)

        The null is simulated exactly. Chi-square asymptotics are not used, because with C = 4 cells the asymptotic approximation is
        unreliable in precisely the tail we care about.

        A rejection here rules out the entire position family simultaneously,
        with NO search and therefore NO multiplicity correction.
        """
        kc = {o: self.key.count(o) for o in self.opts}
        sc = {o: self.marks.count(o) for o in self.opts}
        rows = []
        for th in abilities:
            exp = {o: th * kc[o] + (1 - th) * (self.N - kc[o]) / (self.C - 1)
                   for o in self.opts}
            chi = sum((sc[o] - exp[o]) ** 2 / exp[o] for o in self.opts)
            hits = 0
            for _ in range(n_mc):
                c = {o: 0 for o in self.opts}
                for k in self.key:
                    if self.rng.random() < th:
                        c[k] += 1
                    else:
                        c[self.rng.choice([x for x in self.opts if x != k])] += 1
                if sum((c[o] - exp[o]) ** 2 / exp[o] for o in self.opts) >= chi:
                    hits += 1
            rows.append({
                "theta": th,
                "expected": dict(exp),
                "observed": dict(sc),
                "chi2": chi,
                "p_value": (hits + 1) / (n_mc + 1),
            })
        return rows

    def run_structure_test(self, n_mc: int = 20000) -> Dict:
        """
        Is this sequence 'answer-like' at all?

        A displaced copy of a genuine answer sequence has the SAME run-length
        structure as a genuine answer sequence -- displacement moves a sequence,
        it does not smooth it. So if the marks are far blockier than any real
        answer pattern, the sheet was not produced by displacing real answers.

        Long uniform runs are the classic signature of patterned bubbling: the
        'straight line down the column' a candidate produces when out of time or
        answering at random.
        """
        def runs(s):
            r, c = [], 1
            for i in range(1, len(s)):
                if s[i] == s[i - 1]:
                    c += 1
                else:
                    r.append(c); c = 1
            r.append(c); return r

        rk, rs = runs(self.key), runs(self.marks)
        mean_s = sum(rs) / len(rs)
        hits = 0
        for _ in range(n_mc):
            d = [self.rng.choice(self.opts) for _ in range(self.N)]
            r = runs(d)
            if sum(r) / len(r) >= mean_s:
                hits += 1
        return {
            "key_runs": len(rk), "key_longest": max(rk), "key_mean": sum(rk) / len(rk),
            "mark_runs": len(rs), "mark_longest": max(rs), "mark_mean": mean_s,
            "p_more_blocky_than_random": (hits + 1) / (n_mc + 1),
        }


# ==============================================================================
# 5. REPORT
# ==============================================================================


def full_report(key, marks, options=("A", "B", "C", "D")) -> str:
    tax = MechanicalErrorTaxonomy(key, marks, options)
    L, w = [], 78
    L.append("=" * w)
    L.append("MECHANICAL ERROR TAXONOMY -- EXHAUSTIVE TEST".center(w))
    L.append("=" * w)
    identity = sum(1 for i in range(len(key)) if marks[i] == key[i])
    L.append(f"Items: {len(key)}   options: {len(options)}   "
             f"score as marked: {identity}/{len(key)}")
    L.append(f"Chance expectation: {len(key)/len(options):.1f}/{len(key)}")
    L.append("")

    L.append("-" * w)
    L.append("A. STRUCTURED TRANSFORMATIONS  (each corrected for its search space)")
    L.append("-" * w)
    L.append(f"  {'family':<34}{'best':>7}{'of':>5}{'raw p':>9}{'tests':>7}{'corr. p':>9}")
    for r in tax.run_all():
        L.append(f"  {r.family:<34}{r.n_correct:>7}{r.n_items:>5}"
                 f"{r.raw_p:>9.4f}{r.n_tests:>7}{r.corrected_p:>9.4f}")
        L.append(f"      best: {r.best_transform:<40} -> {r.verdict}")
    L.append("")

    L.append("-" * w)
    L.append("B. INVARIANCE TEST  (disposes of the ENTIRE position family at once)")
    L.append("-" * w)
    L.append("  Re-ordering marks cannot create or destroy an option. So if these")
    L.append("  marks are a re-ordering of a competent candidate's answers -- by ANY")
    L.append("  mechanism, enumerated or not -- the mark COUNTS must match.")
    L.append("")
    kc = {o: list(key).count(o) for o in options}
    sc = {o: list(marks).count(o) for o in options}
    L.append(f"  {'ability':>8} | " + "".join(f"{'exp ' + o:>8}" for o in options) + "    p-value")
    L.append("  " + "-" * 60)
    for row in tax.invariance_test():
        flag = "  <-- INCOMPATIBLE" if row["p_value"] < 0.05 else ""
        L.append(f"  {row['theta']:>8.2f} | "
                 + "".join(f"{row['expected'][o]:>8.1f}" for o in options)
                 + f"   {row['p_value']:>7.4f}{flag}")
    L.append(f"  {'OBSERVED':>8} | " + "".join(f"{sc[o]:>8d}" for o in options))
    L.append(f"  {'key has':>8} | " + "".join(f"{kc[o]:>8d}" for o in options))
    L.append("")

    L.append("-" * w)
    L.append("C. RUN-STRUCTURE TEST  (is this sequence answer-like at all?)")
    L.append("-" * w)
    rs = tax.run_structure_test()
    L.append(f"  key     : {rs['key_runs']:>3} runs, longest {rs['key_longest']}, "
             f"mean length {rs['key_mean']:.2f}")
    L.append(f"  marks   : {rs['mark_runs']:>3} runs, longest {rs['mark_longest']}, "
             f"mean length {rs['mark_mean']:.2f}")
    L.append(f"  P(marks this blocky | uniform random bubbling) = "
             f"{rs['p_more_blocky_than_random']:.4f}")
    L.append("  Displacement MOVES a sequence; it does not smooth it. Marks far")
    L.append("  blockier than any real answer pattern were not produced by")
    L.append("  displacing real answers.")
    L.append("=" * w)
    return "\n".join(L)


if __name__ == "__main__":
    import os, sys
    _H = os.path.dirname(os.path.abspath(__file__))
    for _p in (_H, os.path.dirname(_H)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from omr_shift import load_case_records

    _rec = load_case_records()
    key = [r["correct"] for r in _rec]
    marks = [r["student"] for r in _rec]
    text = full_report(key, marks)
    print(text)
    _out = os.path.join(os.path.dirname(_H), "results", "error_families.txt")
    with open(_out, "w") as f:
        f.write(text)
