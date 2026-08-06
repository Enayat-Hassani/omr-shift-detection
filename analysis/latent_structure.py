#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 LATENT STRUCTURE SEARCH
 Is there ANY hidden regularity, or are these answers incorrect?
================================================================================

The preceding modules test ENUMERATED hypotheses: shift, reversal, rotation,
block swap, option relabelling. That approach has an inherent blind spot -- it
can only find transformations somebody thought to write down.

This module asks the complementary, assumption-light question:

    Does the response sequence contain ANY structure -- of any kind -- that is
    unlikely to arise from a candidate answering incorrectly?

Five independent lenses, chosen because they fail in different ways and share
almost no assumptions. Convergence across them would be meaningful; so would
unanimous silence.

  1. MUTUAL INFORMATION SCAN
     Strictly more general than enumerating the 24 option permutations. MI
     detects ANY statistical dependence between mark and key at a given lag,
     including PARTIAL and NOISY relabellings that no bijection would capture
     (e.g. "confuses A with B but handles C and D correctly"). If a wrongly
     mapped key had been applied, MI would be elevated even though the raw match
     count is at chance.

  2. LEMPEL-ZIV COMPLEXITY
     Is the mark sequence algorithmically patterned? A candidate bubbling a
     repeating figure produces a highly compressible string. This is a property
     of the marks alone -- it never looks at the key, so it cannot be fooled by
     anything to do with alignment.

  3. LOCAL TRANSFORMATION SEARCH
     Every (window x displacement x option-map) triple, local as well as global. A
     mechanical error confined to part of the sheet would show here and nowhere
     else. Multiplicity over the whole search space is corrected exactly.

  4. POSITIONAL-RULE TESTS
     Did the candidate answer by rule, not by knowledge -- cycling
     A,B,C,D, or keying off the question number? Tests dependence of the mark on
     (question number mod m).

  5. MULTI-METHOD CHANGE-POINT CONSENSUS
     Four structurally different break-point detectors -- CUSUM, binary
     segmentation, the pair-HMM posterior, and the coherence scan -- run
     independently. Agreement among methods that share no machinery is real
     evidence; disagreement is the signature of noise.

INTERPRETATION DISCIPLINE
-------------------------
This module can only ever justify FURTHER INVESTIGATION. It cannot justify
changing a score: it is an open-ended search over a large space, so something
will always be the maximum. That is exactly why every test here is reported with
a multiplicity-corrected p-value, and why the consensus analysis is included --
a single flagged breakpoint from one method is noise; the same breakpoint from
four unrelated methods is a finding.
================================================================================
"""

from __future__ import annotations

import itertools
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from error_families import binom_sf, sidak   # noqa: E402

OPTIONS = ("A", "B", "C", "D")


# ==============================================================================
# 1. MUTUAL INFORMATION -- transformation-agnostic dependence
# ==============================================================================


def mutual_information(x: Sequence[str], y: Sequence[str],
                       alphabet: Sequence[str] = OPTIONS) -> float:
    """MI(X;Y) in nats. Zero iff X and Y are statistically independent."""
    n = len(x)
    if n == 0:
        return 0.0
    joint: Dict[Tuple[str, str], int] = {}
    px: Dict[str, int] = {}
    py: Dict[str, int] = {}
    for a, b in zip(x, y):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        px[a] = px.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
    mi = 0.0
    for (a, b), c in joint.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((px[a] / n) * (py[b] / n)))
    return mi


class MutualInformationScan:
    """
    MI between the marks and the key at every displacement, with an exact
    permutation null.

    Why this beats enumerating option permutations: a wrongly built key legend,
    a partially confused candidate, or a scanner that mis-reads one option only,
    all create DEPENDENCE without creating MATCHES. Match-counting is blind to
    them; MI is not.
    """

    def __init__(self, key: Sequence[str], marks: Sequence[str],
                 max_lag: Optional[int] = None, seed: int = 20260804) -> None:
        self.key, self.marks = list(key), list(marks)
        self.n = len(key)
        self.max_lag = max_lag if max_lag is not None else self.n - 10
        self.rng = random.Random(seed)

    def run(self, n_perm: int = 4000) -> Dict:
        rows = []
        for d in range(-self.max_lag, self.max_lag + 1):
            idx = [q for q in range(self.n) if 0 <= q + d < self.n]
            if len(idx) < 12:
                continue
            xs = [self.marks[q + d] for q in idx]
            ys = [self.key[q] for q in idx]
            rows.append({"lag": d, "n": len(idx), "mi": mutual_information(xs, ys),
                         "xs": xs, "ys": ys})
        obs = max(rows, key=lambda r: r["mi"])
        # Permutation null for the MAXIMUM over lags -- multiplicity built in.
        null = []
        for _ in range(n_perm):
            shuf = self.marks[:]
            self.rng.shuffle(shuf)
            best = 0.0
            for r in rows:
                idx = [q for q in range(self.n) if 0 <= q + r["lag"] < self.n]
                mi = mutual_information([shuf[q + r["lag"]] for q in idx],
                                        [self.key[q] for q in idx])
                best = max(best, mi)
            null.append(best)
        p = (sum(1 for v in null if v >= obs["mi"]) + 1) / (n_perm + 1)
        null.sort()
        return {"best_lag": obs["lag"], "best_mi": obs["mi"], "n": obs["n"],
                "p_value": p, "null_mean": sum(null) / len(null),
                "null_q999": null[min(len(null) - 1, int(0.999 * len(null)))],
                "identity_mi": next(r["mi"] for r in rows if r["lag"] == 0)}


# ==============================================================================
# 2. LEMPEL-ZIV COMPLEXITY -- is the sequence patterned in itself?
# ==============================================================================


def lz76_complexity(s: Sequence[str]) -> int:
    """Number of distinct phrases in the Lempel-Ziv (1976) parsing."""
    seq, i, c, l = list(s), 0, 1, 1
    k, kmax = 1, 1
    n = len(seq)
    while l + k <= n:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1
                l += kmax
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    if k != 1:
        c += 1
    return c


class ComplexityTest:
    """Compares the marks' algorithmic complexity to random bubbling.

    Looks only at the marks -- never at the key -- so no alignment assumption
    can affect it."""

    def __init__(self, marks: Sequence[str], seed: int = 20260804) -> None:
        self.marks = list(marks)
        self.rng = random.Random(seed)

    def run(self, n_mc: int = 20000) -> Dict:
        obs = lz76_complexity(self.marks)
        n = len(self.marks)
        null = [lz76_complexity([self.rng.choice(OPTIONS) for _ in range(n)])
                for _ in range(n_mc)]
        # LOW complexity = patterned; that is the alternative of interest.
        p = (sum(1 for v in null if v <= obs) + 1) / (n_mc + 1)
        return {"lz_complexity": obs, "null_mean": sum(null) / len(null),
                "p_more_patterned_than_random": p}


# ==============================================================================
# 3. LOCAL TRANSFORMATION SEARCH
# ==============================================================================


class LocalTransformSearch:
    """
    Every (contiguous window) x (displacement) x (option relabelling).

    A mechanical error affecting only part of the sheet -- one column, one page,
    one block interrupted by a distraction -- would appear here and in no global
    test. The price is a very large search space, corrected for exactly.
    """

    def __init__(self, key, marks, max_d: int = 3, min_len: int = 6):
        self.key, self.marks = list(key), list(marks)
        self.n = len(key)
        self.D, self.L = max_d, min_len

    def run(self, include_option_maps: bool = True) -> Dict:
        maps = list(itertools.permutations(OPTIONS)) if include_option_maps else [OPTIONS]
        best, tests = None, 0
        for perm in maps:
            m = dict(zip(OPTIONS, perm))
            mapped = [m[s] for s in self.marks]
            for d in range(-self.D, self.D + 1):
                idx = [q for q in range(self.n) if 0 <= q + d < self.n]
                if len(idx) < self.L:
                    continue
                cum = [0] * (len(idx) + 1)
                for i, q in enumerate(idx):
                    cum[i + 1] = cum[i] + (1 if mapped[q + d] == self.key[q] else 0)
                for i in range(len(idx)):
                    for j in range(i + self.L, len(idx) + 1):
                        tests += 1
                        nn, kk = j - i, cum[j] - cum[i]
                        if kk * 4 <= nn:
                            continue
                        p = binom_sf(kk, nn, 0.25)
                        if best is None or p < best["raw_p"]:
                            best = {"perm": "".join(perm), "lag": d,
                                    "q_start": idx[i] + 1, "q_end": idx[j - 1] + 1,
                                    "n": nn, "k": kk, "raw_p": p}
        best["n_tests"] = tests
        best["corrected_p"] = sidak(best["raw_p"], tests)
        return best


# ==============================================================================
# 4. POSITIONAL-RULE TESTS
# ==============================================================================


class PositionalRuleTest:
    """Did the candidate answer by a positional rule?

    Tests whether the mark depends on (question number mod m) -- the signature
    of cycling A,B,C,D or of any index-driven bubbling strategy."""

    def __init__(self, marks: Sequence[str], seed: int = 20260804):
        self.marks = list(marks)
        self.rng = random.Random(seed)

    def run(self, mods: Sequence[int] = (2, 3, 4, 5), n_mc: int = 8000) -> List[Dict]:
        out = []
        n = len(self.marks)
        for m in mods:
            pos = [str(i % m) for i in range(n)]
            obs = mutual_information(self.marks, pos, OPTIONS)
            null = []
            for _ in range(n_mc):
                s = self.marks[:]
                self.rng.shuffle(s)
                null.append(mutual_information(s, pos, OPTIONS))
            out.append({"mod": m, "mi": obs,
                        "p_value": (sum(1 for v in null if v >= obs) + 1) / (n_mc + 1)})
        return out


# ==============================================================================
# 5. MULTI-METHOD CHANGE-POINT CONSENSUS
# ==============================================================================


class ChangePointConsensus:
    """
    Four break-point detectors that share no machinery.

    The logic is deliberate: any single detector run over 46 positions will
    return SOMETHING, because a maximum always exists. What carries evidential
    weight is agreement between methods built on different principles. Scatter
    is the signature of noise.
    """

    def __init__(self, key, marks, max_d: int = 3):
        self.key, self.marks = list(key), list(marks)
        self.n = len(key)
        self.D = max_d
        self.correct = [1 if marks[i] == key[i] else 0 for i in range(self.n)]

    def cusum(self) -> int:
        """Break point maximising the CUSUM deviation of the correctness track."""
        mean = sum(self.correct) / self.n
        run, best, arg = 0.0, 0.0, 0
        for i, c in enumerate(self.correct):
            run += c - mean
            if abs(run) > best:
                best, arg = abs(run), i + 1
        return arg

    def binary_segmentation(self) -> int:
        """Break point maximising the two-sample mean difference (t-like)."""
        best, arg = -1.0, 0
        for t in range(5, self.n - 5):
            a, b = self.correct[:t], self.correct[t:]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            stat = abs(ma - mb) * math.sqrt(len(a) * len(b) / self.n)
            if stat > best:
                best, arg = stat, t + 1
        return arg

    def hmm_posterior(self) -> Optional[int]:
        """Where the pair-HMM's posterior displacement first leaves zero."""
        try:
            from omr_shift import (AdjudicationConfig, EvidenceEngine,
                                   ResponseSheet, ScoringModel)
            from dataclasses import replace as _replace
            cfg = _replace(AdjudicationConfig(), external_ability=0.85)
            sheet = ResponseSheet(tuple(self.key), tuple(self.marks))
            post = EvidenceEngine(ScoringModel(sheet, cfg)).posterior_mean_offsets()
            for q in range(self.n):
                d = max(post[q].items(), key=lambda kv: kv[1])[0]
                if d not in (0, None):
                    return q + 1
            return None
        except Exception:
            return None

    def coherence_scan(self) -> Optional[int]:
        """Start of the most coherent displaced block."""
        try:
            from omr_shift import AdjudicationConfig, CoherenceScanStatistic, ResponseSheet
            cfg = AdjudicationConfig()
            _, w = CoherenceScanStatistic(cfg, 4).compute(
                ResponseSheet(tuple(self.key), tuple(self.marks)))
            return w["q_start"] if w else None
        except Exception:
            return None

    def run(self) -> Dict:
        cand = {"CUSUM": self.cusum(), "binary segmentation": self.binary_segmentation(),
                "pair-HMM posterior": self.hmm_posterior(),
                "coherence scan": self.coherence_scan()}
        vals = [v for v in cand.values() if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else None
        # agreement = how many methods fall within +/-3 questions of the median
        agree = None
        if vals:
            s = sorted(vals); med = s[len(s) // 2]
            agree = sum(1 for v in vals if abs(v - med) <= 3)
        return {"candidates": cand, "spread": spread, "n_agreeing": agree,
                "n_methods": len(vals)}


# ==============================================================================
# 6. REPORT
# ==============================================================================


def report(key: Sequence[str], marks: Sequence[str]) -> str:
    L, w = [], 76
    L.append("=" * w)
    L.append("LATENT STRUCTURE SEARCH".center(w))
    L.append("=" * w)
    L.append("Is there ANY hidden regularity in this response sequence, or are the")
    L.append("answers incorrect? Five independent lenses, sharing few assumptions.")
    L.append("")

    L.append("-" * w)
    L.append("1. MUTUAL INFORMATION SCAN  (detects ANY dependence, matches included)")
    L.append("-" * w)
    mi = MutualInformationScan(key, marks).run()
    L.append(f"  MI at the graded alignment (lag 0) : {mi['identity_mi']:.4f} nats")
    L.append(f"  best MI over all lags              : {mi['best_mi']:.4f} nats "
             f"at lag {mi['best_lag']:+d} (n={mi['n']})")
    L.append(f"  permutation null                   : mean {mi['null_mean']:.4f}, "
             f"99.9th pct {mi['null_q999']:.4f}")
    L.append(f"  p-value (max over lags, multiplicity built in) : {mi['p_value']:.4f}")
    L.append("  A wrongly built key legend, a partial option confusion, or a scanner")
    L.append("  mis-reading one option would ALL raise MI without raising the match")
    L.append("  count. This test is blind to none of them.")
    L.append("")

    L.append("-" * w)
    L.append("2. LEMPEL-ZIV COMPLEXITY  (looks only at the marks, never the key)")
    L.append("-" * w)
    cx = ComplexityTest(marks).run()
    L.append(f"  LZ76 complexity of the marks : {cx['lz_complexity']}")
    L.append(f"  random bubbling, mean        : {cx['null_mean']:.2f}")
    L.append(f"  P(this patterned | random)   : {cx['p_more_patterned_than_random']:.4f}")
    L.append("")

    L.append("  CONTROL -- the candidate as their own control. LZ complexity is")
    L.append("  INVARIANT UNDER DISPLACEMENT: moving a sequence does not compress it.")
    L.append("  So if these marks were displaced real answers, their complexity would")
    L.append("  equal that of the candidate's real answers elsewhere on the sheet.")
    try:
        from PRIVATE.full_sheet_analysis import FULL_SHEET
        import random as _r
        _rng = _r.Random(9)
        _null = sorted(lz76_complexity([_rng.choice(OPTIONS) for _ in range(46)])
                       for _ in range(8000))
        def _pv(x): return (sum(1 for v in _null if v <= x) + 1) / (len(_null) + 1)
        L.append(f"  {'equal-length window':<24}{'LZ':>5}{'p(patterned)':>14}")
        for nm, st in [("maths Q1-46", 0), ("other Q47-Q92", 46),
                       ("other Q73-Q118", 72), ("other Q99-Q144", 98)]:
            seg = FULL_SHEET[st:st + 46]
            if len(seg) == 46:
                L.append(f"  {nm:<24}{lz76_complexity(seg):>5}{_pv(lz76_complexity(seg)):>14.4f}")
        L.append(f"  random baseline mean LZ = {sum(_null)/len(_null):.2f}")
    except Exception:
        L.append("  (This control needs the candidate's marks for the other 104 rows.")
        L.append("   Those were transcribed from the published sheet image without an")
        L.append("   independent check, so they are kept out of the repository rather")
        L.append("   than published as though verified.)")
    L.append("")

    L.append("-" * w)
    L.append("3. LOCAL TRANSFORMATION SEARCH  (window x displacement x option map)")
    L.append("-" * w)
    lt = LocalTransformSearch(key, marks).run()
    L.append(f"  best: Q{lt['q_start']}-Q{lt['q_end']}, lag {lt['lag']:+d}, "
             f"map ABCD->{lt['perm']}")
    L.append(f"        {lt['k']}/{lt['n']} correct, raw p = {lt['raw_p']:.5f}")
    L.append(f"  search space = {lt['n_tests']} tests; corrected p = {lt['corrected_p']:.4f}")
    L.append("")

    L.append("-" * w)
    L.append("4. POSITIONAL-RULE TESTS  (did they answer by index, not knowledge?)")
    L.append("-" * w)
    L.append(f"  {'question no. mod m':>20}{'MI (nats)':>12}{'p-value':>10}")
    for r in PositionalRuleTest(marks).run():
        L.append(f"  {r['mod']:>20}{r['mi']:>12.4f}{r['p_value']:>10.4f}")
    L.append("")

    L.append("-" * w)
    L.append("5. CHANGE-POINT CONSENSUS  (four methods sharing no machinery)")
    L.append("-" * w)
    cp = ChangePointConsensus(key, marks).run()
    for name, v in cp["candidates"].items():
        L.append(f"  {name:<24} -> {'no break point' if v is None else 'Q' + str(v)}")
    L.append(f"  spread across methods    : "
             f"{'n/a' if cp['spread'] is None else str(cp['spread']) + ' questions'}")
    L.append(f"  methods within +/-3 of the median : {cp['n_agreeing']}/{cp['n_methods']}")
    L.append("  A single flagged break point is noise -- a maximum always exists.")
    L.append("  Agreement between methods built on different principles is evidence.")
    L.append("=" * w)
    return "\n".join(L)


if __name__ == "__main__":
    from omr_shift import load_case_records
    _rec = load_case_records()
    key = [r["correct"] for r in _rec]
    marks = [r["student"] for r in _rec]
    text = report(key, marks)
    print(text)
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    with open(os.path.join(here, "latent_structure.txt"), "w") as f:
        f.write(text)
