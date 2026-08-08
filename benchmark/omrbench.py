#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 OMRBench -- a falsification benchmark for OMR shift-error detection
================================================================================

PURPOSE
-------
This benchmark exists to BREAK detectors, not to validate them. Every generator
is built to violate an assumption that shift-detection methods rely on. A method
that survives the suite has earned confidence; a method tuned to one generator
will be visibly exposed by the per-generator table.

Accordingly the headline output is NEVER a single pooled number. "92% power" is
precisely the statistic that hides the conditions under which a method fails.
Results are reported per generator, always.

WHAT IT MEASURES
----------------
Three families, because an examination board needs all three and accuracy alone
is not enough to act on:

  DETECTION   false-alarm rate; power by error magnitude.

  HARM        marks wrongly AWARDED (damages examination integrity) and marks
              wrongly WITHHELD (damages the candidate). These are the board's
              actual loss function, they are not symmetric, and a single
              accuracy number cannot express them.

  TRANSPARENCY  three objective proxies for explainability:
              - schema completeness: does the method report each required field
                at all? (a method that cannot name a shift location cannot be
                defended at an appeal)
              - localisation error: when it accepts, how far is the reported
                change point from the true one?
              - CALIBRATION (Brier score, expected calibration error): when a
                method says it is 95% confident, is it right 95% of the time?
                This is the decisive transparency metric. A confidence that is
                not calibrated is not evidence; it is decoration.

DESIGN NOTE -- WHY GENERATOR DIVERSITY, NOT SAMPLE SIZE
-------------------------------------------------------
Measured on the reference implementation before this benchmark was written:
moving from a clean generator to one with correlated items, attractive
distractors and ability drift left the false-alarm rate unchanged (0.000 in
both) but cut power from 0.975 to 0.887 and the shifted-sheet signal from 10.30
to 7.30. Sample size buys precision on a quantity that is already stable;
generator diversity buys the axis along which methods actually differ. One
thousand sheets over ten generators is worth more than ten thousand over one.

REAL DATA
---------
`RealDataSet` defines the schema for confirmed historical shift cases. It is
deliberately empty. Fifty real confirmed re-marks would be worth more than any
number of synthetic sheets, because they calibrate the two quantities
simulation cannot supply: the true base rate of slips and the true distribution
of displacement magnitudes. Skiena & Sumazin (2004) measured the former at 1.8%
across 101,265 SATs; everything else here is assumption until real data arrives.

Usage:
    python3 omrbench.py                # quick run
    python3 omrbench.py --full         # publication-scale run
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
import os
import random
import sys
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance  # noqa: E402
from omr_shift import (  # noqa: E402
    AdjudicationConfig, Adjudicator, Alignment, BandedPairHMM,
    CoherenceScanStatistic, EvidenceEngine, ResponseSheet, ScoringModel,
    SegmentAnalyzer,
    clopper_pearson_upper,
)

OPTIONS = ("A", "B", "C", "D")
N_ITEMS = 46


# ==============================================================================
# 1. GROUND TRUTH AND THE UNIVERSAL DECISION SCHEMA
# ==============================================================================


@dataclass
class GroundTruth:
    """What actually happened to the sheet."""
    has_error: bool
    events: List[Tuple[int, int]]          # (at_question 1-based, delta offset)
    true_alignment: Dict[int, int]         # question idx -> row idx (0-based)
    true_score: int                        # score under the TRUE alignment
    observed_score: int                    # score as naively graded
    generator: str
    injector: str


@dataclass
class Decision:
    """
    THE UNIVERSAL OUTPUT SCHEMA.

    Every detector must return this. A board cannot act on a bare verdict: it
    must be able to state where the shift was, how confident the system is, on
    what evidence, and why. Methods that cannot fill these fields are not
    disqualified -- they are scored on it, which is the point.
    """
    accepted: bool
    shift_locations: List[Dict] = field(default_factory=list)
    confidence: Optional[float] = None          # P(sheet has a shift), in [0,1]
    evidence: Dict = field(default_factory=dict)
    explanation: str = ""
    alignment: Dict[int, int] = field(default_factory=dict)
    awarded_score: int = 0

    REQUIRED_FIELDS = ("accepted", "shift_locations", "confidence",
                       "evidence", "explanation", "alignment")

    def completeness(self) -> float:
        """Fraction of the schema this method actually populates."""
        got = 0
        got += 1                                             # accepted: always
        got += 1 if (self.shift_locations or not self.accepted) else 0
        got += 1 if self.confidence is not None else 0
        got += 1 if self.evidence else 0
        got += 1 if self.explanation else 0
        got += 1 if (self.alignment or not self.accepted) else 0
        return got / len(self.REQUIRED_FIELDS)


# ==============================================================================
# 2. GENERATOR SUITE -- each one attacks a specific assumption
# ==============================================================================


class Generator:
    """Produces (key, marks) for a candidate who made NO mechanical error."""

    name = "base"
    description = ""
    violates = "nothing"

    def __init__(self, ability: float = 0.85, n: int = N_ITEMS) -> None:
        self.ability = ability
        self.n = n

    def _key(self, rng) -> List[str]:
        return [rng.choice(OPTIONS) for _ in range(self.n)]

    def generate(self, rng) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    @staticmethod
    def _wrong(rng, k) -> str:
        return rng.choice([o for o in OPTIONS if o != k])


class CleanGenerator(Generator):
    name = "clean"
    description = "constant ability, uniform distractors, independent items"
    violates = "nothing -- this is the detector's own model"

    def generate(self, rng):
        key = self._key(rng)
        return key, [k if rng.random() < self.ability else self._wrong(rng, k) for k in key]


class IRT2PLGenerator(Generator):
    name = "irt_2pl"
    description = "item difficulty and discrimination vary (2PL)"
    violates = "A1: items are exchangeable given ability"

    def generate(self, rng):
        key = self._key(rng)
        theta = math.log(self.ability / (1 - self.ability))
        marks = []
        for k in key:
            b, a = rng.gauss(0, 1.2), max(0.3, rng.gauss(1.0, 0.4))
            p = 1 / (1 + math.exp(-a * (theta - b)))
            marks.append(k if rng.random() < p else self._wrong(rng, k))
        return key, marks


class TopicClusteredGenerator(Generator):
    name = "topic_clustered"
    description = "items grouped by topic in blocks of 5; difficulty correlated within block"
    violates = "A1: conditional independence of items"

    def generate(self, rng):
        key = self._key(rng)
        theta = math.log(self.ability / (1 - self.ability))
        blocks = [rng.gauss(0, 1.1) for _ in range(self.n // 5 + 1)]
        marks = []
        for i, k in enumerate(key):
            p = 1 / (1 + math.exp(-(theta + blocks[i // 5])))
            marks.append(k if rng.random() < p else self._wrong(rng, k))
        return key, marks


class AttractiveDistractorGenerator(Generator):
    name = "attractive_distractor"
    description = "one distractor per item takes 65% of the wrong answers"
    violates = "A2: wrong answers spread uniformly over distractors"

    def generate(self, rng):
        key = self._key(rng)
        marks = []
        for k in key:
            if rng.random() < self.ability:
                marks.append(k)
            else:
                d = [o for o in OPTIONS if o != k]
                marks.append(d[(ord(k) - 65) % 3] if rng.random() < 0.65 else rng.choice(d))
        return key, marks


class NonStationaryGenerator(Generator):
    name = "nonstationary_ability"
    description = "ability drifts downward across the paper (fatigue / time pressure)"
    violates = "A3: a single ability applies across the whole paper"

    def generate(self, rng):
        key = self._key(rng)
        theta = math.log(self.ability / (1 - self.ability))
        marks = []
        for i, k in enumerate(key):
            p = 1 / (1 + math.exp(-(theta - 2.2 * (i / self.n))))
            marks.append(k if rng.random() < p else self._wrong(rng, k))
        return key, marks


class TwoRegimeGenerator(Generator):
    name = "two_regime"
    description = "strong on the first half, collapses on the second"
    violates = "A3, and MIMICS a shift boundary -- the hardest confound"

    def generate(self, rng):
        key = self._key(rng)
        cut = rng.randint(self.n // 3, 2 * self.n // 3)
        marks = []
        for i, k in enumerate(key):
            p = 0.92 if i < cut else 0.30
            marks.append(k if rng.random() < p else self._wrong(rng, k))
        return key, marks


class TimeTruncatedGenerator(Generator):
    name = "time_truncated"
    description = "runs out of time and bubbles a fixed pattern for the tail"
    violates = "the response model entirely, for part of the sheet"

    def generate(self, rng):
        key = self._key(rng)
        cut = rng.randint(self.n // 2, self.n - 6)
        marks = [k if rng.random() < self.ability else self._wrong(rng, k)
                 for k in key[:cut]]
        pat = rng.choice(OPTIONS)
        marks += [pat] * (self.n - cut)
        return key, marks


class StreakyGuesserGenerator(Generator):
    name = "streaky_guesser"
    description = "guesses in long runs of the same option throughout"
    violates = "independence of successive responses"

    def generate(self, rng):
        key = self._key(rng)
        marks = []
        while len(marks) < self.n:
            marks += [rng.choice(OPTIONS)] * rng.randint(2, 6)
        return key, marks[:self.n]


class OptionBiasGenerator(Generator):
    name = "option_bias"
    description = "when unsure, always picks the same option"
    violates = "A2, and inflates accidental alignment with key-heavy stretches"

    def generate(self, rng):
        key = self._key(rng)
        fav = rng.choice(OPTIONS)
        marks = []
        for k in key:
            if rng.random() < self.ability:
                marks.append(k)
            else:
                marks.append(fav if rng.random() < 0.7 else self._wrong(rng, k))
        return key, marks


class AdversarialAdaptableGenerator(Generator):
    name = "adversarial_adaptable"
    description = "Skiena & Sumazin's maximum-adaptability string L=(01101001)*, mapped to 4 options"
    violates = "deliberately engineered to farm generous shift correction"

    def generate(self, rng):
        key = self._key(rng)
        # L = (01101001)* over a binary alphabet; lift to 4 symbols by pairing.
        pattern = "01101001"
        off = rng.randrange(len(pattern))
        bits = [pattern[(i + off) % len(pattern)] for i in range(self.n * 2)]
        marks = [OPTIONS[int(bits[2 * i]) * 2 + int(bits[2 * i + 1])] for i in range(self.n)]
        return key, marks


GENERATORS: List[Generator] = [
    CleanGenerator(), IRT2PLGenerator(), TopicClusteredGenerator(),
    AttractiveDistractorGenerator(), NonStationaryGenerator(), TwoRegimeGenerator(),
    TimeTruncatedGenerator(), StreakyGuesserGenerator(), OptionBiasGenerator(),
    AdversarialAdaptableGenerator(),
]


# ==============================================================================
# 3. ERROR INJECTORS -- ground truth by construction
# ==============================================================================


class ErrorInjector:
    name = "none"
    description = "no mechanical error"

    def inject(self, rng, key, marks) -> Tuple[List[str], GroundTruth]:
        n = len(key)
        align = {q: q for q in range(n)}
        s = sum(1 for q in range(n) if marks[q] == key[q])
        return list(marks), GroundTruth(False, [], align, s, s, "", self.name)


class RowSkipInjector(ErrorInjector):
    """Physically simulate skipping `mag` bubble rows at a question."""

    def __init__(self, mag: int = 1):
        self.mag = mag
        self.name = f"row_skip_{mag}"
        self.description = f"candidate skipped {mag} bubble row(s) mid-paper"

    def inject(self, rng, key, marks):
        n = len(key)
        at = rng.randint(6, n - 12)
        # A skipped row is left empty. Filling it at random modelled a
        # different event than the controls certify; see large_synthetic.
        obs = marks[:at] + [None for _ in range(self.mag)] + marks[at:]
        obs = obs[:n]
        align = {q: (q if q < at else q + self.mag) for q in range(n)}
        align = {q: r for q, r in align.items() if r < n}
        true_s = sum(1 for q, r in align.items()
                     if obs[r] is not None and obs[r] == key[q])
        obs_s = sum(1 for q in range(n) if obs[q] is not None and obs[q] == key[q])
        return obs, GroundTruth(True, [(at + 1, self.mag)], align, true_s, obs_s, "", self.name)


class DoubleShiftInjector(ErrorInjector):
    name = "double_shift"
    description = "two independent slip events on one sheet"

    def inject(self, rng, key, marks):
        n = len(key)
        a1 = rng.randint(5, n // 2 - 4)
        a2 = rng.randint(n // 2 + 2, n - 10)
        obs = marks[:a1] + [rng.choice(OPTIONS)] + marks[a1:a2] + [rng.choice(OPTIONS)] + marks[a2:]
        obs = obs[:n]
        align = {}
        for q in range(n):
            r = q + (0 if q < a1 else (1 if q < a2 else 2))
            if r < n:
                align[q] = r
        true_s = sum(1 for q, r in align.items()
                     if obs[r] is not None and obs[r] == key[q])
        obs_s = sum(1 for q in range(n) if obs[q] is not None and obs[q] == key[q])
        return obs, GroundTruth(True, [(a1 + 1, 1), (a2 + 1, 1)], align, true_s, obs_s, "", self.name)


INJECTORS: List[ErrorInjector] = [
    ErrorInjector(), RowSkipInjector(1), RowSkipInjector(2), DoubleShiftInjector(),
]


# ==============================================================================
# 4. BASELINE DETECTORS -- all speaking the same schema
# ==============================================================================


class Detector:
    name = "base"
    description = ""

    def decide(self, key: Sequence[str], marks: Sequence[str]) -> Decision:
        raise NotImplementedError

    @staticmethod
    def _score(key, marks, align: Dict[int, int]) -> int:
        return sum(1 for q, r in align.items() if r < len(marks) and marks[r] == key[q])


class NoOpDetector(Detector):
    """The floor. Never corrects anything. A method that cannot beat this on
    HARM is worse than doing nothing -- an essential baseline that accuracy-only
    benchmarks routinely omit."""
    name = "no-op (never correct)"

    def decide(self, key, marks):
        n = len(key)
        align = {q: q for q in range(n)}
        return Decision(False, [], 0.0, {"none": 0.0}, "no correction attempted",
                        align, self._score(key, marks, align))


class BruteForceShiftDetector(Detector):
    """Best single global shift, accepted if it gains more than `gain_threshold`
    marks. The method most people reach for."""
    name = "brute-force shift"

    def __init__(self, gain_threshold: int = 5, max_d: int = 3):
        self.g, self.D = gain_threshold, max_d

    def decide(self, key, marks):
        n = len(key)
        base = sum(1 for i in range(n) if marks[i] == key[i])
        best, bd = base, 0
        for d in range(-self.D, self.D + 1):
            if d == 0:
                continue
            k = sum(1 for q in range(n) if 0 <= q + d < n and marks[q + d] == key[q])
            if k > best:
                best, bd = k, d
        acc = (best - base) >= self.g
        align = ({q: q + bd for q in range(n) if 0 <= q + bd < n} if acc
                 else {q: q for q in range(n)})
        return Decision(
            acc,
            [{"at_question": 1, "offset_before": 0, "offset_after": bd}] if acc else [],
            min(1.0, max(0.0, (best - base) / max(1, n * 0.3))),   # ad-hoc, uncalibrated
            {"score_gain": best - base, "best_offset": bd},
            f"global shift {bd:+d} gains {best-base} marks" if acc else "no shift gains enough",
            align, self._score(key, marks, align))


class LCSDetector(Detector):
    """Maximally generous: score = longest common subsequence. Included
    specifically because Skiena & Sumazin (2004) Sec.7 prove it is exploitable
    -- a random sheet scores 0.58-0.75 where chance gives 0.25."""
    name = "LCS (maximally generous)"

    def decide(self, key, marks):
        n = len(key)
        prev = [0] * (n + 1)
        for i in range(1, n + 1):
            cur = [0] * (n + 1)
            for j in range(1, n + 1):
                cur[j] = prev[j-1] + 1 if marks[i-1] == key[j-1] else max(prev[j], cur[j-1])
            prev = cur
        L = prev[-1]
        base = sum(1 for i in range(n) if marks[i] == key[i])
        return Decision(
            L > base, [], None,           # no location, no calibrated confidence
            {"lcs": L},
            f"LCS grading awards {L}",
            {}, L)


class FixedCostDPDetector(Detector):
    """Affine-gap alignment with arbitrary hand-set costs -- correct algorithm,
    no probabilistic semantics, so no defensible confidence."""
    name = "fixed-cost DP alignment"

    def __init__(self, gap_open: float = 4.0, gap_extend: float = 1.0,
                 match: float = 1.0, mismatch: float = -1.0, max_d: int = 3,
                 accept_gain: float = 3.0):
        self.go, self.ge = gap_open, gap_extend
        self.match, self.mismatch = match, mismatch
        self.D, self.accept_gain = max_d, accept_gain

    def _align(self, key, marks):
        """Gotoh affine-gap alignment over the banded lattice, constant costs.

        No probabilistic interpretation: the numbers below are chosen, and the
        output is an unnormalised score. That is the point of this baseline --
        it is structurally correct and has no confidence to report.
        """
        N, M = len(key), len(marks)
        NEG = float("-inf")
        # (score, backpointer) per state; states are M(atch), X(gap in rows),
        # Y(gap in questions).
        prev = {}
        cur = {}
        best_end, best_score = None, NEG
        traceback = {}

        def in_band(q, r):
            return abs(q - r) <= self.D and 0 <= r <= M

        init = {}
        init[(0, "M")] = 0.0
        prev = init
        for q in range(1, N + 1):
            cur = {}
            for r in range(max(0, q - self.D), min(M, q + self.D) + 1):
                sub = (self.match if (r >= 1 and marks[r - 1] is not None
                                      and marks[r - 1] == key[q - 1])
                       else self.mismatch)
                cands = []
                for st in ("M", "X", "Y"):
                    v = prev.get((r - 1, st))
                    if v is not None and r >= 1:
                        cands.append((v + sub, st))
                if cands:
                    v, bp = max(cands)
                    cur[(r, "M")] = v
                    traceback[(q, r, "M")] = bp
                # X: question q consumes no row
                cands = []
                for st, pen in (("M", self.go), ("X", self.ge), ("Y", self.go)):
                    v = prev.get((r, st))
                    if v is not None:
                        cands.append((v - pen, st))
                if cands:
                    v, bp = max(cands)
                    cur[(r, "X")] = v
                    traceback[(q, r, "X")] = bp
                # Y: row r consumes no question
                cands = []
                for st, pen in (("M", self.go), ("X", self.go), ("Y", self.ge)):
                    v = cur.get((r - 1, st))
                    if v is not None and r >= 1:
                        cands.append((v - pen, st))
                if cands:
                    v, bp = max(cands)
                    cur[(r, "Y")] = v
                    traceback[(q, r, "Y")] = bp
            prev = cur

        # Best terminal cell, then walk the pointers back for the pairing.
        end = max(prev.items(), key=lambda kv: kv[1], default=(None, NEG))
        if end[0] is None:
            return {q: q for q in range(N)}
        r, st = end[0]
        pairs = {}
        q = N
        while q > 0:
            if st == "M":
                pairs[q - 1] = r - 1
                nxt = traceback.get((q, r, "M"), "M")
                q, r, st = q - 1, r - 1, nxt
            elif st == "X":
                nxt = traceback.get((q, r, "X"), "M")
                q, st = q - 1, nxt
            else:
                nxt = traceback.get((q, r, "Y"), "M")
                r, st = r - 1, nxt
            if r < 0:
                break
        return {q_: r_ for q_, r_ in pairs.items() if 0 <= r_ < M}

    def decide(self, key, marks):
        pairs = self._align(key, marks)
        base = sum(1 for i in range(len(key))
                   if marks[i] is not None and marks[i] == key[i])
        got = self._score(key, marks, pairs)
        acc = (got - base) >= self.accept_gain
        cfg = replace(AdjudicationConfig(), max_displacement=self.D)
        sheet = ResponseSheet(tuple(key), tuple(marks))
        al = Alignment(pairs,
                       blank_questions=[q for q in range(len(key)) if q not in pairs],
                       orphan_rows=[r for r in range(len(marks))
                                    if r not in set(pairs.values())],
                       log_score=float(got))
        segs = SegmentAnalyzer(sheet, cfg).segments(al)
        locs = [{"at_question": b.q_start + 1, "offset_before": a.offset,
                 "offset_after": b.offset} for a, b in zip(segs, segs[1:])] if acc else []
        return Decision(acc, locs, None, {"alignment_gain": got - base},
                        "alignment DP with fixed gap costs",
                        pairs if acc else {q: q for q in range(len(key))},
                        got if acc else base)


class GatedPairHMMDetector(Detector):
    """The reference method: banded pair-HMM + Bayes factor + Monte-Carlo
    calibration + per-segment coherence + per-item posterior thresholds."""
    name = "gated pair-HMM (reference)"

    def __init__(self, cfg: Optional[AdjudicationConfig] = None, n_perm: int = 1200):
        self.cfg = cfg or replace(AdjudicationConfig(), external_ability=0.85)
        self.n_perm = n_perm

    def decide(self, key, marks):
        sheet = ResponseSheet(tuple(key), tuple(marks))
        # early_stop and fast are both EXACT here: the benchmark only needs the
        # accept/reject decision and the awarded marks. early_stop abandons a
        # null once the remaining draws provably cannot change it; fast skips
        # calibration entirely on sheets whose MAP registration is the identity,
        # which have already failed a gate. Neither can alter a verdict, and the
        # only cost is that a rejected sheet carries no coherence statistic.
        adj = Adjudicator(sheet, self.cfg).run(n_permutations=self.n_perm,
                                               verbose=False, early_stop=True,
                                               fast=True)
        locs = [{"at_question": c["at_question"], "offset_before": c["offset_before"],
                 "offset_after": c["offset_after"], "mechanism": c["mechanism"]}
                for c in adj.change_points] if adj.accepted else []
        expl = (f"{adj.verdict}; gates: " +
                ", ".join(f"{k}={'pass' if v['passed'] else 'fail'}"
                          for k, v in adj.gates.items()))
        return Decision(
            adj.accepted, locs, adj.evidence["posterior_h1"],
            {"log10_bayes_factor": adj.evidence["log10_bayes_factor"],
             "monte_carlo_p": adj.calibration["p_value"],
             "coherence_statistic": (
                 adj.calibration["observed_statistic"]
                 if adj.calibration.get("computed", True) else None)},
            expl, adj.awarded_map, adj.adjudicated_score)


# ==============================================================================
# 5. METRICS
# ==============================================================================


def brier(conf: Sequence[float], truth: Sequence[bool]) -> Optional[float]:
    pairs = [(c, t) for c, t in zip(conf, truth) if c is not None]
    if not pairs:
        return None
    return sum((c - (1.0 if t else 0.0)) ** 2 for c, t in pairs) / len(pairs)


def ece(conf: Sequence[float], truth: Sequence[bool], bins: int = 10) -> Optional[float]:
    """Expected calibration error: does 'p% confident' mean 'right p% of the time'?"""
    pairs = [(c, t) for c, t in zip(conf, truth) if c is not None]
    if not pairs:
        return None
    tot, n = 0.0, len(pairs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, t) for c, t in pairs if (lo <= c < hi or (b == bins - 1 and c == 1.0))]
        if not sel:
            continue
        acc = sum(1 for _, t in sel if t) / len(sel)
        avg = sum(c for c, _ in sel) / len(sel)
        tot += (len(sel) / n) * abs(acc - avg)
    return tot


@dataclass
class CellResult:
    generator: str
    detector: str
    n_null: int
    n_error: int
    false_alarms: int
    detections: int
    marks_wrongly_awarded: float
    marks_wrongly_withheld: float
    awarded_on_clean: float          # per error-free sheet
    awarded_on_error: float          # per sheet that contains an error
    localisation_errors: List[float]
    completeness: float
    brier: Optional[float]
    ece: Optional[float]

    @property
    def fpr(self):
        return self.false_alarms / self.n_null if self.n_null else float("nan")

    @property
    def power(self):
        return self.detections / self.n_error if self.n_error else float("nan")

    def expected_awarded(self, base_rate: float) -> float:
        """Unearned marks per sheet at a stated base rate of errors.

        The benchmark deliberately over-samples error sheets, so a raw mean over
        its cells is not what a board would see. This reweights to the rate they
        actually expect."""
        return (1 - base_rate) * self.awarded_on_clean + base_rate * self.awarded_on_error

    @property
    def median_localisation(self):
        if not self.localisation_errors:
            return None
        s = sorted(self.localisation_errors)
        return s[len(s) // 2]


# ==============================================================================
# 6. HARNESS
# ==============================================================================


class OMRBench:
    def __init__(self, generators=None, detectors=None, injectors=None,
                 n_per_cell: int = 30, seed: int = 20260804, n_items: int = N_ITEMS):
        self.generators = generators or GENERATORS
        self.detectors = detectors or [
            NoOpDetector(), BruteForceShiftDetector(), LCSDetector(),
            FixedCostDPDetector(), GatedPairHMMDetector(),
        ]
        self.injectors = injectors or INJECTORS
        self.n = n_per_cell
        self.seed = seed
        self.n_items = n_items

    def _build_cases(self, gen, rng):
        """Pre-build the sheets ONCE so every detector sees identical data."""
        cases = []
        for inj in self.injectors:
            for _ in range(self.n):
                key, marks = gen.generate(rng)
                obs, gt = inj.inject(rng, key, marks)
                gt = replace(gt, generator=gen.name)
                cases.append((key, obs, gt))
        return cases

    def run(self, verbose: bool = True) -> List[CellResult]:
        results = []
        for gen in self.generators:
            # zlib.crc32 is stable across processes. Python randomises string hashing
            # per process, which would make every run draw different sheets.
            rng = random.Random(self.seed + zlib.crc32(gen.name.encode()) % 10000)
            cases = self._build_cases(gen, rng)
            if verbose:
                print(f"  generator {gen.name:<24} ({len(cases)} sheets) ...", flush=True)
            for det in self.detectors:
                fa = det_ct = 0
                n_null = n_err = 0
                awarded = withheld = aw_cln = aw_err = 0.0
                loc, comp, confs, truths = [], [], [], []
                for key, obs, gt in cases:
                    d = det.decide(key, obs)
                    comp.append(d.completeness())
                    confs.append(d.confidence)
                    truths.append(gt.has_error)
                    if gt.has_error:
                        n_err += 1
                        det_ct += bool(d.accepted)
                        if d.accepted and d.shift_locations and gt.events:
                            loc.append(min(abs(l.get("at_question", 0) - gt.events[0][0])
                                           for l in d.shift_locations))
                    else:
                        n_null += 1
                        fa += bool(d.accepted)
                    # HARM: signed error against the score the candidate deserves
                    delta = d.awarded_score - gt.true_score
                    awarded += max(0, delta)
                    withheld += max(0, -delta)
                    if gt.has_error:
                        aw_err += max(0, delta)
                    else:
                        aw_cln += max(0, delta)
                results.append(CellResult(
                    gen.name, det.name, n_null, n_err, fa, det_ct,
                    awarded / max(1, len(cases)), withheld / max(1, len(cases)),
                    aw_cln / max(1, n_null), aw_err / max(1, n_err),
                    loc, sum(comp) / len(comp), brier(confs, truths), ece(confs, truths)))
        return results


# ==============================================================================
# 7. REAL DATA SLOT -- deliberately empty
# ==============================================================================


@dataclass
class RealDataSet:
    """
    Schema for confirmed historical shift cases from an examination board.

    Empty until a board contributes. Fifty confirmed re-marks would outweigh any
    quantity of synthetic sheets, because they calibrate what simulation cannot:
    the true base rate of slips, and the true distribution of displacement
    magnitudes and start positions. Everything synthetic here is assumption.
    """
    cases: List[Dict] = field(default_factory=list)

    SCHEMA = {
        "sheet_id": "str",
        "key": "list[str] -- the answer key actually applied at marking",
        "marks": "list[str|None], in PHYSICAL ROW order",
        "scanner_output": "list[str|None] -- raw read, before any manual cleanup",
        "manual_remark": "list[str|None] -- what a human read from the paper sheet",
        "confirmed_alignment": "dict[question_index -> row_index], from manual re-mark",
        "confirmation_source": "str -- how the truth was established",
        "cause": "candidate_slip | scanner_misread | booklet_or_key | none",
        "displacement": "int -- signed rows, where one was confirmed",
        "start_question": "int|None -- where the error began",
        "section_bounds": "list[(int,int)] -- question ranges per subject, if known",
        "n_options": "int",
    }

    def load(self, path: str) -> "RealDataSet":
        if os.path.exists(path):
            with open(path) as f:
                self.cases = json.load(f)
        return self

    def base_rate(self) -> Optional[float]:
        if not self.cases:
            return None
        return sum(1 for c in self.cases
                   if any(int(q) != int(r) for q, r in c["confirmed_alignment"].items())
                   ) / len(self.cases)


# ==============================================================================
# 8. REPORTING -- per generator, never pooled
# ==============================================================================


def recovery(results: List[CellResult], detector: str,
             baseline: str = "no-op (never correct)") -> float:
    """Share of the marks an error costs that `detector` returns.

    The no-op withholds every recoverable mark by definition, so it anchors 0%.
    This is the figure quoted as "recovery" in the documents; it is computed here
    in code, so it is reproducible from the committed output.
    """
    base = [r for r in results if r.detector == baseline]
    det = [r for r in results if r.detector == detector]
    if not base or not det:
        return float("nan")
    b = sum(r.marks_wrongly_withheld for r in base) / len(base)
    d = sum(r.marks_wrongly_withheld for r in det) / len(det)
    return (b - d) / b if b else float("nan")


def render(results: List[CellResult], n_per_cell: int) -> str:
    L, w = [], 108
    dets = sorted({r.detector for r in results}, key=lambda d: [
        x.name for x in (NoOpDetector(), BruteForceShiftDetector(), LCSDetector(),
                         FixedCostDPDetector(), GatedPairHMMDetector())].index(d))
    L.append("=" * w)
    L.append("OMRBench -- PER-GENERATOR OPERATING CHARACTERISTICS".center(w))
    L.append("=" * w)
    L.append(f"{n_per_cell} sheets per (generator x injector) cell. Every detector sees "
             f"IDENTICAL sheets.")
    L.append("Results are NEVER pooled across generators: a single averaged number hides")
    L.append("exactly the conditions under which a method fails.")
    L.append("")
    L.append("  FPR    false-alarm rate on error-free sheets       (lower is better)")
    L.append("  POW    power, over all injected error types        (higher is better)")
    L.append("  AWARD  mean marks WRONGLY AWARDED per sheet        (harms integrity)")
    L.append("  AW@BR  the same, reweighted to a 1.8% base rate of errors")
    L.append("  REC    share of lost marks RETURNED vs the no-op    (higher is better)")
    L.append("  HOLD   mean marks WRONGLY WITHHELD per sheet       (harms the candidate)")
    L.append("  LOC    median localisation error, questions        (lower is better)")
    L.append("  CMP    decision-schema completeness, 0-1           (transparency)")
    L.append("  BRIER  comparative confidence metric               (lower is better)")
    L.append("  ECE    expected calibration error, comparative     (lower is better)")
    L.append("")
    L.append("  All sheets below are SYNTHETIC. Nothing here is a confirmed historical")
    L.append("  case, so none of it establishes operational performance.")
    L.append("")
    L.append("  BRIER and ECE rank detectors on identical data. They do NOT establish")
    L.append("  deployment calibration: these cells hold three error sheets per")
    L.append("  error-free sheet, while the reported confidence uses a 1.8% base rate")
    L.append("  as its prior. A base-rate-thresholded posterior cannot be validated on")
    L.append("  a test set whose base rate is 75%.")
    L.append("")
    L.append(f"  Each FPR below is over {n_per_cell} error-free sheets. An observed 0.000")
    L.append(f"  bounds the true rate below {clopper_pearson_upper(0, n_per_cell)*100:.1f}% at 95% confidence")
    L.append("  (Clopper-Pearson, exact). Zero observed is not zero risk.")
    L.append("")

    for gen in sorted({r.generator for r in results}):
        meta = next((g for g in GENERATORS if g.name == gen), None)
        L.append("-" * w)
        L.append(f"GENERATOR: {gen}")
        if meta:
            L.append(f"  {meta.description}")
            L.append(f"  violates: {meta.violates}")
        L.append("-" * w)
        L.append(f"  n = {results[0].n_null} error-free and {results[0].n_error} "
                 f"error sheets per detector in this cell.")
        L.append(f"  {'detector':<28}{'FPR':>8}{'(fp/n)':>9}{'POW':>8}{'(d/n)':>9}"
                 f"{'AWARD':>8}{'AW@BR':>8}{'HOLD':>8}{'LOC':>6}{'BRIER':>8}")
        for d in dets:
            r = next((x for x in results if x.generator == gen and x.detector == d), None)
            if r is None:
                continue
            loc = "-" if r.median_localisation is None else f"{r.median_localisation:.0f}"
            br = "-" if r.brier is None else f"{r.brier:.3f}"
            ec = "-" if r.ece is None else f"{r.ece:.3f}"
            L.append(f"  {d:<28}{r.fpr:>8.3f}"
                     f"{f'({r.false_alarms}/{r.n_null})':>9}"
                     f"{r.power:>8.3f}{f'({r.detections}/{r.n_error})':>9}"
                     f"{r.marks_wrongly_awarded:>8.2f}{r.expected_awarded(0.018):>8.3f}"
                     f"{r.marks_wrongly_withheld:>8.2f}{loc:>6}{br:>8}")
        L.append("")
    L.append("-" * w)
    L.append("RECOVERY, POOLED ACROSS GENERATORS")
    L.append("-" * w)
    L.append("  Share of the marks an error costs that each detector returns, with the")
    L.append("  no-op anchoring 0%. This is the figure quoted in the documents.")
    L.append("")
    L.append(f"  {'detector':<28}{'recovery':>10}{'AWARD (raw)':>14}{'AWARD @1.8%':>14}")
    for d in dets:
        rs = [r for r in results if r.detector == d]
        if not rs:
            continue
        L.append(f"  {d:<28}{recovery(results, d):>9.0%}"
                 f"{sum(r.marks_wrongly_awarded for r in rs)/len(rs):>14.2f}"
                 f"{sum(r.expected_awarded(0.018) for r in rs)/len(rs):>14.3f}")
    L.append("")
    L.append("  AWARD (raw) is a mean over benchmark cells, which hold three error")
    L.append("  sheets per error-free sheet. A board screening at the historical 1.8%")
    L.append("  base rate would see the reweighted figure instead.")
    L.append("=" * w)
    return "\n".join(L)


def render_figure(results, dets, outdir: str) -> Optional[str]:
    """Recovery against worst-case false-positive rate, one point per detector.

    Values come from the same objects that produce the table above, so the
    figure cannot disagree with it.
    """
    try:
        import matplotlib.pyplot as plt
        from figstyle import (SURFACE, INK, ACCENT, CONTEXT, ANNOT, frame,
                              title, footnote)
    except ImportError:
        return None

    LABEL = {
        "no-op (never correct)": ("No correction", "right"),
        "brute-force shift": ("Global displacement scan", "right"),
        "fixed-cost DP alignment": ("Fixed-cost affine alignment", "left"),
        "LCS (maximally generous)": ("Longest common subsequence", "left"),
        "gated pair-HMM (reference)": ("Gated pair HMM", "right"),
    }
    pts = []
    for d in dets:
        rs = [r for r in results if r.detector == d]
        if not rs or d not in LABEL:
            continue
        name, side = LABEL[d]
        pts.append((name, 100.0 * recovery(results, d),
                    max(r.fpr for r in rs), side, "Gated" in name))

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.20)
    frame(ax)
    # Labels are placed beside their point. Where two points sit at a similar
    # false-positive rate the labels would overlap, so they are nudged apart
    # vertically. Which points collide depends on the data, so this is computed
    # rather than positioned by hand.
    for i, (name, x, y, side, hl) in enumerate(pts):
        ax.scatter([x], [y], s=190 if hl else 78,
                   color=ACCENT if hl else CONTEXT,
                   edgecolor=SURFACE, linewidth=1.8, zorder=4)
        near = [j for j, o in enumerate(pts)
                if j != i and abs(o[2] - y) < 0.06 and abs(o[1] - x) < 45]
        dy = 0
        if near:
            dy = 11 if x <= min(pts[j][1] for j in near) else -11
        dx = 13 if side == "right" else -13
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center",
                    fontsize=ANNOT, color=ACCENT if hl else INK)
    ax.set_xlabel("marks returned to wrongly-scored candidates  (%)", labelpad=9)
    ax.set_ylabel("worst-case false-positive rate", labelpad=9)
    title(fig, ax, "Recovery against worst-case false-positive rate")
    ax.set_xlim(-9, 116)
    ax.set_ylim(-0.11, 1.16)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    n_null = sum(r.n_null for r in results if r.detector == dets[0])
    footnote(fig, f"{sum(r.n_null + r.n_error for r in results if r.detector == dets[0])} "
                  f"synthetic sheets \u00b7 ten candidate models \u00b7 "
                  f"{len(dets)} detectors on identical data\n"
                  f"Worst-case FPR is the maximum across generators; an observed "
                  f"0.00 covers {n_null} error-free sheets")
    path = os.path.join(outdir, "figures", "recovery_vs_fpr.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="publication-scale run")
    ap.add_argument("--n", type=int, default=None, help="sheets per cell")
    args = ap.parse_args()
    # 12 per cell. This benchmark reports every figure per generator and never
    # pools them, because a pooled average hides the generator under which a
    # method fails; 12 error-free sheets is what makes a cell separately
    # reportable. It is a weak bound on its own -- 0 of 12 bounds that cell at
    # 22.1% -- so this benchmark ranks the models and the large corpus bounds
    # the winner's rate. --full raises it to 40 (22.1% -> 7.2% per cell) at
    # proportional cost.
    n = args.n or (40 if args.full else 12)

    print("=" * 78)
    print("OMRBench")
    print("=" * 78)
    print(f"generators: {len(GENERATORS)}   injectors: {len(INJECTORS)}   "
          f"sheets/cell: {n}")
    print(f"total sheets: {len(GENERATORS) * len(INJECTORS) * n}\n")

    bench = OMRBench(n_per_cell=n)
    results = bench.run()
    text = render(results, n)
    print()
    print(text)

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    provenance.write_text(os.path.join(here, "benchmark.txt"), text,
                          sheets_per_cell=n, seed=bench.seed)
    provenance.write_json(
        os.path.join(here, "benchmark.json"),
        [{**r.__dict__, "fpr": r.fpr, "power": r.power,
          "median_localisation": r.median_localisation} for r in results],
        sheets_per_cell=n, seed=bench.seed)
    fig = render_figure(results, DETECTOR_ORDER if "DETECTOR_ORDER" in globals()
                        else sorted({r.detector for r in results}), here)
    print(f"\nwrote benchmark.txt and benchmark.json"
          + (f" and {os.path.basename(fig)}" if fig else ""))

    rd = RealDataSet()
    print(f"\nReal-data slot: {len(rd.cases)} confirmed cases loaded.")
    print("Schema for contribution:")
    for k, v in RealDataSet.SCHEMA.items():
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
