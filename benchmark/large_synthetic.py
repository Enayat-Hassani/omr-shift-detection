#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 LARGE SYNTHETIC STRESS BENCHMARK  for OMR shift-error detection
===============================================================================

A large, adversarial, SYNTHETIC benchmark that stress-tests the gated pair-HMM
shift detector (and two simple baselines) across candidate ability levels, exam
lengths, realistic error mechanisms, scanner corruption, and a suite of
deliberately adversarial answer patterns.

WHAT THIS BENCHMARK IS, AND IS NOT
-----------------------------------
Entirely synthetic: every sheet has known ground truth constructed by the
generator.  No metric reported here is an operational or deployment calibration
of the detector on real examination paper.  The reference implementation (PE) is a
Bayesian pair-HMM whose confidence is a posterior that places a base-rate prior
over the shift hypothesis; a base-rate-thresholded posterior cannot be validated
on a test set whose shift base rate differs from deployment.  Brier score and
the mean squared posterior are reported here for the same reason the other
benchmark reports them, and under the same restriction: as synthetic
within-corpus rankings between detectors on identical data, NEVER as an absolute
operating characteristic on real paper.

This benchmark reports EVENT COUNTS alongside every rate (a rate without its
numerator hides the uncertainty), gives the exact Clopper-Pearson upper
confidence bound whenever zero false positives are observed, and reports
recovery (marks genuinely restored by realignment) and unearned marks per
mechanism rather than hiding them under a pooled average.

The threshold sweep is run on a DISJOINT corpus from the metrics corpus and is
presented purely as SAFETY / RECOVERY trade-offs for an examination board to
choose between -- NOT as tuning for best performance and then reporting it as
validation.

DETERMINISM
-----------
All randomness is seeded and driven by `random.Random`.  Per-corpus and
per-condition seeds use `zlib.crc32` (stable across runs and machines).
Python's `hash()` is randomised per process and is NEVER used.

USAGE
-----
    python3 benchmark/large_synthetic.py --quick
    python3 benchmark/large_synthetic.py --full
    python3 benchmark/large_synthetic.py --full --n 60 --perm 1500

    --quick   small run, validates the whole pipeline end to end
    --full    research-scale run (a few thousand synthetic sheets total)
    --n N     sheets per (condition x detector) cell   (default 2 quick / 8 full)
    --perm P  Monte-Carlo permutations per null search  (default 1100; must be
              >= ~1000 for alpha=0.001 to be satisfiable; quick caps to 120)
    --jobs J  parallel workers across cells  (default = CPU count, e.g. 8)
    --seed S  master seed (default 20260804)
    --out DIR output directory (default results/large_synthetic)
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUTDIR = os.path.join(_ROOT, "results", "large_synthetic")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_ROOT, ".mplconfig"))

import numpy as np                       # noqa: E402
import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import provenance                          # noqa: E402
from omr_shift import (                   # noqa: E402
    AdjudicationConfig, Adjudicator, ResponseSheet, clopper_pearson_upper,
)
from omrbench import (                    # noqa: E402
    Decision, Detector, NoOpDetector, BruteForceShiftDetector,
    LCSDetector, FixedCostDPDetector,
)

HAVE_PANDAS = True
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
    HAVE_PANDAS = False


def _cp_upper_safe(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact Clopper-Pearson (1-alpha) upper bound for small n; Wilson
    approximation (z_{0.975}) for large n.

    The exact form used to raise OverflowError at corpus scale, which is what
    this wrapper was written to avoid. That is fixed in
    omr_shift.clopper_pearson_upper, which now accumulates the binomial tail in
    log space. Wilson is kept for large n because it is cheaper, not because the
    exact form fails. Switching to exact throughout would move the pooled bounds
    slightly and require a full re-run."""
    if n <= 200:
        return clopper_pearson_upper(k, n, alpha)
    z = 1.6448536269514722
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + width) / denom)

OPTIONS_4 = ("A", "B", "C", "D")
OPTIONS_5 = ("A", "B", "C", "D", "E")
OPTION_SETS = [OPTIONS_4, OPTIONS_5]


# ==============================================================================
# 1. TRUTHFUL CONTENT GENERATORS  --  produce the candidate's real answers
# ==============================================================================


def make_key(rng: random.Random, n: int, options: Sequence[str]) -> List[str]:
    return [rng.choice(options) for _ in range(n)]


def _wrong(rng: random.Random, k: str, options: Sequence[str]) -> str:
    return rng.choice([o for o in options if o != k])


def _logit(p: float) -> float:
    return math.log(max(1e-6, p / (1 - p)))


def b_plain(rng, key, options, ability):
    theta = _logit(ability)
    return [k if rng.random() < 1 / (1 + math.exp(-theta)) else _wrong(rng, k, options)
            for k in key]


def b_2pl(rng, key, options, ability):
    theta = _logit(ability)
    out = []
    for k in key:
        b, a = rng.gauss(0, 1.3), max(0.3, rng.gauss(1.1, 0.4))
        p = 1 / (1 + math.exp(-a * (theta - b)))
        out.append(k if rng.random() < p else _wrong(rng, k, options))
    return out


def b_topic(rng, key, options, ability):
    theta = _logit(ability)
    sw = len(key) // 6 + 1
    blk = [rng.gauss(0, 1.0) for _ in range(sw)]
    out = []
    for i, k in enumerate(key):
        p = 1 / (1 + math.exp(-(theta + blk[i // 6])))
        out.append(k if rng.random() < p else _wrong(rng, k, options))
    return out


def b_timid_decline(rng, key, options, ability):
    theta = _logit(ability)
    n = len(key)
    out = []
    for i, k in enumerate(key):
        p = 1 / (1 + math.exp(-(theta - 2.2 * (i / max(1, n)))))
        if rng.random() < p:
            out.append(k)
        else:
            out.append(rng.choice([o for o in options if o != k]))
    return out


def b_early_strong(rng, key, options, ability):
    cut = rng.randint(len(key) // 3, 2 * len(key) // 3)
    out = []
    for i, k in enumerate(key):
        p = 0.94 if i < cut else 0.30
        out.append(k if rng.random() < p else _wrong(rng, k, options))
    return out


def b_option_bias(rng, key, options, ability):
    fav = rng.choice(options)
    out = []
    for k in key:
        if rng.random() < ability:
            out.append(k)
        else:
            out.append(fav if rng.random() < 0.7 else _wrong(rng, k, options))
    return out


def b_streak_overrun(rng, key, options, ability):
    out, i = [], 0
    n = len(key)
    while i < n:
        run_opt = rng.choice(options)
        for _ in range(rng.randint(2, 7)):
            if i < n:
                if rng.random() < ability:
                    out.append(key[i])
                else:
                    out.append(run_opt if key[i] != run_opt else _wrong(rng, key[i], options))
                i += 1
    return out


def b_time_truncated(rng, key, options, ability):
    cut = rng.randint(len(key) // 2, len(key) - 4)
    out = [k if rng.random() < ability else _wrong(rng, k, options) for k in key[:cut]]
    out += [rng.choice(options)] * (len(key) - cut)
    return out


def b_random(rng, key, options, ability):
    return [rng.choice(options) for _ in key]


def b_consistent(rng, key, options, ability):
    return [k if rng.random() < min(1.0, ability) else _wrong(rng, k, options) for k in key]


def b_mixed(rng, key, options, ability):
    out = []
    mid = len(key) // 2
    for i, k in enumerate(key):
        p = ability * (1.0 if i < mid else 0.6) + (0.04 if i % 9 == 0 else 0.0)
        out.append(k if rng.random() < p else _wrong(rng, k, options))
    return out


BEHAVIORS: Dict[str, Callable] = {
    "plain": b_plain,
    "2pl": b_2pl,
    "topic_block": b_topic,
    "timid_decline": b_timid_decline,
    "early_strong": b_early_strong,
    "option_bias": b_option_bias,
    "streak_overrun": b_streak_overrun,
    "time_truncated": b_time_truncated,
    "random_blitz": b_random,
    "consistent": b_consistent,
    "mixed_effort": b_mixed,
}


# ==============================================================================
# 2. INJECTION MODEL  +  MECHANISM INJECTORS
# ==============================================================================


@dataclass
class Injection:
    observed: List[Optional[str]]      # physical rows (None = unreadable/blank)
    has_shift: bool                    # a genuine mechanical displacement exists
    events: List[Tuple[int, int]] = field(default_factory=list)   # 0-based (q, delta)
    alignment: Dict[int, int] = field(default_factory=dict)       # q -> row
    mechanism: str = "none"
    monotone: bool = True


def inj_single_jump(rng, key, content, options, at, mag, mech):
    """A skipped bubble row leaves the row EMPTY, not filled at random.

    The controls in benchmark/verify_corpus.py and the profile study both model
    the skip as a blank; this used to fill the skipped rows with random options,
    so the controls certified a slightly different event than the corpora
    measured. Blank is the physically honest one: a candidate who skips a row
    does not mark it.
    """
    n = len(key)
    rows = list(content[:at]) + [None for _ in range(abs(mag))] + list(content[at:])
    rows = rows[:n]
    align = {}
    for q in range(n):
        r = q if q < at else q + mag
        if 0 <= r < len(rows):
            align[q] = r
    return Injection(rows, True, [(at, mag)], align, mech)


def inj_none(rng, key, content, options):
    n = len(key)
    align = {q: q for q in range(n)}
    return Injection(list(content), False, [], align, "none")


def inj_row_skip_1(rng, key, content, options):
    n = len(key)
    at = rng.randint(5, max(6, n - 6))
    return inj_single_jump(rng, key, content, options, at, 1, "row_skip_1")


def inj_row_skip_2(rng, key, content, options):
    n = len(key)
    at = rng.randint(5, max(6, n - 7))
    return inj_single_jump(rng, key, content, options, at, 2, "row_skip_2")


def inj_early_full(rng, key, content, options):
    return inj_single_jump(rng, key, content, options, rng.randint(1, 4), 1, "early_full")


def inj_boundary(rng, key, content, options):
    n = len(key)
    at = rng.choice([int(0.46 * n), int(0.90 * n)])
    return inj_single_jump(rng, key, content, options, at, 1, "boundary")


def inj_short_local(rng, key, content, options):
    n = len(key)
    at = rng.randint(4, max(5, n - 14))
    end = min(n - 3, at + rng.randint(5, 10))
    rows = list(content[:at]) + [rng.choice(options)] + list(content[at:end]) + list(content[end + 1:])
    rows = rows[:n]
    align = {}
    for q in range(n):
        if q < at:
            align[q] = q
        elif q < end:
            align[q] = q + 1
        elif q > end:
            align[q] = q
    return Injection(rows, True, [(at, 1), (end, -1)], align, "short_local")


def inj_whole_section(rng, key, content, options):
    n = len(key)
    at = n // 2
    return inj_single_jump(rng, key, content, options, at, 1, "whole_section")


def inj_anxiety(rng, key, content, options):
    n = len(key)
    at = rng.randint(6, n - 14)
    content = list(content)
    for i in range(max(0, at - 4), min(n, at + 6)):
        if rng.random() < 0.55:
            content[i] = _wrong(rng, key[i], options)
    return inj_single_jump(rng, key, content, options, at, 1, "anxiety")


def inj_two_separate(rng, key, content, options):
    n = len(key)
    a1 = rng.randint(5, n // 2 - 4)
    a2 = rng.randint(n // 2 + 2, n - 8)
    rows = (list(content[:a1]) + [rng.choice(options)] + list(content[a1:a2])
            + [rng.choice(options)] + list(content[a2:])[:max(0, n - a2 - 2)])
    rows = rows[:n]
    align = {}
    for q in range(n):
        r = q + (0 if q < a1 else (1 if q < a2 else 2))
        if r < n:
            align[q] = r
    return Injection(rows, True, [(a1, 1), (a2, 1)], align, "double_shift")


def inj_self_corrected(rng, key, content, options):
    n = len(key)
    a = rng.randint(4, max(5, n - 26))
    b = a + rng.randint(15, min(24, n - a - 1))
    rows = list(content[:a]) + [rng.choice(options)] + list(content[a:b]) + list(content[b + 1:])
    rows = rows[:n]
    align = {}
    for q in range(n):
        if q < a:
            align[q] = q
        elif q < b:
            align[q] = q + 1
        else:
            align[q] = q
    return Injection(rows, True, [(a, 1), (b, -1)], align, "self_corrected")


def inj_deferred(rng, key, content, options):
    n = len(key)
    a = rng.randint(3, max(4, n - 12))
    rows = list(content[:a]) + list(content[a + 1:]) + [content[a]]
    rows = rows[:n]
    align = {}
    for q in range(n):
        if q < a:
            align[q] = q
        elif q == a:
            align[q] = n - 1
        else:
            align[q] = q - 1
    return Injection(rows, True, [(a, -1)], align, "deferred", monotone=False)


def inj_isolated(rng, key, content, options):
    n = len(key)
    a = rng.randint(4, n - 4)
    rows = list(content)
    rows[a], rows[a + 1] = rows[a + 1], rows[a]
    align = {q: q for q in range(n)}
    align[a], align[a + 1] = a + 1, a
    return Injection(rows, True, [(a, 0)], align, "isolated", monotone=False)


# ---- scanner corruption (blank rows, deleted rows, misreads) ----------------


def inj_scanner_blank(rng, key, content, options, level):
    n = len(key)
    at = rng.randint(4, n - 4)
    rows = list(content[:at]) + [None] * level + list(content[at:])
    align = {}
    for q in range(n):
        r = q if q < at else q + level
        if 0 <= r < len(rows):
            align[q] = r
    return Injection(rows, True, [(at, level)], align, f"scanner_blank_{level}")


def inj_scanner_row_delete(rng, key, content, options, level):
    n = len(key)
    at = rng.randint(4, n - 4)
    rows = list(content[:at]) + list(content[at + level:])
    align = {}
    for q in range(n):
        r = q if q < at else q - level
        if 0 <= r < len(rows):
            align[q] = r
    return Injection(rows, True, [(at + level, -level)], align, f"scanner_row_delete_{level}")


def inj_scanner_misread(rng, key, content, options, level):
    rows = list(content)
    idx = rng.sample(range(len(rows)), min(level, len(rows)))
    for i in idx:
        if rows[i] is not None:
            rows[i] = _wrong(rng, rows[i], options)
    align = {q: q for q in range(len(key))}
    return Injection(rows, False, [], align, f"scanner_misread_{level}")


def inj_scanner_faint(rng, key, content, options, level):
    rows = list(content)
    idx = rng.sample(range(len(rows)), min(level, len(rows)))
    for i in idx:
        rows[i] = None
    align = {q: q for q in range(len(key))}
    return Injection(rows, False, [], align, f"scanner_faint_{level}")


def inj_scanner_double(rng, key, content, options, level):
    """Two bubbles filled on one row. The scanner cannot resolve which was
    intended, so the row reads as a different option than the candidate meant.

    This was previously identical to inj_scanner_faint -- both blanked the row --
    so the corpus measured faint marks twice and claimed to cover double marks.
    A double mark is not a blank: the row carries a symbol, and a wrong one.
    """
    rows = list(content)
    idx = rng.sample(range(len(rows)), min(level, len(rows)))
    for i in idx:
        intended = rows[i]
        alternatives = [o for o in options if o != intended]
        rows[i] = rng.choice(alternatives) if alternatives else intended
    align = {q: q for q in range(len(key))}
    return Injection(rows, False, [], align, f"scanner_double_{level}")


MECHANISM_FNS: Dict[str, Callable] = {
    "none": inj_none,
    "row_skip_1": inj_row_skip_1,
    "row_skip_2": inj_row_skip_2,
    "early_full": inj_early_full,
    "boundary": inj_boundary,
    "short_local": inj_short_local,
    "whole_section": inj_whole_section,
    "anxiety": inj_anxiety,
    "double_shift": inj_two_separate,
    "self_corrected": inj_self_corrected,
    "deferred": inj_deferred,
    "isolated": inj_isolated,
    "scanner_blank_1": lambda r, k, c, o: inj_scanner_blank(r, k, c, o, 1),
    "scanner_blank_2": lambda r, k, c, o: inj_scanner_blank(r, k, c, o, 2),
    "scanner_row_delete_1": lambda r, k, c, o: inj_scanner_row_delete(r, k, c, o, 1),
    "scanner_row_delete_2": lambda r, k, c, o: inj_scanner_row_delete(r, k, c, o, 2),
    "scanner_misread_1": lambda r, k, c, o: inj_scanner_misread(r, k, c, o, 1),
    "scanner_misread_4": lambda r, k, c, o: inj_scanner_misread(r, k, c, o, 4),
    "scanner_faint_1": lambda r, k, c, o: inj_scanner_faint(r, k, c, o, 1),
    "scanner_faint_4": lambda r, k, c, o: inj_scanner_faint(r, k, c, o, 4),
    "scanner_double_4": lambda r, k, c, o: inj_scanner_double(r, k, c, o, 4),
}

MECHANISMS = list(MECHANISM_FNS.keys())


# ---- adversarial answer-pattern producers (observed marks) -----------------
# These are folded into the mechanism grid below as ordinary cells.  Clean
# adversarial patterns (adv_*) are targets for FALSE-POSITIVE testing; planted
# ones (adv_key_leakage) are genuine shifts by construction.


def inj_adv_long_runs(rng, key, options):
    n = len(key)
    m, i = [], 0
    while i < n:
        opt = rng.choice(options)
        for _ in range(rng.randint(3, 9)):
            if i < n:
                m.append(opt)
                i += 1
    return Injection(m, False, [], {q: q for q in range(n)}, "adv_long_runs")


def inj_adv_favourite(rng, key, options):
    fav = rng.choice(options)
    m = [k if rng.random() < 0.55 else fav for k in key]
    return Injection(m, False, [], {q: q for q in range(len(key))}, "adv_favourite")


def inj_adv_easy_block(rng, key, options):
    n = len(key)
    m = [k if rng.random() < 0.9 else _wrong(rng, k, options) for k in key[: n // 2]]
    m += [rng.choice(options) for _ in range(n - len(m))]
    return Injection(m[:n], False, [], {q: q for q in range(n)}, "adv_easy_block")


def inj_adv_key_leakage(rng, key, options):
    # PLANTED genuine shift: the shifted window coincides with the answer key,
    # the classic "the marks line up with the key if you shift them" case.
    n = len(key)
    shift = rng.choice([1, 2])
    rows = [None] * n
    for q in range(n):
        if q + shift < n:
            rows[q + shift] = key[q]
    for i in range(n):
        if rows[i] is None:
            rows[i] = _wrong(rng, key[i], options)
    align = {q: q + shift for q in range(n) if q + shift < n}
    return Injection(rows, True, [(0, shift)], align, "adv_key_leakage")


def inj_adv_cycle(rng, key, options):
    n = len(key)
    idx = [i % len(options) for i in range(n)]
    off = rng.randrange(1, n)
    m = [options[idx[(i + off) % len(options)]] for i in range(n)]
    return Injection(m, False, [], {q: q for q in range(n)}, "adv_cycle")


ADVERSARIAL = {
    "adv_long_runs": "long option runs",
    "adv_favourite": "always-favourite option",
    "adv_easy_block": "easy-then-guessed block",
    "adv_key_leakage": "shifted-key coincidence (planted shift)",
    "adv_cycle": "cycling option pattern",
}

# fold the adversarial producers into the mechanism table (adapters drop the
# unused `content` argument) so they flow through the same grid + harness
for _adv, _fn in {"adv_long_runs": inj_adv_long_runs,
                  "adv_favourite": inj_adv_favourite,
                  "adv_easy_block": inj_adv_easy_block,
                  "adv_key_leakage": inj_adv_key_leakage,
                  "adv_cycle": inj_adv_cycle}.items():
    MECHANISM_FNS[_adv] = (lambda f: (lambda r, k, c, o: f(r, k, o)))(_fn)

MECHANISMS = list(MECHANISM_FNS.keys())



# ==============================================================================
# 3. DETECTORS  (all implement Detector.decide(key, marks))
# ==============================================================================


class ReferenceGated(Detector):
    """The reference shift detector: banded pair-HMM + Bayes factor +
    Monte-Carlo calibration + per-segment coherence + per-item posterior,
    driven by an explicit AdjudicationConfig (the hook the threshold sweep uses).
    """

    name = "gated pair-HMM (reference)"

    def __init__(
        self,
        cfg: Optional[AdjudicationConfig] = None,
        n_perm: int = 1500,
        options: Sequence[str] = OPTIONS_4,
    ):
        self.cfg = cfg if cfg is not None else AdjudicationConfig()
        self.n_perm = n_perm
        self.options = tuple(options)

    def decide(self, key, marks):
        try:
            sheet = ResponseSheet(tuple(key), tuple(marks), self.options)
            adj = Adjudicator(sheet, self.cfg).run(
                n_permutations=self.n_perm, verbose=False, early_stop=True, fast=True)
        except Exception as exc:
            # The text, not just the fact. A run that silently converts crashes
            # into clean verdicts cannot be audited afterwards.
            return Decision(False, [], None, {"error": f"{type(exc).__name__}: {exc}"},
                            f"adjudicator error: {type(exc).__name__}: {exc}",
                            {}, None)
        locs = [{"at_question": c.get("at_question"),
                 "offset_before": c.get("offset_before"),
                 "offset_after": c.get("offset_after")}
                for c in (adj.change_points or [])]
        gates = {k: bool(g.get("passed")) for k, g in adj.gates.items()}
        ev = dict(adj.evidence or {})
        evidence = {
            "log10_bayes_factor": ev.get("log10_bayes_factor"),
            "posterior_h1": ev.get("posterior_h1"),
            "gates": gates,
            "mc_p": (adj.calibration or {}).get("p_value"),
            "groups": (adj.calibration or {}).get("decisive_null"),
        }
        explanation = " | ".join(
            f"{k}={'ok' if v else 'FAIL'}" for k, v in gates.items())
        return Decision(adj.accepted, locs, ev.get("posterior_h1"), evidence,
                        explanation, adj.awarded_map, adj.adjudicated_score)


class _Baseline(Detector):
    """Tolerantly wraps an existing omrbench detector (4-option, may choke on
    None rows / pathological sheets) so the harness never crashes."""

    def __init__(self, inner: Detector):
        self.inner = inner
        self.verdict = getattr(inner, "name", inner.__class__.__name__)

    def decide(self, key, marks):
        try:
            d = self.inner.decide(list(key), list(marks))
        except Exception:
            return Decision(False, [], None, {}, f"{self.verdict}: decide error", {}, None)
        conf = d.confidence
        return Decision(bool(d.accepted), list(d.shift_locations or []), conf,
                        dict(d.evidence or {}), f"{self.verdict}: {d.explanation or ''}",
                        d.alignment, d.awarded_score)


def make_detector(kind: str, cfg: Optional[AdjudicationConfig], n_perm: int,
                  options: Sequence[str] = OPTIONS_4):
    if kind == "gated":
        return ReferenceGated(cfg, n_perm, options)
    if kind == "bruteforce":
        return _Baseline(BruteForceShiftDetector(max_d=3))
    if kind == "noop":
        return _Baseline(NoOpDetector())
    if kind == "lcs":
        return _Baseline(LCSDetector())
    if kind == "fixedcost":
        return _Baseline(FixedCostDPDetector(max_d=3))
    raise ValueError(kind)


# ==============================================================================
# 4. THRESHOLD POLICIES  (for the DISJOINT sweep corpus)
# ==============================================================================


BASE_CFG = AdjudicationConfig()

POLICIES: Dict[str, Dict] = {
    # The sweep varies the acceptance level and NOTHING else, so its rows are the
    # profiles an examination board can actually select (omr_shift.Policy). A
    # level below Conservative was measured and dropped: at alpha = 0.0005 the
    # derived draw count is 19,999 and one policy row costs 108 minutes, which
    # buys a single point below a curve section 6.1 already covers.
    #
    # External ability is supplied at the generating value throughout, which is
    # the same allowance the metrics arm makes; see ASSUMPTIONS.md A2 for what
    # that costs in interpretation.
    "conservative": dict(permutation_alpha=0.001, external_ability=0.85),
    "balanced": dict(permutation_alpha=0.010, external_ability=0.85),
    "sensitive": dict(permutation_alpha=0.050, external_ability=0.85),
}

# Draws per policy. The count is not independent of the level it tests: the gate
# admits at most floor(alpha*(n+1)-1) exceedances, so a level below 1/(n+1) can
# never be reached. Derived rather than chosen, matching omr_shift.Policy.
POLICY_PERM: Dict[str, int] = {
    "conservative": 9999,
    "balanced": 999,
    "sensitive": 199,
}
POLICY_ORDER = ["conservative", "balanced", "sensitive"]


def _policy_kwargs(name: str) -> Dict:
    """POLICIES entry with its draw count folded in.

    AdjudicationConfig now rejects a permutation_alpha the draw count cannot
    reach, and it validates on construction. Setting alpha first and the count
    second raises on the intermediate object, so both have to go in one call.
    """
    kw = dict(POLICIES[name])
    perm = POLICY_PERM.get(name)
    if perm:
        kw["n_permutations"] = perm
    return kw


def policy_cfg(name: str) -> AdjudicationConfig:
    if name == "default":
        return AdjudicationConfig()
    return replace(AdjudicationConfig(), **_policy_kwargs(name))


# A board does not know a candidate's true ability. It has their record on other
# papers, which is an estimate with error on it. Handing the detector the exact
# value the generator used overstates what it can do, which is why A2 reads the
# detection figures as an upper bound.
#
# `observed_ability` simulates the record instead: PRIOR_ITEMS questions the
# candidate also sat, scored, and the proportion correct taken as the estimate.
# The error is binomial and shrinks as PRIOR_ITEMS grows, so the parameter says
# plainly how much prior evidence a board is assumed to hold.

PRIOR_ITEMS = 40


def observed_ability(true_ability: float, cell_id: int, n_items: int = PRIOR_ITEMS) -> float:
    """Ability as a board would have it: measured on other papers, with error."""
    rng = random.Random(zlib.crc32(f"prior:{cell_id}:{true_ability}".encode()))
    correct = sum(1 for _ in range(n_items) if rng.random() < true_ability)
    # Laplace smoothing keeps the estimate inside the admissible band.
    return (correct + 1) / (n_items + 2)


def reference_for(ability: float, perm: int = 1100, policy: Optional[str] = None,
                  fast: bool = False, options: Sequence[str] = OPTIONS_4,
                  level: Optional[float] = None) -> ReferenceGated:
    """Reference detector. `ability` is what the detector is told, which the
    caller derives from `observed_ability` rather than from the generator, so the
    figures are not inflated by knowledge no board has.

    `level` overrides the acceptance level only, leaving every other threshold
    at its shipped value. That is what makes a before-and-after on the default
    a controlled comparison: one number changes, nothing else does."""
    if policy is None or policy == "default":
        base = AdjudicationConfig()
    else:
        base = replace(AdjudicationConfig(), **_policy_kwargs(policy))
    if level is not None:
        # n_permutations has to move with the level or the config rejects it.
        base = replace(base, permutation_alpha=level,
                       n_permutations=max(perm, math.ceil(10.0 / level) - 1))
    base = replace(base, external_ability=ability)
    if fast:
        # --quick smoke: cheap Monte-Carlo gate, still structurally valid
        perm = min(perm, 120)
        base = replace(base, permutation_alpha=0.05, posterior_shift_threshold=0.85,
                       item_posterior_threshold=0.9, min_segment_length=4,
                       n_permutations=perm)
    return ReferenceGated(base, perm, options)


# ==============================================================================
# 5. CORPUS BUILDERS
# ==============================================================================

METRICS_LENGTHS = [46, 90, 100, 150]
METRICS_ABILITIES = [0.26, 0.40, 0.55, 0.70, 0.85, 0.95]
SWEEP_LENGTHS = [46, 100, 150]
SWEEP_ABILITIES = [0.55, 0.70, 0.85]
# One mechanism per structural family, so the sweep spans the kinds of error the
# model can and cannot represent without paying for all 26: clean, a single
# monotone skip, two separate events, a transient the candidate corrected, a
# scanner artefact, and a block-level displacement. The metrics arm carries the
# full set; the sweep exists to compare profiles, not to characterise mechanisms.
SWEEP_MECHANISMS = ["none", "row_skip_1", "double_shift", "self_corrected",
                    "scanner_blank_1", "whole_section"]
QUICK_LENGTHS = [46]
QUICK_ABILITIES = [0.70]
QUICK_MECHANISMS = [
    "none", "row_skip_1", "early_full", "self_corrected",
    "scanner_misread_1", "adv_long_runs",
]


def sheet_for(rng, options, length, ability, mechanism):
    """A single (key, observed, Injection)."""
    key = make_key(rng, length, options)
    behaviour = rng.choice(list(BEHAVIORS.values()))
    content = behaviour(rng, key, options, ability)
    inj = MECHANISM_FNS[mechanism](rng, key, content, options)
    return key, inj.observed, inj


def cell_rngs(seed: int, cell_id: int, n: int) -> List[random.Random]:
    """Deterministic, independent RNG per sheet inside a cell."""
    return [random.Random(seed + 100000 * (cell_id + 1) + i) for i in range(n)]


def _option_sets(option_sets):
    if not option_sets:
        return [OPTIONS_4]
    if isinstance(option_sets[0], str):
        return [tuple(option_sets)]
    return [tuple(o) for o in option_sets]


def build_metrics_conditions(option_sets, quick: bool = False):
    """Grid: length x ability x mechanism (incl. adversarial kinds)."""
    cells = []
    cid = 0
    lengths = QUICK_LENGTHS if quick else METRICS_LENGTHS
    abilities = QUICK_ABILITIES if quick else METRICS_ABILITIES
    mechanisms = QUICK_MECHANISMS if quick else MECHANISMS
    option_choices = [OPTIONS_4] if quick else _option_sets(option_sets)
    for options in option_choices:
        for length in lengths:
            for ability in abilities:
                for mechanism in mechanisms:
                    cells.append({"id": cid, "options": options, "length": length,
                                  "ability": ability, "mechanism": mechanism})
                    cid += 1
    return cells


def build_sweep_conditions(option_sets, quick: bool = False):
    cells = []
    cid = 0
    lengths = QUICK_LENGTHS if quick else SWEEP_LENGTHS
    abilities = QUICK_ABILITIES if quick else SWEEP_ABILITIES
    mechanisms = ["none", "row_skip_1", "self_corrected"] if quick else SWEEP_MECHANISMS
    option_choices = [OPTIONS_4] if quick else _option_sets(option_sets)
    for options in option_choices:
        for length in lengths:
            for ability in abilities:
                for mechanism in mechanisms:
                    cells.append({"id": cid, "options": options, "length": length,
                                  "ability": ability, "mechanism": mechanism})
                    cid += 1
    return cells


# ==============================================================================
# 6. EVALUATION HARNESS
# ==============================================================================


def event_match(locs: Sequence, events: Sequence[Tuple[int, int]],
                tol: int = 1) -> bool:
    """True if any detected location lies within `tol` questions of a ground
    truth event with a compatible offset (|delta| <= 1)."""
    if not events:
        return bool(locs)
    if not locs:
        return False
    for l in locs:
        if isinstance(l, dict):
            dq = l.get("at_question")
            before = l.get("offset_before") or 0
            after = l.get("offset_after") or 0
            dd = after - before
        elif isinstance(l, (tuple, list)):
            dq, dd = l[0], (l[1] if len(l) > 1 else 0)
        else:
            dq, dd = l, 0
        if dq is None:
            continue
        for (eq, ed) in events:
            if abs(int(dq) - (eq + 1)) <= tol and abs(dd - ed) <= 1:
                return True
    return False


def naive_score(key: Sequence[str], observed: Sequence[Optional[str]]) -> int:
    return sum(1 for q in range(min(len(key), len(observed)))
               if observed[q] is not None and observed[q] == key[q])


def run_corpus(cells, det_factory, det_name, seed, n, dataset, policy,
               failures: Optional[List] = None, failures_max: int = 60) -> List[dict]:
    rows = []
    for cell in cells:
        det = det_factory(cell)
        rngs = cell_rngs(seed, cell["id"], n)
        for i, rng in enumerate(rngs):
            key, observed, inj = sheet_for(rng, cell["options"], cell["length"],
                                           cell["ability"], cell["mechanism"])
            # A crash used to be silently recorded as a rejection: a true
            # negative on a clean sheet, a false negative on an error sheet.
            # The headline "0 false positives" would then be indistinguishable,
            # from the record alone, from "the detector crashed". The error is
            # carried into the row so a run can be audited for it.
            error = ""
            try:
                d = det.decide(key, observed)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:200]
                d = Decision(False, [], None, {}, "harness error", None, None)
            accepted = bool(d.accepted)
            target = inj.has_shift
            matched = event_match(d.shift_locations, inj.events)
            if target:
                outcome = "TP" if (accepted and matched) else (
                    "TP_wrongloc" if accepted else "FN")
            else:
                outcome = "FP" if accepted else "TN"
            conf = d.confidence
            if isinstance(conf, dict):
                conf = conf.get("posterior_h1", 0.0)
            if conf is None:
                conf = float("nan")
            brier = (
                (float(conf) - (1.0 if target else 0.0)) ** 2
                if not math.isnan(float(conf)) else float("nan")
            )
            # Mean squared posterior: a sharpness diagnostic, how far the
            # detector commits away from zero. It was previously the raw
            # posterior, which is a mean, not a mean square.
            mse = float(conf) ** 2 if not math.isnan(float(conf)) else float("nan")
            naive = naive_score(key, observed)
            true_sc = sum(1 for q, r in inj.alignment.items()
                          if 0 <= r < len(observed) and observed[r] is not None
                          and observed[r] == key[q])
            at_stake = max(0, true_sc - naive)
            final_score = int(d.awarded_score) if d.awarded_score is not None else naive
            recovered = max(0, min(final_score, true_sc) - naive) if target else 0
            withheld = max(0, true_sc - final_score)
            unearned = max(0, final_score - true_sc)
            rows.append({
                "dataset": dataset, "detector": det_name, "policy": policy,
                "options": "".join(cell["options"]), "length": cell["length"],
                "ability": cell["ability"], "mechanism": cell["mechanism"],
                "sheet_i": i, "has_shift": int(target), "accepted": int(accepted),
                "matched": int(matched), "outcome": outcome,
                "confidence": round(float(conf), 6), "brier": round(brier, 6),
                "mse": round(mse, 6), "naive_score": naive,
                "true_score": true_sc, "final_score": final_score,
                "at_stake": at_stake, "saved": recovered,
                "withheld": withheld, "unearned": unearned,
                "error": error,
            })
            # collect honest failure / near-miss cases (gated only)
            if failures is not None and det_name == "gated" \
                    and outcome in ("FN", "FP", "TP_wrongloc") and len(failures) < failures_max:
                failures.append({
                    "dataset": dataset, "policy": policy,
                    "condition": {k: cell[k] for k in
                                  ("options", "length", "ability", "mechanism")},
                    "sheet_i": i, "seed": seed,
                    "has_shift": bool(target), "events": inj.events,
                    "outcome": outcome, "accepted": accepted,
                    "shift_locations": d.shift_locations or [],
                    "posterior_h1": round(float(conf), 6) if not math.isnan(conf) else None,
                    "log10_bayes_factor": (d.evidence or {}).get("log10_bayes_factor"),
                    "mc_p": (d.evidence or {}).get("mc_p"),
                    "explanation": d.explanation or "",
                })
    return rows


# -----------------------------------------------------------------------------
# Parallel workers.  Every sheet is RNG-seeded from (seed, cell_id, i), so
# cell slices may be dispatched to any worker with bit-identical results.
# -----------------------------------------------------------------------------


def _cell_detector(cell, det_name, perm, fast, level=None):
    if det_name == "gated":
        return reference_for(observed_ability(cell["ability"], cell["id"]),
                             perm, fast=fast, level=level)
    return make_detector(det_name, None, perm)


def _chunked(cells, k):
    for i in range(0, len(cells), k):
        yield cells[i:i + k]


def _run_batch(args):
    """ProcessPool worker: judge a deterministic slice of cells."""
    cells, det_name, perm, fast, policy, dataset, seed, n, level = args
    if policy is None:
        fac = (lambda c: _cell_detector(c, det_name, perm, fast, level))
        pol = "metrics"
    else:
        fac = (lambda c: reference_for(observed_ability(c["ability"], c["id"]),
                                       perm, policy=policy))
        pol = policy
    failures = []
    rows = run_corpus(cells, fac, det_name, seed, n, dataset, pol, failures)
    return rows, failures


# ==============================================================================
# 7. AGGREGATION
# ==============================================================================


def aggregate(rows: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        k = (r["dataset"], r["detector"], r["policy"], r["options"],
             r["length"], r["ability"], r["mechanism"])
        groups.setdefault(k, []).append(r)
    out = []
    for (dataset, detector, policy, options, length, ability, mechanism), rs in groups.items():
        n = len(rs)
        n_target = sum(r["has_shift"] for r in rs)
        n_clean = n - n_target
        tp_c = sum(r["outcome"] == "TP" for r in rs)
        tp_w = sum(r["outcome"] == "TP_wrongloc" for r in rs)
        fp = sum(r["outcome"] == "FP" for r in rs)
        fn = sum(r["outcome"] == "FN" for r in rs)
        tn = n_clean - fp
        power = tp_c / n_target if n_target else float("nan")
        recall_all = (tp_c + tp_w) / n_target if n_target else float("nan")
        fpr = fp / n_clean if n_clean else float("nan")
        fpr_ci = _cp_upper_safe(fp, n_clean) if n_clean else float("nan")
        # Averaged over the sheets that carry a posterior at all. A detector
        # that reports no confidence (the aligners) contributes no rows here,
        # and one undefined row previously turned the whole cell into NaN.
        scored = [r for r in rs if not math.isnan(r["brier"])]
        n_scored = len(scored)
        brier = (sum(r["brier"] for r in scored) / n_scored
                 if n_scored else float("nan"))
        mse = (sum(r["mse"] for r in scored) / n_scored
               if n_scored else float("nan"))
        at_stake = sum(r["at_stake"] for r in rs)
        saved = sum(r["saved"] for r in rs if r["saved"] is not None)
        saved_n = sum(1 for r in rs if r["saved"] is not None)
        recovery = (saved / at_stake if at_stake else float("nan"))
        unearned = sum(r["unearned"] for r in rs)
        out.append({
            "dataset": dataset, "detector": detector, "policy": policy,
            "options": options, "length": length, "ability": ability,
            "mechanism": mechanism, "n": n, "n_target": n_target,
            "n_clean": n_clean, "tp_correct": tp_c, "tp_wrongloc": tp_w,
            "fp": fp, "fn": fn, "tn": tn, "power": round(power, 6),
            "recall_all": round(recall_all, 6), "fpr": round(fpr, 6),
            "fpr_ci_upper": round(fpr_ci, 6), "brier": round(brier, 6),
            "mse": round(mse, 6), "n_scored": n_scored,
            "recovery": round(recovery, 6),
            "at_stake": at_stake, "saved": saved, "saved_sheets": saved_n,
            "unearned": unearned,
        })
    return out


# ==============================================================================
# 8. FIGURES  (all from the aggregated frames)
# ==============================================================================


def _style(ax):
    ax.grid(True, ls=":", alpha=0.4)


def fig_safety_recovery(dfs, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs, ys, labels = [], [], []
    for pol in POLICY_ORDER:
        d = dfs[dfs["policy"] == pol]
        if d.empty:
            continue
        t = d[d["n_target"] > 0]
        c = d[d["n_clean"] > 0]
        extra = (t["saved"].sum() / t["n_target"].sum()) if len(t) else 0.0
        fpr = (c["fp"].sum() / c["n_clean"].sum()) if len(c) else 0.0
        xs.append(extra); ys.append(fpr); labels.append(pol)
        ax.annotate(pol, (extra, fpr), xytext=(4, 4), textcoords="offset points",
                    fontsize=8)
    if not xs:
        plt.close(fig)
        return
    ax.plot(xs, ys, "o-", color="#4C72B0")
    ax.set_xlabel("mean marks saved per error sheet (recovery)")
    ax.set_ylabel("false-positive rate (clean sheets)")
    ax.set_title("Safety / Recovery trade-off by policy (disjoint sweep corpus)")
    ax.set_xlim(0, max(xs) * 1.15 if xs else 1)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_power_by_ability(dfm, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ab in sorted(dfm["ability"].unique()):
        d = dfm[(dfm["ability"] == ab) & (dfm["n_target"] > 0)]
        if d.empty:
            continue
        g = d.groupby("mechanism")["power"].mean()
        ax.plot(range(len(g)), g, "o-", label=f"ability {ab}")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=60, ha="right")
    ax.set_ylabel("detection power (TP / error sheets)")
    ax.set_title("Power by candidate ability (gated detector, synthetic)")
    ax.legend()
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_recovery_by_mechanism(dfm, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    d = dfm[(dfm["n_target"] > 0) & (~dfm["mechanism"].isin(ADVERSARIAL))]
    g = d.groupby("mechanism").agg(recovery=("recovery", "mean")).dropna()
    g = g.reindex([m for m in MECHANISMS if m in g.index])
    x = range(len(g))
    ax.bar(x, g["recovery"], color="#4C72B0")
    for i, v in enumerate(g["recovery"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(g.index, rotation=60, ha="right")
    ax.set_ylabel("recovery rate (marks restored / at stake)")
    ax.set_title("Recovery by mechanism (gated detector, synthetic)")
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_fpr_by_policy(dfs, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    pols, fprs, uppers = [], [], []
    for pol in POLICY_ORDER:
        d = dfs[(dfs["policy"] == pol) & (dfs["n_clean"] > 0)]
        if d.empty:
            continue
        fp = d["fp"].sum(); nc = d["n_clean"].sum()
        pols.append(pol)
        fprs.append(fp / nc)
        uppers.append(_cp_upper_safe(fp, nc) if nc else 0.0)
    ax.bar(range(len(pols)), fprs, color="#55A868")
    for i, (v, u) in enumerate(zip(fprs, uppers)):
        ax.errorbar(i, v, yerr=[[max(0.0, v - 0.0)], [max(0.0, u - v)]],
                    fmt="none", ecolor="black", capsize=3)
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols, rotation=30, ha="right")
    ax.set_ylabel("false-positive rate (with CP 95% upper bound)")
    ax.set_title("FPR by policy (disjoint sweep corpus)")
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_exam_length_effect(dfm, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for length in sorted(dfm["length"].unique()):
        d = dfm[(dfm["length"] == length) & (dfm["n_target"] > 0)]
        if d.empty:
            continue
        g = d.groupby("ability")["power"].mean()
        ax.plot([f"{a:.2f}" for a in g.index], g, "o-", label=f"length {length}")
    ax.set_xlabel("candidate ability")
    ax.set_ylabel("detection power")
    ax.set_title("Exam length effect on power (gated detector, synthetic)")
    ax.legend()
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_scanner_noise_effect(dfm, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = [m for m in dfm["mechanism"].unique() if m.startswith("scanner_")]
    drawn = False
    for m in sorted(sc):
        d = dfm[(dfm["mechanism"] == m) & (dfm["n_target"] > 0)]
        if d.empty:
            continue
        g = d.groupby("length")["power"].mean()
        ax.plot([str(l) for l in g.index], g, "o-", label=m)
        drawn = True
    ax.set_xlabel("exam length")
    ax.set_ylabel("detection power")
    ax.set_title("Scanner noise effect on power (gated detector, synthetic)")
    if drawn:
        ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_failure_heatmap(dfm, path):
    d = dfm[(dfm["n_target"] > 0) & (~dfm["mechanism"].isin(ADVERSARIAL))]
    pivot = d.pivot_table(index="mechanism", columns="ability", values="power",
                          aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, max(3.4, 0.3 * len(pivot) + 2)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(
        [f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Failure heatmap: power by mechanism x ability")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


FIGURES = {
    "safety_recovery_curve": fig_safety_recovery,
    "power_by_ability": fig_power_by_ability,
    "recovery_by_mechanism": fig_recovery_by_mechanism,
    "fpr_by_policy": fig_fpr_by_policy,
    "exam_length_effect": fig_exam_length_effect,
    "scanner_noise_effect": fig_scanner_noise_effect,
    "failure_heatmap": fig_failure_heatmap,
}


# ==============================================================================
# 9. EXPORT
# ==============================================================================


def write_csv(path, rows: List[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_json(path, obj, **params):
    provenance.write_json(path, obj, **params)


def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.4f}"


def build_summary(agg_metrics, agg_sweep, meta) -> Tuple[str, dict]:
    # overall per detector
    overall = {}
    for r in agg_metrics:
        key = r["detector"]
        o = overall.setdefault(key, {"n": 0, "n_target": 0, "n_clean": 0,
                                     "tp_correct": 0, "tp_wrongloc": 0, "fp": 0,
                                     "fn": 0, "tn": 0, "brier_sum": 0.0,
                                     "mse_sum": 0.0, "n_scored": 0,
                                     "at_stake": 0, "saved": 0,
                                     "unearned": 0})
        o["n"] += r["n"]; o["n_target"] += r["n_target"]
        o["n_clean"] += r["n_clean"]
        o["tp_correct"] += r["tp_correct"]; o["tp_wrongloc"] += r["tp_wrongloc"]
        o["fp"] += r["fp"]; o["fn"] += r["fn"]; o["tn"] += r["tn"]
        # Weighted by the sheets that actually carry a posterior, so a cell
        # with none contributes nothing rather than poisoning the pool.
        ns = r.get("n_scored", 0)
        if ns:
            o["brier_sum"] += r["brier"] * ns
            o["mse_sum"] += r["mse"] * ns
            o["n_scored"] += ns
        o["at_stake"] += r["at_stake"]; o["saved"] += r["saved"]
        o["unearned"] += r["unearned"]
    def _summ(o):
        power = o["tp_correct"] / o["n_target"] if o["n_target"] else float("nan")
        fpr = o["fp"] / o["n_clean"] if o["n_clean"] else float("nan")
        fpr_ci = _cp_upper_safe(o["fp"], o["n_clean"]) if o["n_clean"] else float("nan")
        rec = o["saved"] / o["at_stake"] if o["at_stake"] else float("nan")
        return {"n": o["n"], "error_sheets": o["n_target"],
                "clean_sheets": o["n_clean"],
                "tp_correct": o["tp_correct"], "tp_wrongloc": o["tp_wrongloc"],
                "fp": o["fp"], "fn": o["fn"], "tn": o["tn"],
                "power": round(power, 4), "fpr": round(fpr, 4),
                "fpr_ci_upper": round(fpr_ci, 4),
                "brier": (round(o["brier_sum"] / o["n_scored"], 4)
                          if o["n_scored"] else float("nan")),
                "mse": (round(o["mse_sum"] / o["n_scored"], 4)
                        if o["n_scored"] else float("nan")),
                "n_scored": o["n_scored"],
                "recovery": round(rec, 4), "saved": o["saved"],
                "at_stake": o["at_stake"], "unearned": o["unearned"]}
    overall = {k: _summ(v) for k, v in overall.items()}

    # sweep per-policy summary
    sweep_pol = {}
    for pol in POLICY_ORDER:
        rs = [r for r in agg_sweep if r["policy"] == pol]
        if not rs:
            continue
        fp = sum(r["fp"] for r in rs); nc = sum(r["n_clean"] for r in rs)
        tp = sum(r["tp_correct"] for r in rs); nt = sum(r["n_target"] for r in rs)
        saved = sum(r["saved"] for r in rs); stake = sum(r["at_stake"] for r in rs)
        sweep_pol[pol] = {
            "fpr": round(fp / nc, 4) if nc else None,
            "fpr_ci_upper": round(_cp_upper_safe(fp, nc), 4) if nc else None,
            "power": round(tp / nt, 4) if nt else None,
            "extra_per_error_sheet": round(saved / nt, 3) if nt else None,
            "recovery": round(saved / stake, 4) if stake else None,
            "fp": fp, "tp": tp,
        }

    summary = {"meta": meta, "overall": overall, "sweep_policies": sweep_pol,
               "failures_collected": meta["failures_collected"]}

    lines = []
    W = 64
    lines.append("#" * W)
    lines.append(" LARGE SYNTHETIC SHIFT-DETECTION BENCHMARK  --  summary")
    lines.append("#" * W)
    lines.append("")
    lines.append("Synthetic-only.  Every sheet has constructed ground truth.  No")
    lines.append("metric here is a deployment calibration of the detector on real")
    lines.append("paper (see README.md for why).  Brier / MSE are synthetic, in-")
    lines.append("corpus comparators, never absolute operating characteristics.")
    lines.append("")
    lines.append(f"mode        : {meta['mode']}")
    lines.append(f"master seed : {meta['seed']}")
    lines.append(f"sheets/cell : metrics={meta['n_metrics']}  sweep={meta['n_sweep']}")
    lines.append(f"permutations: reference MC nulls (metrics)")
    lines.append("")
    lines.append("-- OVERALL (metrics corpus, event counts shown alongside rates) --")
    for det, o in overall.items():
        lines.append(f"  {det:12s} n={o['n']:6d}  "
                     f"TP={o['tp_correct']:5d} TPwrongloc={o['tp_wrongloc']:4d} "
                     f"FP={o['fp']:4d} FN={o['fn']:4d} TN={o['tn']:5d}")
        lines.append(f"              power={_fmt(o['power'])}  "
                     f"fpr={_fmt(o['fpr'])}  (CP95 upper={_fmt(o['fpr_ci_upper'])})  "
                     f"recovery={_fmt(o['recovery'])}  unearned={o['unearned']}")
        lines.append(f"              brier={_fmt(o['brier'])}  mse={_fmt(o['mse'])}  "
                     f"saved={o['saved']}/{o['at_stake']} marks at stake")
    lines.append("")
    lines.append("-- THRESHOLD SWEEP (DISJOINT corpus; safety/recovery trade-offs) --")
    for pol in POLICY_ORDER:
        if pol not in sweep_pol:
            continue
        s = sweep_pol[pol]
        lines.append(f"  {pol:18s} fpr={_fmt(s['fpr'])} (cp95 {_fmt(s['fpr_ci_upper'])})  "
                     f"power={_fmt(s['power'])}  extra/err-sheet={_fmt(s['extra_per_error_sheet'])}  "
                     f"recovery={_fmt(s['recovery'])}")
    lines.append("")
    lines.append("The sweep is presented as policy trade-offs, NOT as tuning then")
    lines.append("reporting performance.  Default = out-of-the-box AdjudicationConfig.")
    lines.append("")
    lines.append("CP 95% bounds: exact Clopper-Pearson per cell (n<=200); Wilson")
    lines.append("approximation for the pooled overall bound (large n).")
    lines.append("")
    lines.append(f"Honest failure cases recorded in failure_cases.json : {meta['failures_collected']}")
    return "\n".join(lines), summary


# ==============================================================================
# 10. MAIN
# ==============================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke run (~400 sheets)")
    ap.add_argument("--full", action="store_true", help="research-scale run")
    ap.add_argument("--n", type=int, default=None, help="sheets per cell")
    ap.add_argument("--perm", type=int, default=None,
                    help="MC permutations for the reference detector (default 1100)")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers (default = CPU count)")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--level", type=float, default=None,
                    help="override the acceptance level for the metrics arm only, "
                         "leaving every other threshold shipped; use to reproduce a "
                         "before-and-after on the default")
    args = ap.parse_args(argv)

    mode = "quick" if args.quick else ("full" if args.full else "full")
    # 8 per cell. The metrics grid is 1,248 cells, so 8 already yields 9,984
    # sheets of which 3,840 are error-free -- and the error-free count is what
    # bounds the false positive claim (0 of 3,840 is 0.07% at 95%). Raising it
    # buys a tighter bound at proportional cost; lowering it below 8 leaves some
    # (length x ability x mechanism) cells too thin to report separately, which
    # is how this benchmark reports.
    n_metrics = args.n or (2 if mode == "quick" else 8)
    n_sweep = args.n or (2 if mode == "quick" else 8)
    perm = args.perm or 1100

    outdir = args.out or OUTDIR
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    t0 = time.time()
    rng = random.Random(args.seed)

    print(f"[bench] mode={mode}  n_metrics={n_metrics}  n_sweep={n_sweep}  "
          f"perm={perm}  seed={args.seed}")
    print(f"[bench] output -> {outdir}")

    # ---- metrics corpus -----------------------------------------------------
    cells = build_metrics_conditions(OPTION_SETS, quick=(mode == "quick"))
    print(f"[bench] metrics grid: {len(cells)} cells")
    nproc = max(1, args.jobs or (os.cpu_count() or 4))
    chunk_k = max(1, len(cells) // (nproc * 2))
    all_rows, failures = [], []
    fast = mode == "quick"
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for det_name in ("gated", "bruteforce", "lcs", "fixedcost", "noop"):
            print(f"[bench] running detector: {det_name} ...")
            tasks = [(chunk, det_name, perm, fast, None, "metrics",
                      args.seed, n_metrics, args.level)
                     for chunk in _chunked(cells, chunk_k)]
            n_rows = 0
            for rows, fails in ex.map(_run_batch, tasks):
                all_rows.extend(rows)
                failures.extend(fails)
                n_rows += len(rows)
            print(f"[bench]   {det_name}: {n_rows} sheets")

    failures = failures[:60]  # keep the cap exact across workers

    # ---- threshold sweep (disjoint corpus) ---------------------------------
    sweep_cells = build_sweep_conditions(OPTION_SETS, quick=(mode == "quick"))
    print(f"[bench] sweep grid: {len(sweep_cells)} cells x {len(POLICY_ORDER)} policies")
    sweep_rows = []
    schunk_k = max(1, len(sweep_cells) // nproc)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for pol in POLICY_ORDER:
            print(f"[bench]   policy {pol} ...")
            tasks = [(chunk, "gated", POLICY_PERM[pol], False, pol, "sweep",
                      args.seed + 777, n_sweep, None)
                     for chunk in _chunked(sweep_cells, schunk_k)]
            n_rows = 0
            for rows, _f in ex.map(_run_batch, tasks):
                sweep_rows.extend(rows)
                n_rows += len(rows)
            print(f"[bench]     {pol}: {n_rows} sheets")

    agg_metrics = aggregate(all_rows)
    agg_sweep = aggregate(sweep_rows)

    # ---- figures ------------------------------------------------------------
    dfm = pd.DataFrame(agg_metrics)
    dfs = pd.DataFrame(agg_sweep)
    for name, fn in FIGURES.items():
        try:
            fn(dfm if name in ("power_by_ability", "recovery_by_mechanism",
                               "exam_length_effect", "scanner_noise_effect",
                               "failure_heatmap") else dfs,
               os.path.join(figdir, name + ".png"))
            print(f"[bench] figure: {name}.png")
        except Exception as e:  # figure bugs must not kill the run
            print(f"[bench] figure {name} failed: {e}")

    # ---- exports ------------------------------------------------------------
    write_csv(os.path.join(outdir, "per_condition.csv"), agg_metrics + agg_sweep)
    write_csv(os.path.join(outdir, "per_sheet.csv"), all_rows + sweep_rows)
    write_csv(os.path.join(outdir, "threshold_sweep.csv"),
              [r for r in agg_sweep if r["policy"] != "metrics"])
    write_json(os.path.join(outdir, "failure_cases.json"), failures)

    meta = {"mode": mode, "seed": args.seed, "n_metrics": n_metrics,
            "n_sweep": n_sweep, "perm": perm,
            "failures_collected": len(failures)}
    text, summary = build_summary(agg_metrics, agg_sweep, meta)

    # No run duration and no wall-clock stamp in either file. Both are real
    # facts about the run and neither is a result, and carrying them would put
    # a diff in every regeneration -- which is exactly the signal the
    # provenance block exists to keep meaningful.
    stamp_params = dict(mode=mode, seed=meta["seed"], perm=meta["perm"])
    provenance.write_text(os.path.join(outdir, "summary.txt"), text,
                          **stamp_params)
    write_json(os.path.join(outdir, "summary.json"), summary, **stamp_params)

    # ---- README ------------------------------------------------------------
    readme = build_readme(meta, summary, mode)
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme)

    print()
    print(text)
    print(f"\n[bench] done in {time.time() - t0:.1f}s  -> {outdir}")


def build_readme(meta, summary, mode):
    return f"""# Large Synthetic Shift-Detection Benchmark  (results)

**Run**: mode `{mode}`  ·  seed {meta['seed']}

## Synthetic-only, NOT a deployment calibration

This benchmark is **entirely synthetic**.  Every sheet has known ground truth
constructed by the generator.  No metric in this folder is an operational or
deployment calibration of the shift detector on real examination paper.

The reference detector is a Bayesian pair-HMM whose confidence is a posterior
`P(shift | marks)` that places a **base-rate prior** over the shift hypothesis.
A posterior thresholded against a base rate cannot be validated on a test set
whose shift base rate differs from deployment (the base rate determines the
posterior through the prior term).  The Brier score and mean-squared-posterior
reported here are therefore **within-corpus comparators between detectors on
identical synthetic data only** — they rank detectors; they do not give
absolute false-positive rates for real paper.

## What is measured

* **Detection power** = TP-correctly-localized / error sheets (TP with a wrong
  location is reported separately, never merged in).
* **False-positive rate** = FP / clean sheets, with the **Clopper-Pearson
  95% upper bound** even when zero FPs are observed.  Cells (n <= 200 clean
  sheets) use the exact bound; the pooled overall bound uses the Wilson
  approximation (large n), since the exact binomial CDF overflows there.
* **Recovery** = marks genuinely restored by realignment / marks at stake.
* **Unearned marks** on clean sheets that were wrongly flagged.
* All of the above are reported as **event counts alongside rates**.

## Threshold sweep

`threshold_sweep.csv` and the policy figures are built on a **disjoint corpus**
from the metrics corpus.  Sweeps are policy trade-offs for an examination board
to choose between (very-conservative -> permissive), **not** tuning performed
on the metrics corpus and reported as validation.

## Reproduce

    python3 benchmark/large_synthetic.py --quick
    python3 benchmark/large_synthetic.py --full

Determinism: every sheet is drawn from `random.Random` with seed
`{meta['seed']} + 100000*(cell+1) + i`; seeds never rely on `hash()`.

## Files

* `summary.txt` / `summary.json`  — headline numbers
* `per_condition.csv`             — one row per (condition x detector x policy)
* `per_sheet.csv`                 — every synthetic sheet judged
* `threshold_sweep.csv`           — policy sweep rows
* `failure_cases.json`            — honest FN / FP / wrong-location cases
* `figures/*.png`                 — 7 figures
"""


if __name__ == "__main__":
    main()
