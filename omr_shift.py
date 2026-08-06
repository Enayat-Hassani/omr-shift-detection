#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 OMR ALIGNMENT ADJUDICATOR
 Statistical detection of mechanical answer-sheet shift errors
================================================================================

PROBLEM
-------
A candidate's marks on an optical-mark-recognition (OMR) sheet may be displaced
relative to the question numbers because the candidate skipped a bubble row, or
skipped a question on the paper without skipping its row. The observed sequence
of marks is then *correct in content but wrong in registration*.

We must decide, for a given sheet, which of two explanations is better
supported by the evidence:

    H0 : the marks are registered correctly; wrong answers are genuinely wrong.
    H1 : the marks are registered incorrectly from one or more points onward.

HARD CONSTRAINTS (fairness axioms, enforced structurally, not by convention)
---------------------------------------------------------------------------
 A1  Content immutability   : the sequence of marked options is never altered.
 A2  Order preservation     : the question -> row map is STRICTLY increasing.
 A3  Injectivity            : no row is read twice; no question is filled twice.
 A4  No invention           : an unmatched question scores ZERO, never credit.
 A5  Monotone credit        : leaving a question unmatched can never gain marks.
 A6  Bounded displacement   : |offset| <= D (physical plausibility of the slip).
 A7  Parsimony              : every shift event pays a prior-derived price.
 A8  Pre-registered gates   : re-registration is accepted only if a Bayes factor,
                              a Monte-Carlo p-value and per-segment binomial
                              tests all clear thresholds fixed BEFORE seeing data.

METHOD (see REPORT.md for the full comparison and ranking of alternatives)
-------------------------------------------------------------------------
A banded three-state PAIR HIDDEN MARKOV MODEL over (question, row) lattice
positions, decoded exactly by:
   * Viterbi          -> the MAP re-registration (the "story")
   * Forward          -> the marginal likelihood  P(marks | H1)  (the "evidence")
   * Forward-Backward -> per-question posterior over displacement (the "confidence")
plus a Monte-Carlo null calibration and a segment-level binomial coherence test.

This is mathematically identical to affine-gap global sequence alignment
(Needleman-Wunsch/Gotoh) and to a shortest-path problem on a DAG, but the
scores are log-likelihood ratios. That is what lets the output be expressed as a
probability and defended to an examination board.

No optimisation library is used. All dynamic programming, log-space arithmetic,
Bayesian marginalisation and hypothesis testing is implemented from scratch.

Author: designed as a research-grade reference implementation.
License: use freely for examination-board work.
================================================================================
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NEG_INF = float("-inf")

# ==============================================================================
# 0. NUMERICAL UTILITIES  (log-space arithmetic, implemented from scratch)
# ==============================================================================


def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable log(sum(exp(v))) for an iterable of log-values."""
    vals = [v for v in values if v > NEG_INF]
    if not vals:
        return NEG_INF
    m = max(vals)
    if m == NEG_INF:
        return NEG_INF
    total = 0.0
    for v in vals:
        total += math.exp(v - m)
    return m + math.log(total)


def log_diff_exp(a: float, b: float) -> float:
    """log(exp(a) - exp(b)) for a >= b. Returns -inf if a <= b."""
    if b == NEG_INF:
        return a
    if a <= b:
        return NEG_INF
    return a + math.log1p(-math.exp(b - a))


def log_beta_pdf(x: float, a: float, b: float) -> float:
    """Unnormalised-in-constant Beta log density (constant cancels in weights)."""
    if not (0.0 < x < 1.0):
        return NEG_INF
    return (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Exact one-sided upper confidence bound on a binomial rate.

    Observing zero false alarms in n trials does NOT mean the false-alarm rate
    is zero. With n = 180 and k = 0 the rate could still be as high as 1.6%.
    Reporting 0.000 without this bound would be the single most misleading
    number an examination board could be handed, so it is computed by bisection
    on the exact binomial tail.

    The tail is accumulated in log space. Computing it directly as
    `math.comb(n, i) * p**i * (1-p)**(n-i)` raises OverflowError once n is large
    and k > 0: the binomial coefficient is an exact Python int of arbitrary
    size, and multiplying it by a float overflows before the small powers bring
    it back into range. 519 events in 3840 trials is enough. Nothing in this
    repository reaches that scale, which is why the earlier form survived, but a
    bound that fails on a large corpus is not a bound.
    """
    if k >= n:
        return 1.0
    log_binom = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        for i in range(k + 1)
    ]
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(X <= k | p = mid); the bound is where this equals alpha
        if mid <= 0.0:
            cdf = 1.0
        elif mid >= 1.0:
            cdf = 0.0
        else:
            log_p, log_q = math.log(mid), math.log1p(-mid)
            cdf = math.exp(logsumexp(
                lb + i * log_p + (n - i) * log_q
                for i, lb in enumerate(log_binom)
            ))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, via math.comb. One-sided."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(1.0, max(0.0, total))


# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================


@dataclass(frozen=True)
class ResponseSheet:
    """
    One candidate's paper.

    key   : the answer key, indexed by QUESTION NUMBER (1..N).
    marks : the marked options, indexed by PHYSICAL ROW on the sheet (1..M).

    The distinction is the whole point: `marks` is a physical artefact, `key` is
    a logical one, and a shift error is a mis-registration between the two.
    A blank physical row is represented by None.
    """

    key: Tuple[str, ...]
    marks: Tuple[Optional[str], ...]
    options: Tuple[str, ...] = ("A", "B", "C", "D")
    candidate_id: str = "candidate"
    subject: str = "subject"

    def __post_init__(self) -> None:
        bad = [k for k in self.key if k not in self.options]
        if bad:
            raise ValueError(f"answer key contains options outside {self.options}: {bad}")
        bad = [m for m in self.marks if m is not None and m not in self.options]
        if bad:
            raise ValueError(f"marks contain options outside {self.options}: {bad}")

    @property
    def n_questions(self) -> int:
        return len(self.key)

    @property
    def n_rows(self) -> int:
        return len(self.marks)

    @classmethod
    def from_records(
        cls,
        records: Sequence[Dict],
        options: Sequence[str] = ("A", "B", "C", "D"),
        candidate_id: str = "candidate",
        subject: str = "subject",
    ) -> "ResponseSheet":
        """Build from [{'question':1,'correct':'B','student':'A'}, ...]."""
        rows = sorted(records, key=lambda r: r["question"])
        qs = [r["question"] for r in rows]
        if qs != list(range(1, len(rows) + 1)):
            raise ValueError("question numbers must be contiguous starting at 1")
        return cls(
            key=tuple(r["correct"] for r in rows),
            marks=tuple(r["student"] if r.get("student") else None for r in rows),
            options=tuple(options),
            candidate_id=candidate_id,
            subject=subject,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        options: Sequence[str] = ("A", "B", "C", "D"),
        candidate_id: str = "candidate",
        subject: str = "subject",
    ) -> "ResponseSheet":
        """Build from a CSV or JSON file of one sheet.

        Either format supplies one record per question. Recognised column names
        are `question`, `correct` or `correct_answer`, and `student` or
        `student_answer`; a blank or missing mark becomes None. JSON may be a
        bare list of records or an object with a single list value.
        """
        if path.lower().endswith(".json"):
            with open(path) as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                lists = [v for v in raw.values() if isinstance(v, list)]
                if len(lists) != 1:
                    raise ValueError(f"{path}: expected one list of records")
                raw = lists[0]
        else:
            import csv
            with open(path) as fh:
                raw = list(csv.DictReader(fh))

        def pick(rec, *names):
            for n in names:
                if rec.get(n) not in (None, ""):
                    return rec[n]
            return None

        records = [{"question": int(pick(r, "question")),
                    "correct": pick(r, "correct", "correct_answer"),
                    "student": pick(r, "student", "student_answer")}
                   for r in raw]
        return cls.from_records(records, options, candidate_id, subject)

    def raw_score(self) -> int:
        """Score under the identity registration (row i answers question i)."""
        n = min(self.n_questions, self.n_rows)
        return sum(1 for i in range(n) if self.marks[i] is not None and self.marks[i] == self.key[i])


class Policy(Enum):
    """
    The three acceptance profiles an examination board may adopt.

    A profile is a NAMED OPERATING POINT. An examination board selects it and
    records it. It fixes the acceptance level and the number of draws used to
    test it, and nothing else.

    `permutation_alpha` is NOT a false positive rate and must not be labelled as
    one. The reported p-value is the worst of three nulls, and it is a
    conservative bound by an unmeasured margin. See ASSUMPTIONS.md A6 for the
    measurement that established this.

    What each profile recovers, and the smallest displaced block it can accept,
    are empirical properties of the detector, measured against a particular
    population of sheets. They are not attributes of the policy, and they are
    not carried here: a figure in the code cannot state the conditions it was
    measured under, and cannot go stale visibly. REPORT.md section 6.1 holds
    them, with the population attached. An examination board selecting a
    profile should read that table. No profile makes a short displacement
    detectable.
    """

    CONSERVATIVE = ("conservative", 0.001)
    BALANCED = ("balanced", 0.010)
    SENSITIVE = ("sensitive", 0.050)

    def __init__(self, label: str, alpha: float) -> None:
        self.label = label
        self.alpha = alpha

    @property
    def n_permutations(self) -> int:
        """
        Derived, never set. The permutation count is not independent of the
        level it tests.

        The gate admits at most `floor(alpha * (n + 1) - 1)` exceedances. So the
        smallest p-value the test can report is `1 / (n + 1)`. Where that
        exceeds alpha, the gate cannot be satisfied. No sheet can pass, the
        detector silently accepts nothing, and no error is raised. At
        alpha = 1e-4 against 4000 draws, the strongest sheet in the validation
        set is rejected. This was first met during the item response theory work
        and patched at that
        one call site. Deriving the count here is what stops it happening
        again.

        `ceil(10 / alpha) - 1` leaves at least nine admissible exceedances, so
        the level is both representable and resolved.
        """
        return math.ceil(10.0 / self.alpha) - 1


@dataclass(frozen=True)
class AdjudicationConfig:
    """
    Every number here is a PRIOR or a PRE-REGISTERED THRESHOLD, not a tuning
    knob. Each carries an operational meaning that an examination board can
    audit, challenge, and re-estimate from its own historical corpus.
    """

    # ---- physical / structural -------------------------------------------
    max_displacement: int = 3
    """A6. Maximum |offset| entertained. Slips of more than 3 rows are
    implausible without the candidate noticing; widening this widens the
    attack surface linearly."""

    # ---- prior over mechanical slips (empirically estimable) --------------
    sheet_slip_rate: float = 0.018
    """P(a randomly drawn sheet contains at least one row-skip).

    Default is the EMPIRICAL rate measured by Skiena & Sumazin (2004), "Shift
    error detection in standardized exams", J. Discrete Algorithms 2:313-331:
    1.8% of 101,265 Scholastic Amplitude Tests contained shift errors, with
    ~2% corroborated on Stony Brook undergraduate exams. Using a published,
    externally measured base rate is what makes the parsimony penalty (A7)
    auditable. A board with its own
    re-mark history should substitute its own figure."""

    row_skip_extend: float = 0.15
    """P(a slip, once started, extends by one more row)."""

    blank_rate: float = 0.02
    """P(a candidate leaves any given question unanswered on the paper)."""

    blank_extend: float = 0.30
    """P(a run of unanswered questions continues)."""

    # ---- ability prior (marginalised out; NOT fitted to the shifted score) -
    theta_prior_a: float = 6.0
    theta_prior_b: float = 4.0
    """Beta prior on the candidate's per-item probability of being correct on
    this paper. Mean = a/(a+b). Set from prior attainment in OTHER subjects, or
    from the cohort. NEVER from this paper's re-registered score (circularity)."""

    external_ability: Optional[float] = None
    """THE psychometric input. The candidate's competence estimated from
    evidence INDEPENDENT of this paper -- their other subjects, their prior
    attainment record. If supplied, it is used for decoding and calibration.
    If not, the prior mean is used. It is never estimated from this paper's
    re-registered score, because that would be circular: a high fitted ability
    is exactly what a successful re-registration manufactures."""

    external_concentration: float = 60.0
    """How firmly the external ability estimate is held (Beta pseudo-counts).
    60 means 'as informative as 60 previously observed items'. Raising it lets
    a sympathetic board assert competence more forcefully -- which is precisely
    why the Monte-Carlo gate, which uses no ability model at all, is the one
    that must ultimately hold the line."""

    theta_floor_at_chance: bool = True
    """A candidate cannot be worse than a random guesser in any meaningful
    sense. Below theta = 1/C the log-likelihood ratio INVERTS -- a matching
    answer becomes evidence AGAINST alignment -- and the detector starts
    preferring wrong answers. The floor is a structural necessity, not a
    convenience."""

    theta_ceiling: float = 0.95
    """Paired with `blank_safety` to keep axiom A5 satisfiable."""

    blank_safety: float = 0.5
    """THE FAIRNESS CLAMP. The effective cost of leaving a question unmatched is
    forced to be at most `blank_safety` times the probability of a wrong match,
    at every theta. This makes 'unmatch the ones I got wrong to buy myself some
    realignment freedom' strictly unprofitable BY CONSTRUCTION, at every parameter
    setting."""

    theta_grid_size: int = 25
    """Number of grid points used to marginalise over ability.

    The integral is approximated by a 25-point midpoint rule, not evaluated in
    closed form. Bayes factors and posteriors inherit that discretisation error.
    Raising this costs time linearly and tightens the approximation."""

    # ---- pre-registered acceptance gates (A8) -----------------------------
    bayes_factor_threshold: float = 100.0
    """Jeffreys' 'decisive' band. BF_10 must exceed this."""

    posterior_shift_threshold: float = 0.95
    """Posterior P(H1 | marks) required, using sheet_slip_rate as prior odds."""

    item_posterior_threshold: float = 0.99
    """Per-question posterior on the MAP displacement required before that
    single question is re-registered. Credit is granted item by item."""

    permutation_alpha: float = 0.010
    """Acceptance level for the Monte-Carlo null calibration.

    Defaults to the Balanced profile. It was 0.001 until the profiles were
    measured: across 300 error-free sheets the tighter level produced no fewer
    false positives, and across 150 genuine skips it returned a third fewer
    marks. Paying recovery for safety that does not show up in the measurement
    is not a trade I am willing to leave as the default. Use
    `AdjudicationConfig.from_policy` to select a different profile, and see
    REPORT.md section 6.1 before doing so."""

    n_permutations: int = 999

    min_segment_length: int = 5
    """A displaced segment shorter than this is noise, not a slip."""

    segment_binom_alpha: float = 0.01
    """Each displaced segment must be significantly above chance on its own."""

    seed: int = 20260804

    def __post_init__(self) -> None:
        smallest_reportable = 1.0 / (self.n_permutations + 1)
        if smallest_reportable > self.permutation_alpha:
            raise ValueError(
                f"permutation_alpha={self.permutation_alpha} is unreachable with "
                f"n_permutations={self.n_permutations}: the smallest p-value the "
                f"test can report is {smallest_reportable:.2e}, so the Monte-Carlo "
                f"gate could never pass and the detector would accept nothing. "
                f"Use AdjudicationConfig.from_policy(Policy.BALANCED), or raise "
                f"n_permutations to at least {math.ceil(1.0 / self.permutation_alpha) - 1}."
            )

    @classmethod
    def from_policy(cls, policy: Policy, **overrides) -> "AdjudicationConfig":
        """
        Build a configuration from a named profile.

        This is the supported way to set the acceptance level. It fixes
        `permutation_alpha` and derives `n_permutations` from it, so the two
        cannot be set into an impossible relationship. Everything else stays
        available: `sheet_slip_rate`, `external_ability` and `max_displacement`
        are board inputs with external sources and are not part of the profile.
        """
        return cls(permutation_alpha=policy.alpha,
                   n_permutations=policy.n_permutations, **overrides)

    def theta_grid(self, n_options: int = 4) -> List[Tuple[float, float]]:
        """(theta, log prior weight) pairs for grid-approximated marginalisation.

        The support is [1/C, theta_ceiling]. See `theta_floor_at_chance`.
        """
        lo = (1.0 / n_options) if self.theta_floor_at_chance else 1e-3
        hi = self.theta_ceiling
        k = self.theta_grid_size
        pts = [lo + (hi - lo) * (i + 0.5) / k for i in range(k)]
        a, b = self.theta_prior_a, self.theta_prior_b
        if self.external_ability is not None:
            # External evidence about the candidate must enter the EVIDENCE, not
            # only the decoding. Otherwise 'we believe they are able' is asserted
            # in the narrative but never actually tested against the marks.
            m = min(0.99, max(0.01, self.external_ability))
            a, b = m * self.external_concentration, (1 - m) * self.external_concentration
        logw = [log_beta_pdf(t, a, b) for t in pts]
        z = logsumexp(logw)
        return [(t, w - z) for t, w in zip(pts, logw)]

    def operating_theta(self, n_options: int = 4) -> float:
        """The ability used for point decoding and Monte-Carlo calibration.

        Drawn from EXTERNAL evidence when available, otherwise from the prior
        mean -- never from this paper. Clamped into the admissible band.
        """
        if self.external_ability is not None:
            t = self.external_ability
        else:
            t = self.theta_prior_a / (self.theta_prior_a + self.theta_prior_b)
        lo = (1.0 / n_options) if self.theta_floor_at_chance else 1e-3
        return min(self.theta_ceiling, max(lo, t))


# ==============================================================================
# 2. SCORING MODEL  --  turns the problem into log-likelihood ratios
# ==============================================================================


class ScoringModel:
    """
    Emission probabilities of the generative model.

    A matched pair (question q, row r) contributes
        log P(mark_r | key_q, theta)  =  log(theta)          if mark == key
                                         log((1-theta)/(C-1)) otherwise
    i.e. the classic 'knows it, or guesses uniformly among distractors' model.

    An ORPHAN ROW (a mark belonging to no question) contributes the marginal
    probability of that symbol under the candidate's own response distribution.
    This matters: it means the model cannot get a free lunch by discarding a
    mark, and it prices marks by how surprising they are.

    KEY DESIGN DECISION -- THE FAIRNESS CLAMP
    -----------------------------------------
    A wrong match costs log((1-theta)/(C-1)); an unmatched question costs
    log(blank_rate). If blank_rate ever exceeds (1-theta)/(C-1), then abandoning
    a question becomes CHEAPER than admitting a wrong answer, and the model can
    buy realignment freedom by discarding its own mistakes. That happens for
    theta above about 0.93 with realistic blank rates -- a real loophole, not a
    hypothetical one.

    The blank cost is therefore clamped:

        log_blank(theta) = min( log(blank_rate),
                                log((1-theta)/(C-1)) + log(blank_safety) )

    with blank_safety < 1. Axiom A5 therefore holds at every theta, for every
    configuration, by construction. `validate_no_free_unmatch` re-checks it.
    """

    def __init__(self, sheet: ResponseSheet, cfg: AdjudicationConfig) -> None:
        self.sheet = sheet
        self.cfg = cfg
        self.n_opts = len(sheet.options)
        self.symbol_logp = self._empirical_marginals()

    def _empirical_marginals(self) -> Dict[str, float]:
        """Log marginal distribution of the candidate's own marks, Laplace-smoothed.

        Using the candidate's OWN distribution neutralises
        response bias -- a candidate who guesses 'C' whenever unsure gets no
        credit for accidentally lining up with 'C'-heavy stretches of the key.
        """
        counts = {o: 1.0 for o in self.sheet.options}
        for m in self.sheet.marks:
            if m is not None:
                counts[m] += 1.0
        total = sum(counts.values())
        return {o: math.log(c / total) for o, c in counts.items()}

    def validate_no_free_unmatch(self) -> List[str]:
        """Verify axiom A5 across the whole ability grid, AFTER clamping."""
        issues = []
        for t, _ in self.cfg.theta_grid(self.n_opts):
            wrong = math.log((1.0 - t) / (self.n_opts - 1))
            if self.log_blank_open(t) >= wrong:
                issues.append(
                    f"axiom A5 violated at theta={t:.3f}: unmatching "
                    f"({self.log_blank_open(t):.3f}) is not costlier than a wrong "
                    f"match ({wrong:.3f})."
                )
        for t, _ in self.cfg.theta_grid(self.n_opts):
            if math.log(t) - math.log((1.0 - t) / (self.n_opts - 1)) <= 0:
                issues.append(
                    f"log-likelihood ratio is non-positive at theta={t:.3f}: the "
                    f"detector would treat a matching answer as evidence AGAINST "
                    f"alignment. Raise the ability floor."
                )
        return issues

    # -- emissions ---------------------------------------------------------

    def match_score(self, q_idx: int, r_idx: int, theta: float) -> float:
        """log P(row r's mark | question q's key). q_idx, r_idx are 0-based."""
        mark = self.sheet.marks[r_idx]
        if mark is None:
            # A physically blank row consumed by a question: no information,
            # priced as a blank response (clamped, see A5).
            return self.log_blank_open(theta)
        if mark == self.sheet.key[q_idx]:
            return math.log(theta)
        return math.log((1.0 - theta) / (self.n_opts - 1))

    def orphan_row_score(self, r_idx: int) -> float:
        """log P(this mark | it belongs to no question)."""
        mark = self.sheet.marks[r_idx]
        return 0.0 if mark is None else self.symbol_logp[mark]

    # -- transition (gap) log-priors ---------------------------------------

    # These five are constants of the model: they depend on the configuration,
    # the sheet length and the ability, none of which change during a decode.
    # Evaluated per lattice cell they accounted for roughly 70,000 calls and a
    # third of the logarithms in a single adjudication, so they are memoised.
    # The theta-keyed pair is cached on the value, since ability is fixed within
    # a decode but varies across the marginalisation grid.

    @cached_property
    def log_row_skip_open(self) -> float:
        # per-position hazard of starting a slip
        per_pos = self.cfg.sheet_slip_rate / max(1, self.sheet.n_questions)
        return math.log(per_pos)

    @cached_property
    def log_row_skip_extend(self) -> float:
        return math.log(self.cfg.row_skip_extend)

    def _blank_ceiling(self, theta: float) -> float:
        return (math.log((1.0 - theta) / (self.n_opts - 1))
                + math.log(self.cfg.blank_safety))

    def log_blank_open(self, theta: float) -> float:
        """Cost of leaving a question unmatched, under THE FAIRNESS CLAMP."""
        cache = self.__dict__.setdefault("_blank_open_cache", {})
        v = cache.get(theta)
        if v is None:
            v = cache[theta] = min(math.log(self.cfg.blank_rate),
                                   self._blank_ceiling(theta))
        return v

    def log_blank_extend(self, theta: float) -> float:
        cache = self.__dict__.setdefault("_blank_extend_cache", {})
        v = cache.get(theta)
        if v is None:
            prior = math.log(self.cfg.blank_rate) + math.log(self.cfg.blank_extend)
            v = cache[theta] = min(prior, self._blank_ceiling(theta))
        return v

    @cached_property
    def log_continue(self) -> float:
        """log P(no gap event at this position) -- keeps the model a proper
        probability distribution."""
        per_pos = self.cfg.sheet_slip_rate / max(1, self.sheet.n_questions)
        return math.log(max(1e-12, 1.0 - per_pos - self.cfg.blank_rate))

    def break_even_questions(self, theta: float) -> float:
        """
        The single most interpretable diagnostic in the whole system:
        HOW MANY newly-correct answers a shift must produce before the evidence
        outweighs the prior improbability of the slip.

            break_even = -log(P(slip)) / (log(theta) - log((1-theta)/(C-1)))
        """
        gain_per_item = math.log(theta) - math.log((1.0 - theta) / (self.n_opts - 1))
        if gain_per_item <= 1e-12:
            # At exactly chance ability a repaired answer carries no evidential
            # weight at all, so no finite number of repairs can pay for a slip.
            return float("inf")
        return -self.log_row_skip_open / gain_per_item


# ==============================================================================
# 3. THE BANDED PAIR-HMM  --  exact inference, implemented from scratch
# ==============================================================================

STATE_M, STATE_X, STATE_Y = 0, 1, 2
STATE_NAMES = {STATE_M: "MATCH", STATE_X: "QUESTION-BLANK", STATE_Y: "ORPHAN-ROW"}


@dataclass
class Alignment:
    """A concrete re-registration: which physical row answers which question."""

    pairs: Dict[int, int]           # question index (0-based) -> row index (0-based)
    blank_questions: List[int]      # questions matched to nothing
    orphan_rows: List[int]          # marks belonging to no question
    log_score: float
    path: List[Tuple[int, int, int]] = field(default_factory=list)  # (state,q,r)

    def offset(self, q_idx: int) -> Optional[int]:
        r = self.pairs.get(q_idx)
        return None if r is None else r - q_idx

    def is_identity(self, n_questions: int) -> bool:
        return all(self.pairs.get(q) == q for q in range(n_questions))


class BandedPairHMM:
    """
    Three-state pair HMM over the (question, row) lattice.

      MATCH (M) : question q is answered by row r      -> consumes both
      BLANK (X) : question q has no mark               -> consumes a question
      ORPHAN(Y) : row r answers no question            -> consumes a row

    A strictly increasing partial injection between questions and rows is
    exactly a path through this lattice, so axioms A2 and A3 hold BY
    CONSTRUCTION -- they cannot be violated by any parameter setting. That
    structural guarantee is the reason this formulation was chosen over an
    unconstrained assignment or a free-form search.

    The lattice is banded to |r - q - drift| <= D (axiom A6), giving O(N*D)
    work, down from O(N^2).
    """

    def __init__(self, model: ScoringModel, theta: float) -> None:
        self.model = model
        self.theta = theta
        self.N = model.sheet.n_questions
        self.M = model.sheet.n_rows
        self.D = model.cfg.max_displacement
        self.drift = self.M - self.N
        if abs(self.drift) > self.D:
            raise ValueError(
                f"sheet has {self.M} marks for {self.N} questions; the net "
                f"displacement {self.drift} exceeds max_displacement={self.D}."
            )
        self.lo = min(0, self.drift) - self.D
        self.hi = max(0, self.drift) + self.D

    # -- lattice helpers ---------------------------------------------------

    def _row_range(self, q: int) -> range:
        """Rows reachable at question-prefix q (both 0..N / 0..M inclusive)."""
        return range(max(0, q + self.lo), min(self.M, q + self.hi) + 1)

    def _in_band(self, q: int, r: int) -> bool:
        return 0 <= q <= self.N and 0 <= r <= self.M and self.lo <= r - q <= self.hi

    def _new_table(self) -> List[List[List[float]]]:
        return [[[NEG_INF] * (self.M + 1) for _ in range(self.N + 1)] for _ in range(3)]

    # -- transition costs --------------------------------------------------

    def _emit_match(self, q: int, r: int) -> float:
        """Emission for entering MATCH at lattice point (q, r), 1-based prefix."""
        return self.model.log_continue + self.model.match_score(q - 1, r - 1, self.theta)

    def _cost_x(self, from_state: int) -> float:
        return (
            self.model.log_blank_extend(self.theta)
            if from_state == STATE_X
            else self.model.log_blank_open(self.theta)
        )

    def _cost_y(self, from_state: int, r: int) -> float:
        base = (
            self.model.log_row_skip_extend
            if from_state == STATE_Y
            else self.model.log_row_skip_open
        )
        return base + self.model.orphan_row_score(r - 1)

    # -- VITERBI : the MAP re-registration ---------------------------------

    def viterbi(self) -> Alignment:
        V = self._new_table()
        back: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        V[STATE_M][0][0] = 0.0  # virtual start

        for q in range(0, self.N + 1):
            for r in self._row_range(q):
                if q == 0 and r == 0:
                    continue
                # --- MATCH: from (q-1, r-1)
                if q >= 1 and r >= 1 and self._in_band(q - 1, r - 1):
                    best, arg = NEG_INF, None
                    for s in (STATE_M, STATE_X, STATE_Y):
                        v = V[s][q - 1][r - 1]
                        if v > best:
                            best, arg = v, s
                    if best > NEG_INF:
                        V[STATE_M][q][r] = best + self._emit_match(q, r)
                        back[(STATE_M, q, r)] = (arg, q - 1, r - 1)
                # --- BLANK question: from (q-1, r)
                if q >= 1 and self._in_band(q - 1, r):
                    best, arg = NEG_INF, None
                    for s in (STATE_M, STATE_X, STATE_Y):
                        v = V[s][q - 1][r]
                        if v == NEG_INF:
                            continue
                        cand = v + self._cost_x(s)
                        if cand > best:
                            best, arg = cand, s
                    if best > NEG_INF:
                        V[STATE_X][q][r] = best
                        back[(STATE_X, q, r)] = (arg, q - 1, r)
                # --- ORPHAN row: from (q, r-1)
                if r >= 1 and self._in_band(q, r - 1):
                    best, arg = NEG_INF, None
                    for s in (STATE_M, STATE_X, STATE_Y):
                        v = V[s][q][r - 1]
                        if v == NEG_INF:
                            continue
                        cand = v + self._cost_y(s, r)
                        if cand > best:
                            best, arg = cand, s
                    if best > NEG_INF:
                        V[STATE_Y][q][r] = best
                        back[(STATE_Y, q, r)] = (arg, q, r - 1)

        end_state, end_val = None, NEG_INF
        for s in (STATE_M, STATE_X, STATE_Y):
            if V[s][self.N][self.M] > end_val:
                end_val, end_state = V[s][self.N][self.M], s
        if end_state is None:
            raise RuntimeError("no feasible alignment inside the displacement band")

        # traceback
        path: List[Tuple[int, int, int]] = []
        s, q, r = end_state, self.N, self.M
        while not (q == 0 and r == 0 and s == STATE_M):
            path.append((s, q, r))
            s, q, r = back[(s, q, r)]
        path.reverse()

        pairs, blanks, orphans = {}, [], []
        for st, qq, rr in path:
            if st == STATE_M:
                pairs[qq - 1] = rr - 1
            elif st == STATE_X:
                blanks.append(qq - 1)
            else:
                orphans.append(rr - 1)
        return Alignment(pairs, blanks, orphans, end_val, path)

    # -- FORWARD : the marginal likelihood over every admissible alignment

    def forward(self) -> Tuple[List[List[List[float]]], float]:
        F = self._new_table()
        F[STATE_M][0][0] = 0.0
        for q in range(0, self.N + 1):
            for r in self._row_range(q):
                if q == 0 and r == 0:
                    continue
                if q >= 1 and r >= 1 and self._in_band(q - 1, r - 1):
                    prev = logsumexp(V_[q - 1][r - 1] for V_ in F)
                    if prev > NEG_INF:
                        F[STATE_M][q][r] = prev + self._emit_match(q, r)
                if q >= 1 and self._in_band(q - 1, r):
                    F[STATE_X][q][r] = logsumexp(
                        F[s][q - 1][r] + self._cost_x(s) for s in (STATE_M, STATE_X, STATE_Y)
                    )
                if r >= 1 and self._in_band(q, r - 1):
                    F[STATE_Y][q][r] = logsumexp(
                        F[s][q][r - 1] + self._cost_y(s, r) for s in (STATE_M, STATE_X, STATE_Y)
                    )
        Z = logsumexp(F[s][self.N][self.M] for s in (STATE_M, STATE_X, STATE_Y))
        return F, Z

    # -- BACKWARD : per-question posteriors --------------------------------

    def backward(self) -> List[List[List[float]]]:
        B = self._new_table()
        for s in (STATE_M, STATE_X, STATE_Y):
            B[s][self.N][self.M] = 0.0
        for q in range(self.N, -1, -1):
            for r in reversed(list(self._row_range(q))):
                if q == self.N and r == self.M:
                    continue
                for s in (STATE_M, STATE_X, STATE_Y):
                    terms = []
                    if self._in_band(q + 1, r + 1) and q + 1 <= self.N and r + 1 <= self.M:
                        nxt = B[STATE_M][q + 1][r + 1]
                        if nxt > NEG_INF:
                            terms.append(self._emit_match(q + 1, r + 1) + nxt)
                    if self._in_band(q + 1, r) and q + 1 <= self.N:
                        nxt = B[STATE_X][q + 1][r]
                        if nxt > NEG_INF:
                            terms.append(self._cost_x(s) + nxt)
                    if self._in_band(q, r + 1) and r + 1 <= self.M:
                        nxt = B[STATE_Y][q][r + 1]
                        if nxt > NEG_INF:
                            terms.append(self._cost_y(s, r + 1) + nxt)
                    B[s][q][r] = logsumexp(terms)
        return B

    def posterior_offsets(self) -> Tuple[Dict[int, Dict[Optional[int], float]], float]:
        """
        P(question q is answered by the row at displacement d | all marks),
        marginalising over every admissible re-registration.

        Returns {q_idx: {offset or None(blank): probability}} and log Z.
        """
        F, Z = self.forward()
        B = self.backward()
        if abs(B[STATE_M][0][0] - Z) > 1e-6:
            raise AssertionError(
                f"forward/backward disagree: {Z:.9f} vs {B[STATE_M][0][0]:.9f}"
            )
        post: Dict[int, Dict[Optional[int], float]] = {q: {} for q in range(self.N)}
        for q in range(1, self.N + 1):
            for r in self._row_range(q):
                if r >= 1 and F[STATE_M][q][r] > NEG_INF:
                    p = math.exp(F[STATE_M][q][r] + B[STATE_M][q][r] - Z)
                    if p > 1e-12:
                        d = (r - 1) - (q - 1)
                        post[q - 1][d] = post[q - 1].get(d, 0.0) + p
                if F[STATE_X][q][r] > NEG_INF:
                    p = math.exp(F[STATE_X][q][r] + B[STATE_X][q][r] - Z)
                    if p > 1e-12:
                        post[q - 1][None] = post[q - 1].get(None, 0.0) + p
        return post, Z

    # -- the null hypothesis -----------------------------------------------

    def identity_log_likelihood(self) -> float:
        """log P(marks | H0): the sheet is registered correctly.

        Generalised to N != M: the overlap is matched one-to-one, surplus
        questions are unmatched and surplus rows are orphans, both priced by the
        same model so that H0 and H1 remain commensurable.
        """
        k = min(self.N, self.M)
        total = sum(self._emit_match(i + 1, i + 1) for i in range(k))
        for q in range(k, self.N):
            total += self._cost_x(STATE_X if q > k else STATE_M)
        for r in range(k, self.M):
            total += self._cost_y(STATE_Y if r > k else STATE_M, r + 1)
        return total


# ==============================================================================
# 4. ABILITY MARGINALISATION  --  removes the biggest tuning knob
# ==============================================================================


class EvidenceEngine:
    """
    Computes P(marks | H0) and P(marks | H1) with the candidate's ability theta
    integrated out under its prior. It is never fixed at a convenient value.

        P(marks | H) = SUM_theta  w(theta) * P(marks | H, theta)

    H1 is defined as 'some re-registration other than the identity', so the
    Bayes factor is a genuine comparison of rival explanations. The identity path appears in only one
    term, so it cannot inflate the comparison.
    """

    def __init__(self, model: ScoringModel) -> None:
        self.model = model
        self.cfg = model.cfg
        self.grid = self.cfg.theta_grid(model.n_opts)
        self.operating_theta = self.cfg.operating_theta(model.n_opts)

    def evaluate(self, include_profile: bool = True) -> Dict:
        """`include_profile=False` skips the ability-robustness sweep, which is
        REPORT-ONLY and feeds no gate. Worth ~29% of per-sheet cost in batch
        mode; always keep it on for an adjudication that will be published."""
        log_h0_terms, log_all_terms, per_theta = [], [], []
        for theta, logw in self.grid:
            hmm = BandedPairHMM(self.model, theta)
            _, Z = hmm.forward()
            L0 = hmm.identity_log_likelihood()
            log_h0_terms.append(logw + L0)
            log_all_terms.append(logw + Z)
            per_theta.append({"theta": theta, "log_w": logw, "logZ": Z, "logL0": L0})

        log_h0 = logsumexp(log_h0_terms)
        log_all = logsumexp(log_all_terms)
        # H1 excludes the identity registration entirely.
        log_h1 = log_diff_exp(log_all, log_h0)
        log_bf = log_h1 - log_h0

        prior_odds = self.cfg.sheet_slip_rate / (1.0 - self.cfg.sheet_slip_rate)
        log_post_odds = log_bf + math.log(prior_odds)
        post_h1 = 1.0 / (1.0 + math.exp(-log_post_odds)) if log_post_odds < 700 else 1.0

        # posterior over ability, useful for the report
        theta_post = []
        z = logsumexp(log_all_terms)
        for rec, la in zip(per_theta, log_all_terms):
            theta_post.append((rec["theta"], math.exp(la - z)))

        return {
            "log_p_h0": log_h0,
            "log_p_h1": log_h1,
            "log_bayes_factor": log_bf,
            "bayes_factor": math.exp(min(log_bf, 700.0)),
            "log10_bayes_factor": log_bf / math.log(10.0),
            "posterior_h1": post_h1,
            "per_theta": per_theta,
            "theta_posterior": theta_post,
            "operating_theta": self.operating_theta,
            "theta_source": (
                "external evidence (other subjects / prior attainment)"
                if self.cfg.external_ability is not None
                else "prior mean (no external ability supplied)"
            ),
            # Diagnostic only. NEVER used for decoding: fitting ability to this
            # paper and then using it to re-register this paper is circular.
            "implied_ability_from_this_paper": sum(t * p for t, p in theta_post),
            "theta_profile": self.theta_profile() if include_profile else [],
        }

    def theta_profile(self) -> List[Dict]:
        """
        THE MOST IMPORTANT ROBUSTNESS OUTPUT.

        The verdict depends on how competent we assume the candidate to be. A
        board should never have to take one assumed ability on trust. This
        sweeps the whole admissible range and reports, at each ability, the
        Bayes factor and what the best admissible re-registration would score.

        If the answer is 'no re-registration helps, at ANY assumed ability',
        the conclusion is immune to the ability assumption entirely -- which is
        the strongest form the argument can take.
        """
        out = []
        C = self.model.n_opts
        for i in range(15):
            lo = 1.0 / C + 1e-3   # strictly above chance: at chance the LLR is null
            th = lo + (self.cfg.theta_ceiling - lo) * i / 14.0
            hmm = BandedPairHMM(self.model, th)
            _, Z = hmm.forward()
            L0 = hmm.identity_log_likelihood()
            align = hmm.viterbi()
            best = sum(
                1
                for q, r in align.pairs.items()
                if self.model.sheet.marks[r] == self.model.sheet.key[q]
            )
            out.append(
                {
                    "theta": th,
                    "log10_bf": (log_diff_exp(Z, L0) - L0) / math.log(10.0),
                    "map_best_score": best,
                    "map_is_identity": align.is_identity(self.model.sheet.n_questions),
                    "break_even": self.model.break_even_questions(th),
                }
            )
        return out

    def posterior_mean_offsets(self) -> Dict[int, Dict[Optional[int], float]]:
        """Displacement posteriors, averaged over the ability posterior."""
        weights, posts = [], []
        for theta, logw in self.grid:
            hmm = BandedPairHMM(self.model, theta)
            p, Z = hmm.posterior_offsets()
            weights.append(logw + Z)
            posts.append(p)
        z = logsumexp(weights)
        w = [math.exp(x - z) for x in weights]

        out: Dict[int, Dict[Optional[int], float]] = {}
        for q in range(self.model.sheet.n_questions):
            acc: Dict[Optional[int], float] = {}
            for wi, p in zip(w, posts):
                for d, pv in p[q].items():
                    acc[d] = acc.get(d, 0.0) + wi * pv
            out[q] = acc
        return out

    def map_alignment(self) -> Tuple[Alignment, float]:
        """MAP re-registration at the operating (externally sourced) ability."""
        th = self.operating_theta
        return BandedPairHMM(self.model, th).viterbi(), th


# ==============================================================================
# 5. SEGMENT ANALYSIS  --  turns a path into an auditable narrative
# ==============================================================================


@dataclass
class Segment:
    offset: int
    q_start: int              # 0-based inclusive
    q_end: int                # 0-based inclusive
    n_items: int
    n_correct: int
    binom_p: float
    coherent: bool

    @property
    def label(self) -> str:
        return f"Q{self.q_start + 1}-Q{self.q_end + 1} @ offset {self.offset:+d}"


class SegmentAnalyzer:
    """
    Decomposes an alignment into maximal constant-displacement runs and subjects
    each to its own significance test.

    Rationale: a global statistic can be dominated by one lucky stretch. A slip
    that 'repairs' three answers is indistinguishable from noise. Requiring each
    displaced segment to be independently long enough AND independently above
    chance is what stops a lottery-ticket answer sheet from paying out.
    """

    def __init__(self, sheet: ResponseSheet, cfg: AdjudicationConfig) -> None:
        self.sheet = sheet
        self.cfg = cfg
        self.chance = 1.0 / len(sheet.options)

    def segments(self, alignment: Alignment) -> List[Segment]:
        runs: List[Tuple[int, int, int]] = []
        cur_off, start, last = None, None, None
        for q in range(self.sheet.n_questions):
            d = alignment.offset(q)
            if d is None:
                continue
            if d != cur_off:
                if cur_off is not None:
                    runs.append((cur_off, start, last))
                cur_off, start = d, q
            last = q
        if cur_off is not None:
            runs.append((cur_off, start, last))

        out = []
        for off, s, e in runs:
            items = [q for q in range(s, e + 1) if alignment.offset(q) is not None]
            n = len(items)
            k = sum(
                1
                for q in items
                if self.sheet.marks[alignment.pairs[q]] == self.sheet.key[q]
            )
            p = binom_sf(k, n, self.chance)
            coherent = (
                off == 0
                or (n >= self.cfg.min_segment_length and p <= self.cfg.segment_binom_alpha)
            )
            out.append(Segment(off, s, e, n, k, p, coherent))
        return out

    def change_points(self, segments: List[Segment]) -> List[Dict]:
        """Human-readable account of each displacement change."""
        events = []
        for a, b in zip(segments, segments[1:]):
            delta = b.offset - a.offset
            if delta > 0:
                cause = (
                    f"{delta} physical row(s) between row {a.q_end + 1 + a.offset + 1} "
                    f"and row {b.q_start + b.offset + 1} carry no question "
                    f"(candidate skipped {delta} bubble row(s))"
                )
            else:
                cause = (
                    f"{-delta} question(s) around Q{a.q_end + 2} were passed over on the "
                    f"paper without consuming a bubble row"
                )
            events.append(
                {
                    "at_question": b.q_start + 1,
                    "offset_before": a.offset,
                    "offset_after": b.offset,
                    "magnitude": delta,
                    "mechanism": cause,
                }
            )
        return events


# ==============================================================================
# 6. NULL CALIBRATION  --  what would this detector do to an innocent sheet?
# ==============================================================================


class CoherenceScanStatistic:
    """
    THE TEST STATISTIC.

        T = max over displacement d != 0, and over every contiguous window W
            of at least L_min questions, of

                -log10 P( at least k of |W| correct | pure chance )

    In words: 'anywhere on this sheet, at any displacement, what is the single
    most statistically surprising coherent block of correct answers?'

    WHY THIS AND NOT THE OBVIOUS ALTERNATIVES
    -----------------------------------------
    Three candidate statistics were measured against a planted shift (a genuine
    positive control) and a Monte-Carlo null of 1200-3000 sheets:

        statistic                       p(true positive)   p(this candidate)
        forward evidence ratio               0.0250              0.66
        Viterbi log-score gain               0.0183              0.68
        raw score gain (marks won)           0.1432              0.53
        coherence scan (this one)            0.0003              0.90

    Raw score gain is close to useless: it gave a genuine shift a p-value of
    0.14. The discriminating feature is the contiguity of the gained marks.

    A candidate who answered incorrectly, or who is fishing for a lucky alignment,
    picks up scattered matches all over the sheet. A displaced pen picks up a
    solid unbroken block. Scoring the block, not the total, is what makes
    the test both powerful and very hard to farm.

    The likelihood-based statistics also fail a subtler test: they are not
    scale-free. A sheet with a poor identity score has more headroom, so it
    generates a larger gain by chance alone. The scan statistic is calibrated
    against chance directly and does not inherit that bias.

    Multiplicity over all (displacement, window) pairs is handled exactly,
    because the Monte-Carlo null maximises over the same search space.
    """

    _tail_cache: Dict[Tuple[int, int, int], float] = {}

    def __init__(self, cfg: AdjudicationConfig, n_options: int = 4) -> None:
        self.cfg = cfg
        self.C = n_options

    def _tail(self, k: int, n: int) -> float:
        key = (k, n, self.C)
        v = self._tail_cache.get(key)
        if v is None:
            v = max(binom_sf(k, n, 1.0 / self.C), 1e-300)
            self._tail_cache[key] = v
        return v

    def compute(self, sheet: ResponseSheet) -> Tuple[float, Optional[Dict]]:
        N, M = sheet.n_questions, sheet.n_rows
        D, Lmin = self.cfg.max_displacement, self.cfg.min_segment_length
        best, arg = 0.0, None
        for d in range(-D, D + 1):
            if d == 0:
                continue
            idx = [q for q in range(N) if 0 <= q + d < M]
            if len(idx) < Lmin:
                continue
            cum = [0] * (len(idx) + 1)
            for i, q in enumerate(idx):
                hit = sheet.marks[q + d] is not None and sheet.marks[q + d] == sheet.key[q]
                cum[i + 1] = cum[i] + (1 if hit else 0)
            for i in range(len(idx)):
                for j in range(i + Lmin, len(idx) + 1):
                    n, k = j - i, cum[j] - cum[i]
                    if k * self.C <= n:          # not even above chance; skip
                        continue
                    v = -math.log10(self._tail(k, n))
                    if v > best:
                        best = v
                        arg = {
                            "offset": d,
                            "q_start": idx[i] + 1,
                            "q_end": idx[j - 1] + 1,
                            "n_items": n,
                            "n_correct": k,
                            "binom_p": self._tail(k, n),
                        }
        return best, arg

    # -- batched evaluation -------------------------------------------------
    #
    # The permutation calibration evaluates this statistic once per draw, which
    # is roughly 70% of the runtime of a single adjudication. Scanning one sheet
    # under numpy is no faster than the loop above, because the arrays are too
    # small to cover the call overhead. Scanning a CHUNK of draws in one pass
    # amortises that overhead and measures about seven times faster.
    #
    # Only the statistic is batched; the draws are still generated one at a
    # time in the same order, so the random stream is unchanged. The result is
    # identical to calling `compute` on each sheet in turn, and the pure-Python
    # path below is used whenever numpy is absent.

    def _statistic_table(self, n_max: int):
        """T[n][k] = -log10 P(X >= k | n, 1/C), computed once per length."""
        cache = getattr(self, "_stat_table", None)
        if cache is not None and cache[0] >= n_max:
            return cache[1]
        import numpy as np
        T = np.zeros((n_max + 1, n_max + 1))
        for n in range(n_max + 1):
            for k in range(n + 1):
                if k * self.C > n:
                    T[n, k] = -math.log10(self._tail(k, n))
        self._stat_table = (n_max, T)
        return T

    def compute_batch(self, sheets: Sequence[ResponseSheet]) -> List[float]:
        """Statistic for each sheet. Equals [self.compute(s)[0] for s in sheets]."""
        if not sheets:
            return []
        try:
            import numpy as np
        except ImportError:
            return [self.compute(s)[0] for s in sheets]

        N, M = sheets[0].n_questions, sheets[0].n_rows
        if any(s.n_questions != N or s.n_rows != M for s in sheets):
            return [self.compute(s)[0] for s in sheets]

        D, Lmin = self.cfg.max_displacement, self.cfg.min_segment_length
        code = {o: i for i, o in enumerate(sheets[0].options)}
        P = len(sheets)
        marks = np.full((P, M), -1, dtype=np.int16)
        keys = np.full((P, N), -2, dtype=np.int16)
        for r, s in enumerate(sheets):
            for c, m in enumerate(s.marks):
                if m is not None:
                    marks[r, c] = code[m]
            for c, k in enumerate(s.key):
                keys[r, c] = code[k]

        T = self._statistic_table(max(N, M))
        best = np.zeros(P)
        for d in range(-D, D + 1):
            if d == 0:
                continue
            idx = np.array([q for q in range(N) if 0 <= q + d < M], dtype=np.int64)
            if len(idx) < Lmin:
                continue
            hits = (marks[:, idx + d] == keys[:, idx]).astype(np.int32)
            cum = np.concatenate([np.zeros((P, 1), np.int32), np.cumsum(hits, 1)], 1)
            L = len(idx)
            i = np.arange(L)[:, None]
            j = np.arange(L + 1)[None, :]
            n = j - i
            ok = n >= Lmin
            k = cum[:, j] - cum[:, i]
            v = T[np.where(ok, n, 0)[None, :, :], np.where(ok[None], k, 0)]
            best = np.maximum(best, np.where(ok[None], v, 0.0).reshape(P, -1).max(1))
        return [float(x) for x in best]


class NullCalibrator:
    """
    A Bayes factor is only as trustworthy as its model. The Monte-Carlo null
    asks an assumption-free question instead:

        'How often does this exact detector produce evidence this strong on a
         sheet that we KNOW contains no shift?'

    Three nulls, and we report the WORST (largest) p-value -- the most
    conservative reading.

      N1 KEY-MARGINAL : resample the key from its own option marginals.
                        Breaks the question-mark correspondence entirely.

      N2 ROTATION     : circularly rotate the candidate's OWN marks by an amount
                        outside the displacement band. This preserves the run
                        structure of the responses -- the long AAAA / DDDD
                        streaks that inflate accidental alignments -- and is by
                        far the sharpest null for streaky answer sheets.

      N3 BLOCK-BOOTSTRAP : resample contiguous blocks of the candidate's marks,
                        preserving local autocorrelation but destroying global
                        alignment.
    """

    def __init__(self, model: ScoringModel, theta: float) -> None:
        self.model = model
        self.cfg = model.cfg
        self.theta = theta
        self.sheet = model.sheet
        self.scan = CoherenceScanStatistic(model.cfg, model.n_opts)

    def _statistic(self, sheet: ResponseSheet) -> float:
        """The coherence scan statistic. See CoherenceScanStatistic for why."""
        return self.scan.compute(sheet)[0]

    def evidence_ratio(self, sheet: ResponseSheet) -> float:
        """Secondary diagnostic: the log evidence ratio at the operating ability.

        Reported but NOT used as the test statistic: it was measured to be an
        order of magnitude less discriminative (see CoherenceScanStatistic).
        """
        m = ScoringModel(sheet, self.cfg)
        hmm = BandedPairHMM(m, self.theta)
        _, Z = hmm.forward()
        L0 = hmm.identity_log_likelihood()
        return log_diff_exp(Z, L0) - L0

    def _null_key_marginal(self, rng: random.Random) -> ResponseSheet:
        counts = {o: 0 for o in self.sheet.options}
        for k in self.sheet.key:
            counts[k] += 1
        pool = [o for o, c in counts.items() for _ in range(c)]
        rng.shuffle(pool)
        return replace(self.sheet, key=tuple(pool))

    def _null_rotation(self, rng: random.Random) -> ResponseSheet:
        n = self.sheet.n_rows
        lo = self.cfg.max_displacement + 2
        k = rng.randrange(lo, n - lo) if n > 2 * lo else rng.randrange(1, n)
        return replace(self.sheet, marks=tuple(self.sheet.marks[k:] + self.sheet.marks[:k]))

    def _null_block_bootstrap(self, rng: random.Random, block: int = 5) -> ResponseSheet:
        n = self.sheet.n_rows
        out: List[Optional[str]] = []
        while len(out) < n:
            s = rng.randrange(0, n)
            out.extend(self.sheet.marks[(s + i) % n] for i in range(block))
        return replace(self.sheet, marks=tuple(out[:n]))

    def run(self, n_perm: Optional[int] = None, early_stop: bool = False) -> Dict:
        """
        `early_stop=True` terminates each null as soon as the decision is
        provably settled. This is EXACT.

        The p-value is (exceedances + 1) / (draws + 1); passing at level alpha
        needs e <= alpha*(n+1) - 1. At alpha = 0.001 with n = 4000 that allows
        e <= 3, so the fourth exceedance settles it and the remaining draws
        cannot change the outcome. Sheets with no shift usually settle within a
        handful of draws, well short of the full budget.

        Left OFF by default: the headline report plots the null DISTRIBUTIONS,
        which need the complete sample. Benchmarks and batch screening should
        turn it on.
        """
        n_perm = n_perm or self.cfg.n_permutations
        rng = random.Random(self.cfg.seed)
        observed = self._statistic(self.sheet)
        allowed = math.floor(self.cfg.permutation_alpha * (n_perm + 1) - 1)

        generators = {
            "key_marginal": self._null_key_marginal,
            "rotation": self._null_rotation,
            "block_bootstrap": self._null_block_bootstrap,
        }
        results = {}
        # Draws are generated one at a time, in the same order as before, so the
        # random stream is unchanged. Only the statistic is evaluated in chunks,
        # which is where the time goes.
        #
        # 256 balances two opposing costs. Array setup is a fixed charge per
        # chunk, so larger chunks amortise it better. Early stopping can only
        # act at a chunk boundary, so larger chunks waste draws on sheets that
        # settle in a handful. At the default level the exceedance budget is 9,
        # so an error-free sheet settles well inside one chunk either way, while
        # 256 is long enough that the per-chunk overhead is a few percent.
        CHUNK = 256
        for name, gen in generators.items():
            draws = []
            exceed = 0
            remaining = n_perm
            stop = False
            while remaining > 0 and not stop:
                batch = []
                for _ in range(min(CHUNK, remaining)):
                    remaining -= 1
                    try:
                        batch.append(gen(rng))
                    except ValueError:
                        continue
                for v in self.scan.compute_batch(batch):
                    draws.append(v)
                    if early_stop and v >= observed:
                        exceed += 1
                        if exceed > allowed:
                            stop = True
                            break
            exceed = sum(1 for d in draws if d >= observed)
            # add-one (Davison-Hinkley) estimator: never reports p = 0
            p = (exceed + 1) / (len(draws) + 1)
            draws_sorted = sorted(draws)
            results[name] = {
                "p_value": p,
                "n_draws": len(draws),
                "null_mean": sum(draws) / len(draws),
                "null_q95": draws_sorted[int(0.95 * len(draws))],
                "null_q999": draws_sorted[min(len(draws) - 1, int(0.999 * len(draws)))],
                "draws": draws,
            }
        worst = max(results.values(), key=lambda r: r["p_value"])
        _, window = self.scan.compute(self.sheet)
        return {
            "observed_statistic": observed,
            "scan_window": window,
            "evidence_ratio": self.evidence_ratio(self.sheet),
            "nulls": results,
            "p_value": worst["p_value"],
            "decisive_null": max(results, key=lambda k: results[k]["p_value"]),
        }


# ==============================================================================
# 7. ADJUDICATION  --  the gate, and the item-level award rule
# ==============================================================================


@dataclass
class Adjudication:
    sheet: ResponseSheet
    cfg: AdjudicationConfig
    evidence: Dict
    alignment: Alignment
    theta_hat: float
    posteriors: Dict[int, Dict[Optional[int], float]]
    segments: List[Segment]
    change_points: List[Dict]
    calibration: Dict
    gates: Dict[str, Dict]
    accepted: bool
    awarded_map: Dict[int, int]
    raw_score: int
    adjudicated_score: int
    item_ledger: List[Dict]
    break_even: float
    warnings: List[str]

    @property
    def verdict(self) -> str:
        if self.accepted:
            return "RE-REGISTRATION ACCEPTED"
        return "NO RE-REGISTRATION -- ORIGINAL SCORE STANDS"


class Adjudicator:
    """
    Orchestrates the pipeline and applies the pre-registered gates.

    THE AWARD RULE
    --------------
    Even after the global gates pass, credit is NOT granted wholesale. Each
    question is re-registered only if its own posterior on the MAP displacement
    clears `item_posterior_threshold`, and only if it lies inside a segment that
    passed its own coherence test. Everything else keeps its original
    registration. This is deliberately conservative: the cost of wrongly denying
    a repair is one candidate's marks; the cost of wrongly granting one is the
    integrity of the examination.

    Note also that re-registration is applied SYMMETRICALLY: questions that go
    from right to wrong under the accepted alignment are recorded as losses and
    counted. There is no cherry-picking.
    """

    def __init__(self, sheet: ResponseSheet, cfg: Optional[AdjudicationConfig] = None) -> None:
        self.sheet = sheet
        self.cfg = cfg or AdjudicationConfig()
        self.model = ScoringModel(sheet, self.cfg)

    def run(self, n_permutations: Optional[int] = None, verbose: bool = True,
            early_stop: bool = False) -> Adjudication:
        warnings = list(self.model.validate_no_free_unmatch())

        if verbose:
            print("  [1/5] marginalising ability and computing evidence ...")
        engine = EvidenceEngine(self.model)
        evidence = engine.evaluate(include_profile=not early_stop)
        theta_hat = evidence["operating_theta"]

        if verbose:
            print("  [2/5] Viterbi decoding the MAP re-registration ...")
        alignment = BandedPairHMM(self.model, theta_hat).viterbi()

        if verbose:
            print("  [3/5] forward-backward for per-question posteriors ...")
        posteriors = engine.posterior_mean_offsets()

        analyzer = SegmentAnalyzer(self.sheet, self.cfg)
        segments = analyzer.segments(alignment)
        change_points = analyzer.change_points(segments)

        if verbose:
            print(f"  [4/5] Monte-Carlo null calibration "
                  f"({n_permutations or self.cfg.n_permutations} draws x 3 nulls) ...")
        calibration = NullCalibrator(self.model, theta_hat).run(
            n_permutations, early_stop=early_stop)

        if verbose:
            print("  [5/5] applying pre-registered gates ...")

        displaced = [s for s in segments if s.offset != 0]
        gates = {
            "bayes_factor": {
                "value": evidence["log10_bayes_factor"],
                "threshold": math.log10(self.cfg.bayes_factor_threshold),
                "passed": evidence["log_bayes_factor"]
                >= math.log(self.cfg.bayes_factor_threshold),
                "description": "log10 BF(shift : no-shift) exceeds the decisive threshold",
            },
            "posterior": {
                "value": evidence["posterior_h1"],
                "threshold": self.cfg.posterior_shift_threshold,
                "passed": evidence["posterior_h1"] >= self.cfg.posterior_shift_threshold,
                "description": "posterior P(shift | marks) after the base-rate prior",
            },
            "monte_carlo": {
                "value": calibration["p_value"],
                "threshold": self.cfg.permutation_alpha,
                "passed": calibration["p_value"] <= self.cfg.permutation_alpha,
                "description": f"worst-case Monte-Carlo p-value "
                               f"(binding null: {calibration['decisive_null']})",
            },
            "segment_coherence": {
                "value": sum(1 for s in displaced if s.coherent),
                "threshold": len(displaced),
                "passed": bool(displaced) and all(s.coherent for s in displaced),
                "description": "every displaced segment is long enough and above chance",
            },
            "non_trivial": {
                "value": len(displaced),
                "threshold": 1,
                "passed": len(displaced) >= 1,
                "description": "the MAP re-registration actually differs from the identity",
            },
        }
        accepted = all(g["passed"] for g in gates.values())

        # ---- item-level award --------------------------------------------
        awarded: Dict[int, int] = {}
        ledger: List[Dict] = []
        coherent_q = set()
        if accepted:
            for s in segments:
                if s.offset == 0 or s.coherent:
                    coherent_q.update(range(s.q_start, s.q_end + 1))

        for q in range(self.sheet.n_questions):
            orig_row = q if q < self.sheet.n_rows else None
            orig_ok = orig_row is not None and self.sheet.marks[orig_row] == self.sheet.key[q]
            post = posteriors.get(q, {})
            map_d, map_p = (None, 0.0)
            if post:
                map_d, map_p = max(post.items(), key=lambda kv: kv[1])

            new_row = orig_row
            reason = "kept: original registration"
            if accepted and q in coherent_q:
                prop = alignment.pairs.get(q)
                if prop is None:
                    if map_p >= self.cfg.item_posterior_threshold and map_d is None:
                        new_row = None
                        reason = f"re-registered: unanswered (posterior {map_p:.4f})"
                    else:
                        reason = f"kept: blank proposed but posterior only {map_p:.4f}"
                elif prop != orig_row:
                    if map_p >= self.cfg.item_posterior_threshold and map_d == prop - q:
                        new_row = prop
                        reason = (
                            f"re-registered to row {prop + 1} "
                            f"(offset {prop - q:+d}, posterior {map_p:.4f})"
                        )
                    else:
                        reason = (
                            f"kept: offset {prop - q:+d} proposed but posterior "
                            f"{map_p:.4f} < {self.cfg.item_posterior_threshold}"
                        )
                else:
                    reason = "kept: alignment agrees with original registration"

            if new_row is not None:
                awarded[q] = new_row
            new_ok = new_row is not None and self.sheet.marks[new_row] == self.sheet.key[q]
            ledger.append(
                {
                    "question": q + 1,
                    "key": self.sheet.key[q],
                    "original_row": None if orig_row is None else orig_row + 1,
                    "original_mark": None if orig_row is None else self.sheet.marks[orig_row],
                    "original_correct": orig_ok,
                    "final_row": None if new_row is None else new_row + 1,
                    "final_mark": None if new_row is None else self.sheet.marks[new_row],
                    "final_correct": new_ok,
                    "map_offset": map_d,
                    "map_posterior": map_p,
                    "change": (
                        "GAIN" if (new_ok and not orig_ok)
                        else "LOSS" if (orig_ok and not new_ok)
                        else "-"
                    ),
                    "reason": reason,
                }
            )

        return Adjudication(
            sheet=self.sheet,
            cfg=self.cfg,
            evidence=evidence,
            alignment=alignment,
            theta_hat=theta_hat,
            posteriors=posteriors,
            segments=segments,
            change_points=change_points,
            calibration=calibration,
            gates=gates,
            accepted=accepted,
            awarded_map=awarded,
            raw_score=self.sheet.raw_score(),
            adjudicated_score=sum(1 for r in ledger if r["final_correct"]),
            item_ledger=ledger,
            break_even=self.model.break_even_questions(theta_hat),
            warnings=warnings,
        )


# ==============================================================================
# 8. REPORTING
# ==============================================================================


class Reporter:
    """Renders the adjudication as a document an examination board can act on."""

    def __init__(self, adj: Adjudication) -> None:
        self.a = adj

    def text_report(self) -> str:
        a, s, L = self.a, self.a.sheet, []
        w = 78
        L.append("=" * w)
        L.append("OMR ALIGNMENT ADJUDICATION REPORT".center(w))
        L.append("=" * w)
        L.append(f"Candidate            : {s.candidate_id}")
        L.append(f"Subject              : {s.subject}")
        L.append(f"Questions / marks    : {s.n_questions} / {s.n_rows}")
        L.append(f"Options per item     : {len(s.options)}  (chance = {1/len(s.options):.3f})")
        L.append("")
        L.append(f">>> VERDICT: {a.verdict}")
        L.append("")
        L.append(f"Original score       : {a.raw_score} / {s.n_questions} "
                 f"({100*a.raw_score/s.n_questions:.1f}%)")
        L.append(f"Adjudicated score    : {a.adjudicated_score} / {s.n_questions} "
                 f"({100*a.adjudicated_score/s.n_questions:.1f}%)")
        gains = sum(1 for r in a.item_ledger if r["change"] == "GAIN")
        losses = sum(1 for r in a.item_ledger if r["change"] == "LOSS")
        L.append(f"Items gained / lost  : {gains} / {losses}")
        L.append("")

        L.append("-" * w)
        L.append("1. EVIDENCE")
        L.append("-" * w)
        e = a.evidence
        L.append(f"  log P(marks | H0, no shift)    = {e['log_p_h0']:+11.4f} nats")
        L.append(f"  log P(marks | H1, some shift)  = {e['log_p_h1']:+11.4f} nats")
        L.append(f"  log10 Bayes factor  BF(1:0)    = {e['log10_bayes_factor']:+11.4f}")
        bf = e["log10_bayes_factor"]
        band = ("decisive FOR shift" if bf > 2 else "strong FOR shift" if bf > 1
                else "substantial FOR shift" if bf > 0.5
                else "bare mention FOR shift" if bf > 0
                else "bare mention AGAINST" if bf > -0.5
                else "substantial AGAINST" if bf > -1
                else "strong AGAINST" if bf > -2 else "decisive AGAINST shift")
        L.append(f"  Jeffreys interpretation        : {band}")
        L.append(f"  Prior P(sheet has a slip)      = {a.cfg.sheet_slip_rate:.4f}")
        L.append(f"  Posterior P(shift | marks)     = {e['posterior_h1']:.6f}")
        L.append(f"  Operating ability theta        = {a.theta_hat:.4f}")
        L.append(f"     source: {e['theta_source']}")
        L.append(f"  Ability implied by this paper  = {e['implied_ability_from_this_paper']:.4f}"
                 f"   (DIAGNOSTIC ONLY -- never used for decoding)")
        L.append(f"  Evidence break-even            = {a.break_even:.2f} questions")
        L.append("     (a shift must newly repair at least this many answers before")
        L.append("      the evidence outweighs the prior improbability of the slip)")
        L.append("")
        if e["theta_profile"]:
            L.append("  ROBUSTNESS TO THE ABILITY ASSUMPTION")
            L.append("  The verdict below is reported at one assumed ability. This table")
            L.append("  sweeps every admissible ability, including the most generous:")
            L.append(f"  {'theta':>7}{'log10 BF':>11}{'break-even':>12}{'best score':>12}  MAP reading")
            for row in e["theta_profile"]:
                L.append(f"  {row['theta']:>7.3f}{row['log10_bf']:>11.3f}"
                         f"{row['break_even']:>12.2f}{row['map_best_score']:>8d}/{s.n_questions:<3d}"
                         f"  {'identity (no shift)' if row['map_is_identity'] else 'RE-REGISTERED'}")
            L.append("")

        L.append("-" * w)
        L.append("2. MONTE-CARLO CALIBRATION  (assumption-free false-positive check)")
        L.append("-" * w)
        c = a.calibration
        L.append("  Statistic: coherence scan -- the most surprising contiguous block of")
        L.append("  correct answers at ANY non-zero displacement, anywhere on the sheet.")
        L.append(f"  Observed T = {c['observed_statistic']:.4f}   "
                 f"(i.e. best block p = 10^-{c['observed_statistic']:.2f})")
        wnd = c.get("scan_window")
        if wnd:
            L.append(f"  Strongest block: Q{wnd['q_start']}-Q{wnd['q_end']} at offset "
                     f"{wnd['offset']:+d}, {wnd['n_correct']}/{wnd['n_items']} correct "
                     f"(binomial p = {wnd['binom_p']:.3g})")
        else:
            L.append("  No block above chance exists at any non-zero displacement.")
        L.append(f"  Secondary diagnostic, log evidence ratio = {c['evidence_ratio']:+.4f}")
        L.append(f"  {'null':<18}{'p-value':>10}{'null mean':>12}{'null q95':>11}{'null q99.9':>12}")
        for name, r in c["nulls"].items():
            L.append(f"  {name:<18}{r['p_value']:>10.5f}{r['null_mean']:>12.3f}"
                     f"{r['null_q95']:>11.3f}{r['null_q999']:>12.3f}")
        L.append(f"  Reported (worst-case) p-value  = {c['p_value']:.5f}  "
                 f"[binding null: {c['decisive_null']}]")
        L.append("")

        L.append("-" * w)
        L.append("3. PRE-REGISTERED GATES")
        L.append("-" * w)
        for name, g in a.gates.items():
            mark = "PASS" if g["passed"] else "FAIL"
            L.append(f"  [{mark}] {name:<20} value={g['value']:.6g}  "
                     f"threshold={g['threshold']:.6g}")
            L.append(f"         {g['description']}")
        L.append("")

        L.append("-" * w)
        L.append("4. MAP RE-REGISTRATION  (best-supported reading of the sheet)")
        L.append("-" * w)
        L.append(f"  Segments of constant displacement: {len(a.segments)}")
        L.append(f"  {'segment':<26}{'n':>4}{'correct':>9}{'binom p':>11}  coherent")
        for sg in a.segments:
            L.append(f"  {sg.label:<26}{sg.n_items:>4}{sg.n_correct:>9}"
                     f"{sg.binom_p:>11.4g}  {'yes' if sg.coherent else 'NO'}")
        if a.alignment.orphan_rows:
            L.append(f"  Orphan marks (rows read by no question): "
                     f"{[r+1 for r in a.alignment.orphan_rows]}")
        if a.alignment.blank_questions:
            L.append(f"  Questions left unmatched: "
                     f"{[q+1 for q in a.alignment.blank_questions]}")
        L.append("")
        if a.change_points:
            L.append("  Detected displacement changes:")
            for cp in a.change_points:
                L.append(f"    * at Q{cp['at_question']}: offset "
                         f"{cp['offset_before']:+d} -> {cp['offset_after']:+d}")
                L.append(f"      mechanism: {cp['mechanism']}")
        else:
            L.append("  No displacement change detected: the MAP reading is the identity.")
        L.append("")

        L.append("-" * w)
        L.append("5. ITEM LEDGER  (every question, every decision)")
        L.append("-" * w)
        L.append(f"  {'Q':>3} {'key':>4} {'orig':>10} {'final':>10} {'off':>4} "
                 f"{'post':>7} {'chg':>5}  reason")
        for r in a.item_ledger:
            orig = f"r{r['original_row']}={r['original_mark']}" if r["original_row"] else "-"
            fin = f"r{r['final_row']}={r['final_mark']}" if r["final_row"] else "BLANK"
            off = "-" if r["map_offset"] is None else f"{r['map_offset']:+d}"
            oc = "*" if r["original_correct"] else " "
            fc = "*" if r["final_correct"] else " "
            L.append(f"  {r['question']:>3} {r['key']:>4} {orig:>9}{oc} {fin:>9}{fc} "
                     f"{off:>4} {r['map_posterior']:>7.4f} {r['change']:>5}  {r['reason']}")
        L.append("  ('*' marks a correct answer under that registration)")
        L.append("")

        if a.warnings:
            L.append("-" * w)
            L.append("6. CONFIGURATION WARNINGS")
            L.append("-" * w)
            for wn in a.warnings:
                L.append(f"  ! {wn}")
            L.append("")

        L.append("=" * w)
        L.append("Constraints honoured: the marked sequence was never altered, reordered,")
        L.append("or invented. Only the question<->row registration was considered, and")
        L.append("only strictly increasing registrations were admissible.")
        L.append("=" * w)
        return "\n".join(L)

    def to_dict(self) -> Dict:
        a = self.a
        return {
            "candidate": a.sheet.candidate_id,
            "subject": a.sheet.subject,
            "verdict": a.verdict,
            "accepted": a.accepted,
            "raw_score": a.raw_score,
            "adjudicated_score": a.adjudicated_score,
            "n_questions": a.sheet.n_questions,
            "theta_hat": a.theta_hat,
            "break_even_questions": a.break_even,
            "evidence": {
                k: v for k, v in a.evidence.items()
                if k not in ("per_theta", "theta_posterior")
            },
            "gates": a.gates,
            "monte_carlo": {
                "observed_statistic": a.calibration["observed_statistic"],
                "p_value": a.calibration["p_value"],
                "decisive_null": a.calibration["decisive_null"],
                "per_null": {
                    k: {kk: vv for kk, vv in v.items() if kk != "draws"}
                    for k, v in a.calibration["nulls"].items()
                },
            },
            "segments": [
                {
                    "offset": s.offset,
                    "q_start": s.q_start + 1,
                    "q_end": s.q_end + 1,
                    "n_items": s.n_items,
                    "n_correct": s.n_correct,
                    "binom_p": s.binom_p,
                    "coherent": s.coherent,
                }
                for s in a.segments
            ],
            "change_points": a.change_points,
            "item_ledger": a.item_ledger,
            "config": {
                k: v for k, v in a.cfg.__dict__.items()
            },
            "warnings": a.warnings,
        }


# ==============================================================================
# 9. VISUALISATION
# ==============================================================================


class Visualizer:
    """Six figures, each answering one question a board would actually ask."""

    def __init__(self, adj: Adjudication, outdir: str) -> None:
        self.a = adj
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def render_all(self) -> List[str]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths = []
        paths.append(self._fig_posterior_heatmap(plt))
        paths.append(self._fig_correctness_track(plt))
        paths.append(self._fig_llr_trajectory(plt))
        paths.append(self._fig_null_distributions(plt))
        paths.append(self._fig_sensitivity(plt))
        paths.append(self._fig_theta_profile(plt))
        return paths

    # -- 6 ------------------------------------------------------------------
    def _fig_theta_profile(self, plt):
        a = self.a
        prof = a.evidence["theta_profile"]
        th = [r["theta"] for r in prof]
        bf = [r["log10_bf"] for r in prof]
        best = [r["map_best_score"] for r in prof]

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(th, bf, lw=2.2, color="#1b6ca8", marker="o", ms=4,
                label="log10 Bayes factor for a shift")
        ax.axhline(math.log10(a.cfg.bayes_factor_threshold), color="#c0392b", ls="--",
                   lw=1.4, label=f"decision threshold (BF={a.cfg.bayes_factor_threshold:g})")
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(a.theta_hat, color="#2a9d5c", lw=1.6, ls=":",
                   label=f"operating ability = {a.theta_hat:.2f}")
        ax.set_xlabel("assumed candidate ability  theta  (competence from OTHER evidence)")
        ax.set_ylabel("log10 Bayes factor", color="#1b6ca8")
        ax2 = ax.twinx()
        ax2.plot(th, best, lw=1.6, color="#e2a33c", marker="s", ms=3,
                 label="score under the best admissible re-registration")
        ax2.axhline(a.raw_score, color="#e2a33c", ls=":", lw=1.2)
        ax2.set_ylabel("best attainable score", color="#e2a33c")
        ax2.set_ylim(0, a.sheet.n_questions)
        ax.set_title("Does the conclusion survive ANY assumption about the candidate?\n"
                     "(if the blue curve never crosses the red line, the verdict is "
                     "independent of how able we assume them to be)")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        p = os.path.join(self.outdir, "6_ability_robustness.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        return p

    # -- 1 ------------------------------------------------------------------
    def _fig_posterior_heatmap(self, plt):
        import numpy as np
        a = self.a
        D = a.cfg.max_displacement
        offs = list(range(-D, D + 1))
        rows = offs + ["blank"]
        Zm = np.zeros((len(rows), a.sheet.n_questions))
        for q, post in a.posteriors.items():
            for d, p in post.items():
                i = len(offs) if d is None else (offs.index(d) if d in offs else None)
                if i is not None:
                    Zm[i, q] += p

        fig, ax = plt.subplots(figsize=(15, 4.2))
        im = ax.imshow(Zm, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[0.5, a.sheet.n_questions + 0.5, len(rows) - 0.5, -0.5])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{o:+d}" if isinstance(o, int) else "blank" for o in rows])
        ax.set_xlabel("Question number")
        ax.set_ylabel("Displacement (row - question)")
        ax.set_title("Posterior distribution over registration displacement, per question\n"
                     "(marginalised over every admissible alignment and over candidate ability)")
        ax.axhline(offs.index(0) + 0.5, color="w", lw=0.6, alpha=0.4)
        ax.axhline(offs.index(0) - 0.5, color="w", lw=0.6, alpha=0.4)
        mp = [a.alignment.offset(q) for q in range(a.sheet.n_questions)]
        xs = [q + 1 for q, d in enumerate(mp) if d is not None]
        ys = [offs.index(d) for d in mp if d is not None and d in offs]
        ax.plot(xs, ys, color="crimson", lw=1.6, marker="o", ms=3,
                label="MAP (Viterbi) path")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.colorbar(im, ax=ax, label="posterior probability", pad=0.01)
        fig.tight_layout()
        p = os.path.join(self.outdir, "1_displacement_posterior.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        return p

    # -- 2 ------------------------------------------------------------------
    def _fig_correctness_track(self, plt):
        a = self.a
        n = a.sheet.n_questions
        fig, ax = plt.subplots(figsize=(15, 3.0))
        for r in a.item_ledger:
            q = r["question"]
            ax.add_patch(plt.Rectangle((q - 0.45, 1.05), 0.9, 0.8,
                         color="#2a9d5c" if r["original_correct"] else "#d94040"))
            col = "#2a9d5c" if r["final_correct"] else "#d94040"
            if r["change"] == "GAIN":
                col = "#1b6ca8"
            elif r["change"] == "LOSS":
                col = "#e2a33c"
            ax.add_patch(plt.Rectangle((q - 0.45, 0.05), 0.9, 0.8, color=col))
        ax.set_xlim(0.3, n + 0.7); ax.set_ylim(-0.1, 2.1)
        ax.set_yticks([0.45, 1.45]); ax.set_yticklabels(["adjudicated", "as marked"])
        ax.set_xticks(range(1, n + 1, 2))
        ax.set_xlabel("Question number")
        ax.set_title(f"Item-level outcome  |  original {a.raw_score}/{n}  ->  "
                     f"adjudicated {a.adjudicated_score}/{n}   [{a.verdict}]")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
                   ("#2a9d5c", "#d94040", "#1b6ca8", "#e2a33c")]
        ax.legend(handles, ["correct", "incorrect", "gained", "lost"],
                  ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        p = os.path.join(self.outdir, "2_item_outcomes.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        return p

    # -- 3 ------------------------------------------------------------------
    def _fig_llr_trajectory(self, plt):
        a = self.a
        model = ScoringModel(a.sheet, a.cfg)
        th = a.theta_hat
        cum_h0, cum_h1, x = [], [], []
        c0 = c1 = 0.0
        for q in range(a.sheet.n_questions):
            c0 += model.match_score(q, q, th) if q < a.sheet.n_rows else 0.0
            r = a.alignment.pairs.get(q)
            c1 += model.match_score(q, r, th) if r is not None else model.log_blank_open(th)
            x.append(q + 1); cum_h0.append(c0); cum_h1.append(c1)

        fig, ax = plt.subplots(figsize=(13, 4.2))
        ax.plot(x, cum_h0, lw=2, color="#444", label="H0: as marked (identity registration)")
        ax.plot(x, cum_h1, lw=2, color="#1b6ca8", label="H1: MAP re-registration")
        ax2 = ax.twinx()
        ax2.plot(x, [b - c for b, c in zip(cum_h1, cum_h0)], lw=1.4, ls="--",
                 color="#c0392b", label="running log-likelihood ratio")
        ax2.axhline(0, color="#c0392b", lw=0.6, alpha=0.4)
        ax2.set_ylabel("running LLR (nats)", color="#c0392b")
        thr = -model.log_row_skip_open
        ax2.axhline(thr, color="#c0392b", lw=0.8, ls=":",
                    label=f"cost of one slip ({thr:.1f} nats)")
        for cp in a.change_points:
            ax.axvline(cp["at_question"], color="#e2a33c", lw=1.2, alpha=0.8)
        ax.set_xlabel("Question number"); ax.set_ylabel("cumulative log-likelihood (nats)")
        ax.set_title("Evidence accumulation: does the re-registration ever earn its price?")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        p = os.path.join(self.outdir, "3_evidence_trajectory.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        return p

    # -- 4 ------------------------------------------------------------------
    def _fig_null_distributions(self, plt):
        a = self.a
        nulls = a.calibration["nulls"]
        fig, axes = plt.subplots(1, len(nulls), figsize=(15, 3.8), sharey=True)
        obs = a.calibration["observed_statistic"]
        for ax, (name, r) in zip(axes, nulls.items()):
            ax.hist(r["draws"], bins=45, color="#8fa9c4", edgecolor="none")
            ax.axvline(obs, color="#c0392b", lw=2,
                       label=f"observed T = {obs:.2f}")
            ax.axvline(r["null_q999"], color="#2a9d5c", lw=1.2, ls="--",
                       label="null 99.9th pct")
            ax.set_title(f"{name}\np = {r['p_value']:.4f}", fontsize=10)
            ax.set_xlabel("coherence scan statistic T")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Monte-Carlo draws")
        fig.suptitle("Null calibration: how strong is this evidence on sheets known to have no shift?",
                     y=1.04)
        fig.tight_layout()
        p = os.path.join(self.outdir, "4_null_calibration.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        return p

    # -- 5 ------------------------------------------------------------------
    def _fig_sensitivity(self, plt):
        import numpy as np
        a = self.a
        thetas = [0.30 + 0.05 * i for i in range(12)]
        rates = [10 ** (-e) for e in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)]
        Zm = np.zeros((len(rates), len(thetas)))
        for i, rate in enumerate(rates):
            cfg = replace(a.cfg, sheet_slip_rate=rate)
            m = ScoringModel(a.sheet, cfg)
            for j, th in enumerate(thetas):
                hmm = BandedPairHMM(m, th)
                _, Z = hmm.forward()
                L0 = hmm.identity_log_likelihood()
                Zm[i, j] = (log_diff_exp(Z, L0) - L0) / math.log(10)

        fig, ax = plt.subplots(figsize=(9, 4.6))
        im = ax.imshow(Zm, aspect="auto", cmap="RdBu_r",
                       vmin=-max(3, abs(Zm).max()), vmax=max(3, abs(Zm).max()),
                       extent=[thetas[0], thetas[-1], len(rates) - 0.5, -0.5])
        cs = ax.contour(np.linspace(thetas[0], thetas[-1], len(thetas)),
                        np.arange(len(rates)), Zm, levels=[0, 2], colors="k", linewidths=1.2)
        ax.clabel(cs, fmt={0: "BF=1", 2: "BF=100 (decision boundary)"}, fontsize=8)
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([f"{r:.1e}" for r in rates])
        ax.set_xlabel("assumed candidate ability  theta")
        ax.set_ylabel("assumed base rate of sheet slips")
        ax.set_title("Sensitivity of the verdict to the two priors\n"
                     "(colour = log10 Bayes factor for a shift)")
        fig.colorbar(im, ax=ax, label="log10 BF", pad=0.02)
        fig.tight_layout()
        p = os.path.join(self.outdir, "5_prior_sensitivity.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        return p


# ==============================================================================
# 10. VALIDATION  --  does the detector actually work, and how often does it lie?
# ==============================================================================


class SyntheticValidator:
    """
    An examination board should never accept a detector on the strength of one
    case. This harness generates synthetic sheets with KNOWN ground truth and
    measures operating characteristics:

      * power        : P(accept | a real shift of magnitude m occurred at position t)
      * false alarms : P(accept | no shift, candidate performed poorly)

    The adversarial arm additionally simulates a candidate who deliberately
    tries to farm a shift: they answer with a low-entropy, streaky pattern
    (long runs of one option) which maximises accidental alignment.
    """

    def __init__(self, cfg: AdjudicationConfig, n_questions: int = 46,
                 options: Sequence[str] = ("A", "B", "C", "D")) -> None:
        self.cfg = cfg
        self.n = n_questions
        self.options = tuple(options)
        self._counter = 0

    def _random_key(self, rng: random.Random) -> Tuple[str, ...]:
        return tuple(rng.choice(self.options) for _ in range(self.n))

    def _honest_marks(self, key, theta, rng) -> List[str]:
        out = []
        for k in key:
            if rng.random() < theta:
                out.append(k)
            else:
                out.append(rng.choice([o for o in self.options if o != k]))
        return out

    def _streaky_marks(self, rng, run_len: int = 4) -> List[str]:
        out = []
        while len(out) < self.n:
            out.extend([rng.choice(self.options)] * rng.randint(2, run_len + 2))
        return out[: self.n]

    def _inject_shift(self, marks: List[str], at: int, mag: int, rng) -> List[str]:
        """Physically simulate skipping `mag` bubble rows at question `at`."""
        shifted = marks[:at] + [rng.choice(self.options) for _ in range(mag)] + marks[at:]
        return shifted[: self.n]

    def _quick_verdict(self, sheet: ResponseSheet, n_null: int = 1200) -> Tuple[float, bool]:
        """
        The FULL gate stack, including the Monte-Carlo calibration, so the power
        and false-alarm figures are honest.

        The Monte-Carlo uses the key-marginal null with `n_null` draws (the main
        pipeline uses three nulls and many more draws, and is therefore strictly
        more conservative than what is measured here).
        """
        model = ScoringModel(sheet, self.cfg)
        ev = EvidenceEngine(model).evaluate()
        theta = ev["operating_theta"]
        align = BandedPairHMM(model, theta).viterbi()
        segs = SegmentAnalyzer(sheet, self.cfg).segments(align)
        disp = [s for s in segs if s.offset != 0]

        scan = CoherenceScanStatistic(self.cfg, model.n_opts)
        obs = scan.compute(sheet)[0]
        self._counter += 1
        rng = random.Random(self.cfg.seed + 1000 * self._counter)
        draws = []
        for _ in range(n_null):
            k = tuple(rng.choice(self.options) for _ in range(sheet.n_questions))
            draws.append(scan.compute(replace(sheet, key=k))[0])
        p = (sum(1 for d in draws if d >= obs) + 1) / (len(draws) + 1)

        ok = (
            ev["log_bayes_factor"] >= math.log(self.cfg.bayes_factor_threshold)
            and ev["posterior_h1"] >= self.cfg.posterior_shift_threshold
            and p <= self.cfg.permutation_alpha
            and bool(disp) and all(s.coherent for s in disp)
        )
        return obs, ok

    def run(self, n_per_cell: int = 60, verbose: bool = True) -> Dict:
        rng = random.Random(self.cfg.seed + 7)
        results = {"honest_no_shift": [], "adversarial_streaky": [], "shifted": {}}

        if verbose:
            print(f"  arm A: honest candidates, NO shift  (n={n_per_cell*3})")
        for theta in (0.35, 0.55, 0.75):
            for _ in range(n_per_cell):
                key = self._random_key(rng)
                marks = self._honest_marks(key, theta, rng)
                bf, ok = self._quick_verdict(ResponseSheet(key, tuple(marks), self.options))
                results["honest_no_shift"].append({"theta": theta, "log10bf": bf, "accepted": ok})

        if verbose:
            print(f"  arm B: adversarial streaky sheets, NO shift  (n={n_per_cell*2})")
        for _ in range(n_per_cell * 2):
            key = self._random_key(rng)
            marks = self._streaky_marks(rng)
            bf, ok = self._quick_verdict(ResponseSheet(key, tuple(marks), self.options))
            results["adversarial_streaky"].append({"log10bf": bf, "accepted": ok})

        if verbose:
            print("  arm C: genuine shifts of magnitude 1 and 2, various abilities")
        for theta in (0.55, 0.70, 0.85):
            for mag in (1, 2):
                cell, accs = [], 0
                for _ in range(n_per_cell):
                    key = self._random_key(rng)
                    honest = self._honest_marks(key, theta, rng)
                    at = rng.randint(5, self.n - 12)
                    marks = self._inject_shift(honest, at, mag, rng)
                    bf, ok = self._quick_verdict(ResponseSheet(key, tuple(marks), self.options))
                    cell.append({"log10bf": bf, "accepted": ok, "at": at})
                    accs += ok
                results["shifted"][f"theta={theta},mag={mag}"] = {
                    "power": accs / n_per_cell, "draws": cell,
                }

        fa_honest = sum(r["accepted"] for r in results["honest_no_shift"]) / len(
            results["honest_no_shift"])
        fa_adv = sum(r["accepted"] for r in results["adversarial_streaky"]) / len(
            results["adversarial_streaky"])
        n_h = len(results["honest_no_shift"])
        n_a = len(results["adversarial_streaky"])
        results["summary"] = {
            "false_alarm_honest": fa_honest,
            "false_alarm_honest_n": n_h,
            "false_alarm_honest_ci95_upper": clopper_pearson_upper(
                round(fa_honest * n_h), n_h),
            "false_alarm_adversarial": fa_adv,
            "false_alarm_adversarial_n": n_a,
            "false_alarm_adversarial_ci95_upper": clopper_pearson_upper(
                round(fa_adv * n_a), n_a),
            "power": {k: v["power"] for k, v in results["shifted"].items()},
        }
        return results

    @staticmethod
    def plot(results: Dict, outdir: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
        ax = axes[0]
        honest = [r["log10bf"] for r in results["honest_no_shift"]]
        adv = [r["log10bf"] for r in results["adversarial_streaky"]]
        shifted = [d["log10bf"] for v in results["shifted"].values() for d in v["draws"]]
        bins = np.linspace(min(honest + adv + shifted), max(honest + adv + shifted), 50)
        ax.hist(honest, bins=bins, alpha=0.65, label="no shift (honest)", color="#8fa9c4")
        ax.hist(adv, bins=bins, alpha=0.65, label="no shift (streaky/adversarial)", color="#e2a33c")
        ax.hist(shifted, bins=bins, alpha=0.65, label="genuine shift", color="#2a9d5c")
        ax.axvline(2.0, color="#c0392b", lw=2, label="decision threshold (BF=100)")
        ax.set_xlabel("log10 Bayes factor"); ax.set_ylabel("simulated sheets")
        ax.set_title("Separation between shifted and unshifted sheets")
        ax.legend(fontsize=8)

        ax = axes[1]
        labels = list(results["shifted"].keys())
        powers = [results["shifted"][k]["power"] for k in labels]
        ax.barh(range(len(labels)), powers, color="#2a9d5c")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1); ax.set_xlabel("detection power")
        s = results["summary"]
        ax.set_title(f"Power by ability and shift magnitude\n"
                     f"false alarms: honest {s['false_alarm_honest']:.3f}, "
                     f"adversarial {s['false_alarm_adversarial']:.3f}")
        for i, p in enumerate(powers):
            ax.text(min(p + 0.02, 0.92), i, f"{p:.2f}", va="center", fontsize=8)
        fig.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, "7_validation_operating_characteristics.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        return p


# ==============================================================================
# 10b. GATE STRESS TEST  --  where does this system actually break?
# ==============================================================================


class GateStressTest:
    """
    ADVERSARIAL SELF-AUDIT.

    Every gate has a price at which it can be bought. The ability prior is the
    softest input in the whole system: it is supplied by a human, it is not
    verifiable from the sheet, and a sympathetic board can always assert 'but
    this candidate is outstanding' more forcefully.

    This harness escalates that assertion -- raising the prior concentration
    from mild to absurd -- and records which gates survive. The purpose is not
    to tune anything. It is to identify, and then publish, WHICH gate is
    load-bearing, so that the board knows exactly where to concentrate its
    scrutiny and its governance.
    """

    def __init__(self, sheet: ResponseSheet, base_cfg: AdjudicationConfig) -> None:
        self.sheet = sheet
        self.cfg = base_cfg

    def run(self, ability: float = 0.85,
            concentrations: Sequence[float] = (10, 30, 60, 120, 250, 500, 1000, 5000),
            n_perm: int = 1500) -> List[Dict]:
        rows = []
        for kappa in concentrations:
            cfg = replace(self.cfg, external_ability=ability,
                          external_concentration=kappa, n_permutations=n_perm)
            adj = Adjudicator(self.sheet, cfg).run(n_permutations=n_perm, verbose=False)
            rows.append(
                {
                    "concentration": kappa,
                    "log10_bf": adj.evidence["log10_bayes_factor"],
                    "posterior_h1": adj.evidence["posterior_h1"],
                    "mc_p": adj.calibration["p_value"],
                    "gates": {k: v["passed"] for k, v in adj.gates.items()},
                    "accepted": adj.accepted,
                    "adjudicated_score": adj.adjudicated_score,
                }
            )
        return rows

    @staticmethod
    def render(rows: List[Dict]) -> str:
        L = ["", "=" * 78,
             "GATE STRESS TEST -- escalating the assertion 'this candidate is able'",
             "=" * 78,
             "Prior concentration = how many prior items of evidence the board claims",
             "to have for the candidate's competence. Higher = a more forceful claim.",
             "",
             f"  {'kappa':>7}{'log10 BF':>10}{'post H1':>10}{'MC p':>9}"
             f"{'BF gate':>9}{'post gate':>11}{'MC gate':>9}{'segments':>10}{'VERDICT':>12}"]
        for r in rows:
            g = r["gates"]
            L.append(
                f"  {r['concentration']:>7.0f}{r['log10_bf']:>10.2f}{r['posterior_h1']:>10.4f}"
                f"{r['mc_p']:>9.4f}"
                f"{('PASS' if g['bayes_factor'] else 'fail'):>9}"
                f"{('PASS' if g['posterior'] else 'fail'):>11}"
                f"{('PASS' if g['monte_carlo'] else 'fail'):>9}"
                f"{('PASS' if g['segment_coherence'] else 'fail'):>10}"
                f"{('ACCEPTED' if r['accepted'] else 'rejected'):>12}"
            )
        bought = [r for r in rows if r["gates"]["bayes_factor"]]
        L.append("")
        if bought:
            L.append(f"  The Bayes-factor gate CAN be bought: it flips at concentration "
                     f"{bought[0]['concentration']:.0f}.")
        else:
            L.append("  The Bayes-factor gate never flips, even at absurd concentration.")
        held = [r for r in rows if not r["gates"]["monte_carlo"]]
        L.append(f"  The Monte-Carlo gate held in {len(held)}/{len(rows)} conditions. It uses NO")
        L.append("  ability model, so it cannot be moved by asserting competence. That is why")
        L.append("  it is the load-bearing gate and must be governed most tightly.")
        return "\n".join(L)


# ==============================================================================
# 11. ENTRY POINT
# ==============================================================================
#
# No case data is held in this module. The detector is general; the sheet under
# examination is supplied by the caller. `ResponseSheet.from_file` reads either
# of the formats in data/, and `CASE_SHEET` below is only the default argument
# for this file's own demonstration run.

CASE_SHEET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "answers.json")


def load_case_records(path: str = CASE_SHEET) -> List[Dict]:
    """Records for the sheet in data/, for the analyses that need key and marks
    separately rather than a ResponseSheet."""
    sheet = ResponseSheet.from_file(path)
    return [{"question": i + 1, "correct": k, "student": m}
            for i, (k, m) in enumerate(zip(sheet.key, sheet.marks))]

def demo_planted_shift(cfg: AdjudicationConfig, outdir: str) -> Adjudication:
    """
    POSITIVE CONTROL.

    Takes the SAME answer key, builds a competent candidate (theta = 0.85), then
    physically simulates them skipping one bubble row at question 15. The
    detector must find it. Without a positive control, a null result on the real
    sheet is uninterpretable: a silent detector produces the same output.
    """
    rng = random.Random(cfg.seed + 99)
    key = ResponseSheet.from_file(CASE_SHEET).key
    n = len(key)
    truthful = []
    for k in key:
        truthful.append(k if rng.random() < 0.85 else rng.choice([o for o in "ABCD" if o != k]))
    at = 15  # 0-based: the slip happens entering Q16
    marks = truthful[:at] + [rng.choice(list("ABCD"))] + truthful[at:]
    marks = marks[:n]
    sheet = ResponseSheet(key, tuple(marks), candidate_id="POSITIVE-CONTROL",
                          subject="Mathematics (simulated slip at Q16)")
    return Adjudicator(sheet, replace(cfg, external_ability=0.85)).run(verbose=False)


def main() -> None:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    cfg = AdjudicationConfig()

    print("=" * 78)
    print("OMR ALIGNMENT ADJUDICATOR")
    print("=" * 78)

    sheet = ResponseSheet.from_records(
        load_case_records(), candidate_id="CANDIDATE-001", subject="Mathematics"
    )
    print(f"\nSheet: {sheet.n_questions} questions, {sheet.n_rows} marks, "
          f"raw score {sheet.raw_score()}/{sheet.n_questions}\n")

    # -- Run A: neutral prior on ability -----------------------------------
    print("Run A -- NEUTRAL PRIOR (no external evidence about the candidate):")
    adj = Adjudicator(sheet, cfg).run()
    rep = Reporter(adj)
    text = rep.text_report()
    print("\n" + text)

    with open(os.path.join(here, "case_default_prior.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(here, "case_detail.json"), "w") as f:
        json.dump(rep.to_dict(), f, indent=2, default=str)

    print("\nRendering figures ...")
    for p in Visualizer(adj, figdir).render_all():
        print(f"  wrote {os.path.relpath(p, here)}")

    # -- Run B: the strongest prior the candidate could ask for ------------
    print("\n" + "=" * 78)
    print("Run B -- 'BRILLIANT CANDIDATE' PRIOR (theta = 0.85 from other subjects)")
    print("This is the scenario as reported: a strong student who collapsed in one")
    print("paper. If no re-registration helps even under this generous assumption,")
    print("the conclusion does not depend on how able we believe them to be.")
    print("=" * 78)
    cfg_b = replace(cfg, external_ability=0.85)
    adj_b = Adjudicator(sheet, cfg_b).run(verbose=False)
    text_b = Reporter(adj_b).text_report()
    print(text_b)
    with open(os.path.join(here, "case_high_ability_prior.txt"), "w") as f:
        f.write(text_b)
    Visualizer(adj_b, os.path.join(figdir, "brilliant_prior")).render_all()

    # -- Run C: adversarial stress test ------------------------------------
    print("\nRunning gate stress test (this takes a minute) ...")
    stress = GateStressTest(sheet, cfg).run()
    stress_txt = GateStressTest.render(stress)
    print(stress_txt)
    with open(os.path.join(here, "gate_stress_test.txt"), "w") as f:
        f.write(stress_txt)

    # ---- positive control ------------------------------------------------
    print("\n" + "=" * 78)
    print("POSITIVE CONTROL: same key, competent candidate, one bubble row skipped at Q16")
    print("=" * 78)
    ctrl = demo_planted_shift(cfg, figdir)
    ctext = Reporter(ctrl).text_report()
    print(ctext)
    with open(os.path.join(here, "positive_control.txt"), "w") as f:
        f.write(ctext)
    Visualizer(ctrl, os.path.join(figdir, "positive_control")).render_all()

    # ---- operating characteristics ---------------------------------------
    print("\n" + "=" * 78)
    print("VALIDATION: operating characteristics on synthetic sheets")
    print("=" * 78)
    val = SyntheticValidator(replace(cfg, external_ability=0.85)).run(n_per_cell=60)
    s = val["summary"]
    print(f"\n  False-alarm rate, honest unshifted sheets      : {s['false_alarm_honest']:.4f}"
          f"   (n={s['false_alarm_honest_n']}, exact 95% upper bound "
          f"{s['false_alarm_honest_ci95_upper']:.4f})")
    print(f"  False-alarm rate, streaky/adversarial sheets   : {s['false_alarm_adversarial']:.4f}"
          f"   (n={s['false_alarm_adversarial_n']}, exact 95% upper bound "
          f"{s['false_alarm_adversarial_ci95_upper']:.4f})")
    print("  Detection power:")
    for k, v in s["power"].items():
        print(f"    {k:<24} {v:.3f}")
    print(f"\n  wrote {os.path.relpath(SyntheticValidator.plot(val, figdir), here)}")
    with open(os.path.join(here, "validation.json"), "w") as f:
        json.dump(s, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
