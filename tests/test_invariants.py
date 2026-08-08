#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests structural guarantees and safety properties for every sheet.

WHY THIS EXISTS
The system relies on strong theoretical guarantees, such as fair scoring and cheat
prevention. This script tests those claims against randomly generated sheets to prove
they hold true.

HOW IT WORKS
- Validates strict rules on every individual sheet.
- Ignores overall statistics and averages.
- Fails if any sheet violates a promised guarantee.

Usage:
    python3 tests/test_invariants.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest
from dataclasses import replace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from omr_shift import (  # noqa: E402
    AdjudicationConfig, Adjudicator, BandedPairHMM, Policy, ResponseSheet,
    ScoringModel,
)

OPTIONS = "ABCD"
# Small, because these run on every commit. Properties fail on a witness, and a
# witness turns up in tens of sheets or not in thousands.
N_SHEETS = 24
N_PERM = 199


def make_sheet(seed: int, n: int = 20, shift: bool = False) -> ResponseSheet:
    rng = random.Random(seed)
    key = [rng.choice(OPTIONS) for _ in range(n)]
    theta = rng.choice([0.35, 0.55, 0.75, 0.95])
    marks = [k if rng.random() < theta
             else rng.choice([o for o in OPTIONS if o != k]) for k in key]
    if shift:
        q = rng.randint(2, n - 4)
        marks = marks[:q] + [None] + marks[q:n - 1]
    return ResponseSheet(tuple(key), tuple(marks), candidate_id=f"t{seed}")


def adjudicate(sheet: ResponseSheet, cfg: AdjudicationConfig = None):
    cfg = cfg or AdjudicationConfig()
    return Adjudicator(sheet, cfg).run(n_permutations=N_PERM, verbose=False,
                                       early_stop=True)


class TestPathStructure(unittest.TestCase):
    """A2 and A3: the registration is a strictly increasing partial injection.

    Claimed to hold by construction, because a monotone lattice path is exactly
    such a map. If it can be broken, the claim is wrong and the axioms are
    conventions rather than guarantees.
    """

    def test_alignment_is_strictly_increasing_and_injective(self):
        cfg = AdjudicationConfig()
        for seed in range(N_SHEETS * 3):
            sheet = make_sheet(seed, shift=(seed % 2 == 0))
            model = ScoringModel(sheet, cfg)
            theta = cfg.operating_theta(model.n_opts)
            pairs = BandedPairHMM(model, theta).viterbi().pairs

            questions = sorted(pairs)
            rows = [pairs[q] for q in questions]
            with self.subTest(seed=seed):
                self.assertEqual(len(set(rows)), len(rows),
                                 "a row was read twice (A3)")
                self.assertEqual(rows, sorted(rows),
                                 "the question to row map decreased (A2)")

    def test_displacement_stays_within_the_band(self):
        cfg = AdjudicationConfig()
        for seed in range(N_SHEETS):
            sheet = make_sheet(seed, shift=True)
            model = ScoringModel(sheet, cfg)
            pairs = BandedPairHMM(model, cfg.operating_theta(model.n_opts)).viterbi().pairs
            for q, r in pairs.items():
                with self.subTest(seed=seed, q=q):
                    self.assertLessEqual(abs(r - q), cfg.max_displacement,
                                         "offset left the band (A6)")


class TestRelabelling(unittest.TestCase):
    """The option alphabet carries no meaning.

    Permuting the labels in the key and the marks together describes the same
    examination with different ink. Any dependence on which letter is which
    would be a latent preference for particular answers, which is indefensible
    on a real sheet.
    """

    def test_the_deterministic_verdict_is_exactly_invariant(self):
        """Evidence, alignment and score must be identical to the last bit."""
        perm = {"A": "C", "B": "D", "C": "A", "D": "B"}
        for seed in range(N_SHEETS):
            original = make_sheet(seed, shift=(seed % 2 == 0))
            relabelled = ResponseSheet(
                tuple(perm[k] for k in original.key),
                tuple(None if m is None else perm[m] for m in original.marks),
                candidate_id=original.candidate_id,
            )
            a, b = adjudicate(original), adjudicate(relabelled)
            with self.subTest(seed=seed):
                self.assertEqual(a.adjudicated_score, b.adjudicated_score)
                self.assertEqual(a.raw_score, b.raw_score)
                self.assertAlmostEqual(a.evidence["log10_bayes_factor"],
                                       b.evidence["log10_bayes_factor"], places=9)
                self.assertEqual(a.alignment.pairs, b.alignment.pairs)

    def test_the_permutation_p_value_is_invariant_only_in_distribution(self):
        """And this is the reason the exact test above stops where it does.

        The null generators draw symbols, so relabelling the alphabet reshuffles
        which draws the random stream produces. The p-value is therefore a
        different estimate of the same quantity, not the same number. A sheet
        sitting exactly on the acceptance level can cross it under relabelling.
        That is a property of estimating a p-value by simulation, not an
        asymmetry in the model.

        Early stopping is off here. With it on, a plainly error-free sheet
        settles in a handful of draws and its p-value is 3/4 or 4/4 -- numbers
        that carry the decision correctly but estimate nothing precisely, so
        comparing them measures the stopping rule rather than the invariance.
        """
        perm = {"A": "C", "B": "D", "C": "A", "D": "B"}
        full = lambda sh: Adjudicator(sh, AdjudicationConfig()).run(
            n_permutations=N_PERM, verbose=False, early_stop=False)
        for seed in range(N_SHEETS):
            original = make_sheet(seed, shift=(seed % 2 == 0))
            relabelled = ResponseSheet(
                tuple(perm[k] for k in original.key),
                tuple(None if m is None else perm[m] for m in original.marks),
                candidate_id=original.candidate_id,
            )
            pa = full(original).calibration["p_value"]
            pb = full(relabelled).calibration["p_value"]
            # Two independent binomial estimates of the same p, plus the
            # resolution floor. Derived rather than chosen: a fixed tolerance
            # would be tuned to whatever these particular sheets happened to
            # produce, and would not travel to a different draw count.
            se = math.sqrt(pa * (1 - pa) / N_PERM + pb * (1 - pb) / N_PERM)
            tolerance = 4 * se + 2.0 / (N_PERM + 1)
            with self.subTest(seed=seed):
                self.assertLess(abs(pa - pb), tolerance,
                                f"p-values {pa} and {pb} differ by more than "
                                f"Monte Carlo error allows")


class TestScoreBounds(unittest.TestCase):
    """What the award rule may and may not do to a candidate's score."""

    def test_a_rejected_sheet_is_scored_exactly_as_submitted(self):
        """The strongest safety property in the system. If the sheet is not
        accepted, nothing about it changes -- no partial credit, no rounding,
        no silent re-registration of a single question."""
        for seed in range(N_SHEETS * 2):
            adj = adjudicate(make_sheet(seed, shift=(seed % 3 == 0)))
            if not adj.accepted:
                with self.subTest(seed=seed):
                    self.assertEqual(adj.adjudicated_score, adj.raw_score)

    def test_the_score_stays_inside_the_paper(self):
        for seed in range(N_SHEETS):
            adj = adjudicate(make_sheet(seed, shift=True))
            with self.subTest(seed=seed):
                self.assertGreaterEqual(adj.adjudicated_score, 0)
                self.assertLessEqual(adj.adjudicated_score, adj.sheet.n_questions)

    def test_the_marks_themselves_are_never_altered(self):
        """A1. The system re-indexes; it does not edit what the candidate wrote."""
        for seed in range(N_SHEETS):
            sheet = make_sheet(seed, shift=True)
            before = tuple(sheet.marks)
            adj = adjudicate(sheet)
            with self.subTest(seed=seed):
                self.assertEqual(tuple(adj.sheet.marks), before)

    def test_an_awarded_registration_reads_each_row_once(self):
        """A3 again, but on what was actually awarded rather than on the path."""
        for seed in range(N_SHEETS * 2):
            adj = adjudicate(make_sheet(seed, shift=True))
            if adj.accepted:
                rows = [r for r in adj.awarded_map.values() if r is not None]
                with self.subTest(seed=seed):
                    self.assertEqual(len(set(rows)), len(rows))


class TestProfileOrdering(unittest.TestCase):
    """Looser profiles must not reject what tighter ones accept.

    The profiles differ only in the level the Monte Carlo gate is thresholded
    at, so the accepted sets should nest. This is not quite guaranteed: the
    draw count is derived from the level, so the three profiles estimate the
    p-value at different resolutions from different random streams, and a sheet
    sitting exactly on a boundary could in principle cross it. The test records
    any such sheet rather than asserting it cannot exist, because if one does
    exist the profile table needs a caveat, not a patch.
    """

    def test_acceptance_nests_across_profiles(self):
        order = [Policy.CONSERVATIVE, Policy.BALANCED, Policy.SENSITIVE]
        anomalies = []
        for seed in range(N_SHEETS):
            sheet = make_sheet(seed, shift=True)
            verdicts = []
            for pol in order:
                cfg = replace(AdjudicationConfig(),
                              permutation_alpha=pol.alpha,
                              n_permutations=pol.n_permutations)
                adj = Adjudicator(sheet, cfg).run(
                    n_permutations=pol.n_permutations, verbose=False,
                    early_stop=True)
                verdicts.append(adj.accepted)
            for i in range(len(order) - 1):
                if verdicts[i] and not verdicts[i + 1]:
                    anomalies.append((seed, order[i].label,
                                      order[i + 1].label))
        self.assertEqual(anomalies, [],
                         f"a tighter profile accepted what a looser one "
                         f"rejected: {anomalies}")


class TestMutationProbes(unittest.TestCase):
    """Do the tests above actually bite?

    A suite that passes against broken code is worse than no suite, because it
    converts an unknown into a false assurance. Each probe breaks one thing on
    purpose and asserts that something notices.
    """

    def test_an_inconsistent_relabelling_is_detected(self):
        """Relabelling only the marks, not the key, destroys the sheet. If the
        invariance test cannot tell that apart from a consistent relabelling,
        it is not testing anything."""
        perm = {"A": "C", "B": "D", "C": "A", "D": "B"}
        original = make_sheet(1, shift=True)
        broken = ResponseSheet(
            tuple(original.key),
            tuple(None if m is None else perm[m] for m in original.marks),
            candidate_id=original.candidate_id,
        )
        a, b = adjudicate(original), adjudicate(broken)
        self.assertNotAlmostEqual(a.evidence["log10_bayes_factor"],
                                  b.evidence["log10_bayes_factor"], places=6)

    def test_a_planted_shift_is_visible_to_the_evidence(self):
        """A clean sheet from a strong candidate, and the same sheet with a row
        skipped, must not produce the same Bayes factor. If they do, the
        evidence term is not responding to displacement at all."""
        rng = random.Random(99)
        key = [rng.choice(OPTIONS) for _ in range(20)]
        marks = [k if rng.random() < 0.9
                 else rng.choice([o for o in OPTIONS if o != k]) for k in key]
        clean = ResponseSheet(tuple(key), tuple(marks))
        shifted = ResponseSheet(tuple(key),
                                tuple(marks[:5] + [None] + marks[5:19]))
        bf_clean = adjudicate(clean).evidence["log10_bayes_factor"]
        bf_shift = adjudicate(shifted).evidence["log10_bayes_factor"]
        self.assertGreater(bf_shift, bf_clean + 1.0,
                           "a planted shift moved the evidence by less than one "
                           "order of magnitude")


if __name__ == "__main__":
    unittest.main(verbosity=2)
