#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controls for the synthetic corpora.

A measured false positive rate of zero means nothing on its own. A detector
that is genuinely safe and a counter that is broken produce the same output.
The same holds for ground truth: a generator that silently plants nothing still
produces sheets a detector correctly declines, and the run reports low power
instead of a bug.

These checks are what make the measured rates mean something. None of them
needs a sheet read by hand. That is the point: a corpus of ten thousand sheets
cannot be audited by inspection, only by control.

  A  CONSTRUCTION      Every planted error is independently re-derived from the
                       sheet and compared against its own label. A generator
                       that does not plant what it claims fails here.

  B  POSITIVE CONTROL  A deliberately permissive detector must produce false
                       positives on the SAME error-free sheets the real detector
                       clears. If it does not, the false positive machinery is
                       broken and every zero in the results is meaningless.

  C  NEGATIVE CONTROL  A detector that never accepts must score exactly zero
                       true positives and zero false positives. If it scores
                       anything, the counting is wrong.

  D  KNOWN ANSWER      Sheets whose correct adjudication is fixed by hand, with
                       the expected verdict and score stated in this file.

  E  CASE DATA         The case sheet is stored in two formats, and both are
                       read through the same loader. They can drift from each
                       other; this checks they have not.

Exit status is non-zero if any control fails, so it can be run in CI.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import zlib
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omr_shift import (  # noqa: E402
    AdjudicationConfig,
    Adjudicator,
    Policy,
    ResponseSheet,
    clopper_pearson_upper,
)

SEED = 20260805
OPTIONS = "ABCD"
KEY: Sequence[str] = ("B", "D", "A", "C", "A", "D", "B", "C", "D", "A",
                      "C", "B", "A", "D", "C", "B", "A", "C", "D", "B")


def _rng(tag: str, i: int) -> random.Random:
    return random.Random(zlib.crc32(f"{tag}:{i}".encode()) ^ SEED)


def _answer(theta: float, rng: random.Random) -> List[str]:
    return [k if rng.random() < theta else rng.choice([o for o in OPTIONS if o != k])
            for k in KEY]


def _skip(i: int, theta: float, q: int) -> Tuple[ResponseSheet, List[str]]:
    """Skip one bubble row entering question q. Returns the sheet and the
    underlying truthful answers, so the label can be re-derived."""
    n = len(KEY)
    truth = _answer(theta, _rng("shift", i))
    marks = truth[:q - 1] + [None] + truth[q - 1:n - 1]
    return ResponseSheet(tuple(KEY), tuple(marks), candidate_id=f"shift-{i}"), truth


# ==============================================================================
# A. CONSTRUCTION -- is the planted error the one the label claims?
# ==============================================================================

def check_construction(n: int) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    bad = 0
    for i in range(n):
        theta = (0.55, 0.65, 0.75, 0.85, 0.95)[i % 5]
        q = 2 + (zlib.crc32(f"cp:{i}".encode()) % 15)
        sheet, truth = _skip(i, theta, q)
        m = sheet.marks
        # The label says: row q is blank, and every row after it carries the
        # answer belonging to the question one place earlier. Re-derive both
        # from the sheet without consulting the generator.
        if m[q - 1] is not None:
            bad += 1
            notes.append(f"  sheet {i}: row {q} should be blank, holds {m[q - 1]!r}")
            continue
        misplaced = [j for j in range(q, len(KEY)) if m[j] != truth[j - 1]]
        if misplaced:
            bad += 1
            notes.append(f"  sheet {i}: rows {misplaced[:4]} do not carry the "
                         f"displaced answers the label claims")
            continue
        # And the questions before the change point must be untouched.
        if any(m[j] != truth[j] for j in range(q - 1)):
            bad += 1
            notes.append(f"  sheet {i}: marks before the change point were altered")
    return bad == 0, ([f"  {n - bad}/{n} sheets carry exactly the planted error"]
                      if bad == 0 else notes[:10])


# ==============================================================================
# B & C. CONTROLS -- can the false positive counter fire at all?
# ==============================================================================

class NeverAccept:
    """Negative control. Declines everything, by construction."""

    label = "never-accept"

    def accepts(self, sheet: ResponseSheet) -> bool:
        return False


class AlwaysScan:
    """
    Positive control. The naive rule the report rejects in section 3.1: take the
    best-scoring displacement and accept if it beats the identity at all. It is
    SUPPOSED to false-positive. If it does not, the harness cannot see a false
    positive and no zero it reports means anything.
    """

    label = "best-displacement"

    def accepts(self, sheet: ResponseSheet) -> bool:
        n = len(sheet.key)
        base = sheet.raw_score()
        for d in range(1, 4):
            for sign in (1, -1):
                off = d * sign
                score = sum(1 for q in range(n)
                            if 0 <= q + off < n
                            and sheet.marks[q + off] is not None
                            and sheet.marks[q + off] == sheet.key[q])
                if score > base:
                    return True
        return False


def check_controls(n_clean: int) -> Tuple[bool, List[str]]:
    """Run both controls and the real detector over the SAME error-free sheets."""
    cfg = AdjudicationConfig()
    sheets = []
    for i in range(n_clean):
        theta = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)[i % 7]
        sheets.append(ResponseSheet(tuple(KEY), tuple(_answer(theta, _rng("clean", i))),
                                    candidate_id=f"clean-{i}"))

    never = sum(1 for s in sheets if NeverAccept().accepts(s))
    scan = sum(1 for s in sheets if AlwaysScan().accepts(s))
    gated = sum(1 for s in sheets if Adjudicator(s, cfg).run(verbose=False,
                                                            early_stop=True).accepted)

    notes = [
        f"  error-free sheets                 : {n_clean}",
        f"  never-accept   false positives    : {never}   (must be 0)",
        f"  best-displacement false positives : {scan}   (must be > 0)",
        f"  gated detector false positives    : {gated}   (bound "
        f"{clopper_pearson_upper(gated, n_clean):.4f} at 95%)",
    ]
    ok = (never == 0) and (scan > 0)
    if scan == 0:
        notes.append("  FAIL: a rule that should false-positive did not. The")
        notes.append("        harness cannot observe a false positive, so the")
        notes.append("        gated detector's zero carries no information.")
    if never != 0:
        notes.append("  FAIL: a rule that accepts nothing was counted as accepting.")
    return ok, notes


# ==============================================================================
# D. KNOWN ANSWER -- adjudications fixed by hand
# ==============================================================================

# A blind set of ten sheets. The correct adjudication of each is fixed by
# inspection of the construction, and stated here as the expected result. Two
# carry a genuine displacement; eight carry none. A detector that accepts any of
# the eight, or declines either of the two, has changed behaviour.
#
# (marks, expected accepted, expected adjudicated score, description)
KNOWN: List[Tuple[List[Optional[str]], bool, int, str]] = [
    (list(KEY), False, 20,
     "perfect sheet, nothing to adjudicate"),
    (list(KEY[:3]) + [None] + list(KEY[3:19]), True, 18,
     "row skipped at Q4, 16 displaced marks correct: genuine shift"),
    (list(KEY[:9]) + [None] + list(KEY[9:19]), True, 18,
     "row skipped at Q10, 10 displaced marks correct: genuine shift"),
    (list(KEY[:12]) + [None] + list(KEY[13:]), False, 19,
     "blank at Q13 that does not propagate: not a shift"),
    ([KEY[i] if i != 10 else "D" for i in range(20)], False, 19,
     "single wrong answer at Q11, no displacement"),
    ([KEY[i] if i != 7 else "D" for i in range(20)], False, 19,
     "single wrong answer at Q8, no displacement"),
    (["A", "C", "D", "B", "B", "A", "C", "D", "A", "C",
      "B", "D", "C", "A", "B", "D", "C", "A", "B", "D"], False, 0,
     "no correspondence to the key at any displacement, scores 0"),
    (list(KEY[:19]) + [None], False, 19,
     "blank final row, nothing after it to displace"),
    (list(KEY[:17]) + ["D", "C"] + [KEY[19]], False, 18,
     "adjacent transposition at Q18-Q19: not representable"),
    (list(KEY[:8]) + ["A", "D"] + list(KEY[10:]), False, 18,
     "adjacent transposition at Q9-Q10: not representable"),
]


def check_known() -> Tuple[bool, List[str]]:
    cfg = AdjudicationConfig()
    notes: List[str] = []
    ok = True
    for marks, want_acc, want_score, why in KNOWN:
        sheet = ResponseSheet(tuple(KEY), tuple(marks))
        a = Adjudicator(sheet, cfg).run(verbose=False)
        good = (a.accepted == want_acc) and (a.adjudicated_score == want_score)
        ok &= good
        notes.append(f"  [{'ok  ' if good else 'FAIL'}] {why}")
        if not good:
            notes.append(f"         wanted accepted={want_acc} score={want_score}, "
                         f"got accepted={a.accepted} score={a.adjudicated_score}")
    return ok, notes


# ==============================================================================
# E. CASE DATA -- the embedded copy against the committed files
# ==============================================================================

def check_case_data() -> Tuple[bool, List[str]]:
    """The case sheet exists in two formats, data/answers.csv and
    data/answers.json, and both are read through the same loader. The formats
    can drift from each other; this checks they have not."""
    import csv as _csv
    import json as _json
    from omr_shift import ResponseSheet

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = ResponseSheet.from_file(os.path.join(root, "data", "answers.json"))
    emb = [(i + 1, k, m) for i, (k, m) in enumerate(zip(ref.key, ref.marks))]

    notes = [f"  data/answers.json: {len(emb)} rows (reference)"]
    ok = True
    try:
        with open(os.path.join(root, "data", "answers.csv")) as fh:
            cs = [(int(r["question"]), r["correct_answer"], r["student_answer"])
                  for r in _csv.DictReader(fh)]
        same = cs == emb
        ok &= same
        notes.append(f"  data/answers.csv  matches: {'yes' if same else 'NO'}")
    except Exception as exc:                       # pragma: no cover
        ok = False
        notes.append(f"  data/answers.csv unreadable: {exc}")
    if not ok:
        notes.append("  FAIL: the case data has drifted between its copies.")
    return ok, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", type=int, default=400)
    ap.add_argument("--clean", type=int, default=200)
    args = ap.parse_args()

    w = 74
    print("=" * w)
    print("CORPUS CONTROLS".center(w))
    print("=" * w)
    print("A zero false positive rate means nothing unless the counter can fire.")
    print("These are the checks that make the synthetic results interpretable.")
    print()

    results = []
    for name, (ok, notes) in (
        ("A. CONSTRUCTION   planted error matches its label",
         check_construction(args.construction)),
        ("B/C. CONTROLS     can a false positive be observed?",
         check_controls(args.clean)),
        ("D. KNOWN ANSWER   adjudications fixed by hand",
         check_known()),
        ("E. CASE DATA      the two stored formats agree",
         check_case_data()),
    ):
        print("-" * w)
        print(name)
        print("-" * w)
        for line in notes:
            print(line)
        print(f"  --> {'PASS' if ok else 'FAIL'}")
        print()
        results.append(ok)

    print("=" * w)
    if all(results):
        print("ALL CONTROLS PASS.")
        print("The generator plants the error its label records, a false positive")
        print("is observable in this harness, and the known-answer sheets")
        print("adjudicate as stated. All sheets are synthetic. No confirmed")
        print("historical case is involved.")
    else:
        print("A CONTROL FAILED. The measured rates do not hold until it is fixed.")
    print("=" * w)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
