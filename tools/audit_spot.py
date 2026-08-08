#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spot-check the committed corpus: rebuild sheets from seed and re-adjudicate.

WHAT THIS CHECKS THAT NOTHING ELSE DOES

The rest of the audit machinery checks code against code, or documents against
results. This checks the committed DATA. `per_sheet.csv` records nearly ten
thousand verdicts and every headline figure is an aggregate over it, but the
file is just a table -- nothing about it proves the rows were produced by the
detector in this repository, or that they were produced at all.

So: pick rows at random, rebuild the exact sheet the generator would have made
from the same seed, run the detector on it now, and compare the verdict with
the one on file. A mismatch means one of three things, all worth knowing.

  The CSV is stale.       Someone regenerated the code and not the corpus.
  The run is not reproducible.  A seed does not determine a sheet after all,
                          which would undermine every result in the repository.
  The detector changed.   Expected after a code change, and the reason this
                          reports the fingerprint it was checked against.

WHY SAMPLING RATHER THAN ALL OF IT

Re-adjudicating 9,984 sheets is the benchmark, and it takes ninety minutes. The
point of a spot check is that it is cheap enough to run on a whim. Thirty rows
take under a minute and would catch any systematic staleness; they will not
catch a single altered row, and are not meant to.

    python3 tools/audit_spot.py --n 30
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "benchmark"),
           os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance  # noqa: E402
import large_synthetic as LS  # noqa: E402

DEFAULT_CSV = os.path.join(_ROOT, "results", "large_synthetic", "per_sheet.csv")
DEFAULT_SUMMARY = os.path.join(_ROOT, "results", "large_synthetic",
                               "summary.json")


def _cell_index(cells: List[Dict]) -> Dict[tuple, int]:
    """(options, length, ability, mechanism) -> cell id.

    `per_sheet.csv` records the cell's parameters but not its id, and the id is
    what seeds the sheet. It is recoverable because the id is just the position
    in the enumeration, so rebuilding the grid the same way recovers it.
    """
    return {(("".join(c["options"])), int(c["length"]), float(c["ability"]),
             c["mechanism"]): c["id"] for c in cells}


def rebuild(row: Dict, index: Dict[tuple, int], seed: int):
    """The (key, observed, injection, cell) the generator produced for this row.

    The cell is returned as well as the sheet because the detector is built from
    it, not just seeded by it: the metrics arm supplies an external ability
    derived from the cell, which moves the operating theta and can change the
    verdict. Re-adjudicating with a bare default config compares two different
    detectors and reports the difference as a corpus fault -- which is exactly
    what this tool did on its first run.
    """
    cell_key = ((row["options"]), int(row["length"]), float(row["ability"]),
                row["mechanism"])
    if cell_key not in index:
        raise KeyError(f"no cell matches {cell_key}")
    cid = index[cell_key]
    i = int(row["sheet_i"])
    # cell_rngs builds the whole cell's streams; taking element i is what the
    # benchmark itself does, so the stream is identical rather than merely
    # equivalent.
    rng = LS.cell_rngs(seed, cid, i + 1)[i]
    options = tuple(row["options"])
    cell = {"id": cid, "options": options, "length": int(row["length"]),
            "ability": float(row["ability"]), "mechanism": row["mechanism"]}
    key, observed, inj = LS.sheet_for(rng, options, int(row["length"]),
                                      float(row["ability"]), row["mechanism"])
    return key, observed, inj, cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--n", type=int, default=30, help="rows to re-adjudicate")
    ap.add_argument("--seed", type=int, default=20260804,
                    help="the corpus seed; must match the run that wrote the CSV")
    ap.add_argument("--level", type=float, default=None,
                    help="acceptance level override the run used, if any")
    ap.add_argument("--perm", type=int, default=1100,
                    help="draws per sheet; must match the run that wrote the CSV")
    ap.add_argument("--sample-seed", type=int, default=1,
                    help="which rows to pick, not what they contain")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"no corpus at {args.csv}; run benchmark/large_synthetic.py --full")
        return 1

    with open(args.csv) as fh:
        rows = [r for r in csv.DictReader(fh)
                # Only the reference detector: the baselines are scored on the
                # same sheets, so re-adjudicating them would test the baseline
                # rather than the corpus.
                if r.get("detector") == "gated" and r.get("dataset") == "metrics"]

    if not rows:
        print("no gated metrics rows in the CSV")
        return 1

    cells = LS.build_metrics_conditions(LS.OPTION_SETS)
    index = _cell_index(cells)

    picker = random.Random(args.sample_seed)
    sample = picker.sample(rows, min(args.n, len(rows)))

    print("=" * 78)
    print("CORPUS SPOT CHECK  --  rebuilt from seed and re-adjudicated")
    print("=" * 78)
    print(f"  corpus            : {os.path.relpath(args.csv, _ROOT)}")
    print(f"  rows in file      : {len(rows):,} (gated, metrics)")
    print(f"  re-adjudicated    : {len(sample)}")
    print(f"  fingerprint now   : {provenance.code_fingerprint()}")

    stamped = None
    if os.path.exists(args.summary):
        import json
        with open(args.summary) as fh:
            stamped = json.load(fh).get("_provenance", {}).get("code_fingerprint")
    print(f"  fingerprint on file: {stamped or 'none recorded'}")
    if stamped and stamped != provenance.code_fingerprint():
        print("  NOTE: the corpus was produced by different code from the tree.")
        print("        Disagreement below is expected, and is not evidence that")
        print("        the run was wrong.")
    print("-" * 78)

    agree = 0
    problems = []
    for r in sample:
        try:
            key, observed, inj, cell = rebuild(r, index, args.seed)
        except Exception as exc:
            problems.append((r, f"could not rebuild: {exc}"))
            continue

        # The same factory the benchmark uses, so the detector is the one that
        # produced the row rather than merely a detector of the same class.
        #
        # fast=False because `--full` sets it so. Note this `fast` is not the
        # adjudicator flag of the same name: here it loosens four thresholds for
        # the --quick smoke run, so passing True silently compares against a
        # different, more permissive detector.
        det = LS._cell_detector(cell, "gated", args.perm, fast=False,
                                level=args.level)
        d = det.decide(key, observed)
        was = r["accepted"] == "1"
        if bool(d.accepted) == was:
            agree += 1
        else:
            problems.append((r, f"file says accepted={was}, "
                                f"re-adjudication says {bool(d.accepted)}"))

    for r, why in problems:
        print(f"  MISMATCH  {r['mechanism']:<20} ability={r['ability']:<5} "
              f"len={r['length']:<3} i={r['sheet_i']:<4} {why}")

    print("-" * 78)
    print(f"  agreed: {agree} of {len(sample)}")
    if problems:
        print("\n  A mismatch is not automatically a bug -- check the two")
        print("  fingerprints above first. If they match, the CSV and the code")
        print("  disagree while claiming to be the same run, which is serious.")
    print("=" * 78)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

# probe
