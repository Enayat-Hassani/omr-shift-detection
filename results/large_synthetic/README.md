# Large Synthetic Shift-Detection Benchmark  (results)

**Run**: mode `full`  ·  seed 20260804

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
`20260804 + 100000*(cell+1) + i`; seeds never rely on `hash()`.

## Files

* `summary.txt` / `summary.json`  — headline numbers
* `per_condition.csv`             — one row per (condition x detector x policy)
* `per_sheet.csv`                 — every synthetic sheet judged
* `threshold_sweep.csv`           — policy sweep rows
* `failure_cases.json`            — honest FN / FP / wrong-location cases
* `figures/*.png`                 — 7 figures
