# Data

Two sources are used here: one real answer sheet, and generated sheets with
constructed ground truth. No generated sheet is stored in this directory. They
are produced on demand by the scripts in `benchmark/`, and the outputs those runs
produce are committed under `results/`.

## The real sheet

`answers.csv` and `answers.json` hold 46 mathematics questions: the correct
answer and the mark the candidate made.

| column | meaning |
|---|---|
| `question` | question number, 1 to 46 |
| `correct_answer` | the answer key |
| `student_answer` | the mark on the sheet |

These files are the only copy. The detector holds no case data: `omr_shift.py`
reads a sheet through `ResponseSheet.from_file`, which accepts either format, and
the module is general to any sheet in that shape. The two formats can drift from
each other, so `benchmark/verify_corpus.py` checks they agree and exits non-zero
if they do not.

This is the case described in [CASE_REPORT.md](../CASE_REPORT.md), raised through
BANFES, the assessment system built by Education Bridge for Afghanistan. The
answer sheet and the marking record were published by the organisers.

## What the analysis covers, and what bounds it

The examination had 150 questions across several subjects. The published marking
record showed the answer key for the mathematics section, so the analysis covers
those questions and no others.

Two limits follow, and both matter to how the result should be read.

The claim that prompted the case, that this candidate performed well in every
other subject, cannot be checked without the remaining answer keys. It is the
single most useful piece of missing information.

A registration error whose evidence lies mainly outside this section, or one
affecting the sheet as a whole, is harder to detect from a 46 question window
than from the full 150. The analysis finds no statistical support for a
displacement within the section it can see. That is not the same as ruling out an
error elsewhere on the sheet, or one of a kind the tests in
[CASE_REPORT.md](../CASE_REPORT.md) section 5 do not cover.

The candidate's marks for all 150 rows are legible on the published sheet. Rows 1
to 46 were checked against the marking record and agree exactly. Rows 47 to 150
were transcribed from the image without an independent check, so they are kept
out of the repository, since publishing them as verified would misstate what they
are. They are available on request for anyone re-running the provenance checks.

## Generated data

Every sheet carries a constructed ground truth: the true registration is recorded
when the sheet is built. All randomness is seeded, and per-sheet seeds derive from
`zlib.crc32`, so repeated runs produce identical corpora.

### What each corpus is for, and why it is that size

The two corpora answer different questions, and each sample size is set by the
precision its own claim needs rather than by convenience.

| | Corpus 1 | Corpus 2 |
|---|---|---|
| Question | which model to adopt | does the choice survive conditions it was not chosen under |
| Detectors | five | five |
| Sheets | 480 | 9,984 |
| Error-free sheets | 120 | 3,840 |
| Mechanisms | 9 | 18 |
| Sheet lengths | fixed | 46, 90, 100, 150 |
| Options per question | four | four and five |

Corpus 1 selects the model. It reports every figure per generator and never
pools them, because a pooled average hides the generator under which a method
fails. 12 error-free sheets per cell is what makes each generator separately
reportable, and it is a weak bound on its own: 0 of 12 bounds that generator
below 22.1% at 95%. Corpus 1 is therefore sized to rank the models, not to bound
the winner's false positive rate. Corpus 2 does that.

Corpus 2 tests whether that selection holds under conditions corpus 1 does not
contain: scanner artefacts, adversarially filled sheets, four sheet lengths and
five-option papers. It scores all five detectors rather than the winner alone,
because a validation corpus that can only score the chosen model cannot show
that the choice was wrong. 8 sheets per cell across a 1,248-cell grid yields
3,840 error-free sheets, and that count is what bounds the false positive claim
at 0.07%. The bound depends on the error-free count and nothing else, so raising
it is the only way to tighten the claim.

The threshold sweep runs on a corpus disjoint from the metrics arm, over one
mechanism per structural family, and varies the acceptance level and nothing
else. Its rows are the three profiles a board can select. Its purpose is the
recovery comparison across mechanisms the profile study in
[REPORT.md](../REPORT.md) section 6.1 does not cover; the false positive bound in
6.1 is tighter and remains the one to quote.

### First corpus

`benchmark/omrbench.py`. Ten candidate behaviour models against nine error
mechanisms, with five detectors scored on identical sheets. Each cell holds 12
error-free and 36 error sheets. Four options per question, fixed sheet length.
Used for [REPORT.md](../REPORT.md) sections 4 and 8. Output in
`results/benchmark.txt` and `results/benchmark.json`, with the per-mechanism
breakdown from `benchmark/mechanisms.py` in `results/benchmark_mechanisms.txt`.

### Second corpus

`benchmark/large_synthetic.py`. 9,984 sheets across eleven candidate behaviour
models and eighteen error mechanisms, with three detectors scored on identical
sheets. Covers conditions the first corpus does not:

| Group | Contents |
|---|---|
| Scanner artefacts | blank rows, deleted rows, misreads, faint marks, double marks |
| Adversarial sheets | long option runs, a favoured option, an easy block, key leakage, cyclic patterns |
| Sheet geometry | exam lengths from short to long; four-option and five-option papers |
| Displacement | single and double row skips, boundary slips, two separate shifts on one sheet, self-corrected shifts, deferred questions |

Used for [REPORT.md](../REPORT.md) section 6.4. Output in
`results/large_synthetic/` at the current default acceptance level, and in
`results/large_synthetic_conservative/` at level 0.001. Both runs use seed
20260804 and generate identical sheets, so the two differ only in the acceptance
level.

Each run writes `summary.txt` with the pooled counts, `per_condition.csv` with one
row per condition, `per_sheet.csv` with one row per sheet, `threshold_sweep.csv`
for the five policy rows, and `failure_cases.json` with a capped sample of
sheets the detector handled incorrectly.

`per_sheet.csv` records the ground-truth status and adjudication outcome for
every evaluated sheet, tracking classification (`has_shift`, `accepted`, `TP`,
`TP_wrongloc`, `FP`, `TN`, `FN`), reported confidence, and all score variants
(`naive`, `true`, `final`). Summary figures in `summary.txt` can be verified
directly against this record without a full benchmark re-run. Filtering the file
for `detector == 'gated'`, `dataset == 'metrics'`, and `outcome == 'FP'` returns
zero, confirming the zero-false-positive claim in `REPORT.md` (Section 6.4).

### Profile and control sheets

`benchmark/policy_profiles.py` measures the three acceptance profiles. It uses
sheet length 20 and single-row skips only, deliberately: the detection floor is
an absolute number of correct marks, so it consumes the largest share of a short
paper, and a single skip is the mechanism the model represents most directly.
The figures therefore describe the hard case for the floor and the easy case for
the mechanism, and section 6.1 says so.

Three arms, three sample sizes, each set by its own claim. 300 error-free sheets
for the calibration arm, enough to put a 95% interval of about ±0.03 around the
largest rate it measures. 150 skips for the profile arm, 30 per ability level,
enough for the shape of the recovery curve without claiming precision at any
single point. 1,500 error-free sheets for the certification arm, which is the
binding constraint: the bound depends only on that count, and 0 of 1,500 gives
0.20% at 95%. 3,000 would give 0.10% and 30,000 would give 0.01%. Output in
`results/policy_profiles.txt`.

`benchmark/verify_corpus.py` generates sheets for the controls described in
[README.md](../README.md), including ten known-answer sheets whose correct
adjudication is fixed in the script. Its defaults are 400 sheets for the
construction check and 200 for the controls: the construction check is a
pass/fail assertion on every sheet rather than a rate, so it needs enough sheets
to catch an intermittent generator fault and no more, and the permissive control
must produce at least one false positive on the 200, which it does. It writes no
file and exits non-zero on failure.

## Scoring

The organisers' marking record shows 1 mark for a correct answer and −0.15 for a
wrong one. This code counts correct answers and does not apply the penalty. A
board using it should apply their own scoring rule to the resulting registration.
