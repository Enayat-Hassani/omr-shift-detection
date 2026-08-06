# Case report: the sheet that prompted this work

Analysis of one candidate's answer sheet. The method and its validation are in
[REPORT.md](REPORT.md), and neither depends on this case.

## 1. Background

The case was raised through BANFES, the assessment system built by Education
Bridge for Afghanistan.
A candidate reported as performing well across all subjects scored 7 out of 46 in
mathematics alone. The suggested explanation was that they had skipped a question
and shifted every subsequent answer down one row.

The examination board set out two outcomes in advance: where evidence of a shift
is strong, correct only the shift and leave genuinely wrong answers wrong; where
evidence is insufficient, offer a re-examination. The method was developed against
that framing.

## 2. The data

46 mathematics questions, four options each. The mathematics section is questions
1 to 46 on a 150-question multi-subject sheet arranged in four columns of 40, so
it covers all of column one plus the first six rows of column two.

Score as marked: **7 out of 46**. Chance performance is about 11.5.

The answer key for the other 104 questions was not provided. The claim that this
candidate performed well elsewhere therefore remains unverified.

## 3. Finding

**No re-registration is supported. The original score stands.**

All five acceptance criteria fail, none of them narrowly.

| Criterion | Value | Threshold |
|---|---|---|
| Evidence ratio (log₁₀) | −1.75 | 2.00 |
| Posterior probability of a shift | 0.0001 | 0.95 |
| Monte Carlo p-value | 0.89 | 0.001 |
| Segment coherence | no coherent displaced segment | all must pass |
| Non-trivial | best reading is the identity | must differ |

The evidence break-even for this configuration is 5.22 questions. A shift would
need to repair at least that many answers before it becomes more probable than
the alternative.

## 4. Supporting analyses

Five independent lines, sharing no machinery.

**Displacement search.** Every displacement from −41 to +41, small and large.
Best result is +12 at 15 correct out of 34, uncorrected p = 0.012, which after
correcting for 77 positions tested is p = 0.59.

**Coherence scan.** The strongest run of correct answers at any non-zero
displacement is questions 31 to 35 at offset +3, four correct out of five. Sheets
known to contain no shift produce evidence at least that strong 89% of the time.

**Answer counts.** This test is invariant under every position error at once, so
it applies to the whole family at once. Re-ordering marks cannot create or destroy an
option, so if these marks were a re-ordering of a competent candidate's answers,
the counts would have to match.

| Option | Key contains | Candidate marked | A competent re-ordering would need |
|---|---|---|---|
| A | 16 | **7** | 14 to 16 |
| B | 10 | 12 | 10 |
| C | 11 | 13 | 11 |
| D | 9 | 14 | 9 to 10 |

Incompatible at any assumed ability of 0.75 or above (p ≤ 0.014). This one test
rules out shifting, reversal, wrong booklet version, block transposition and
section misassignment simultaneously.

**Mutual information.** Tests for any statistical association between marks and
key at any displacement, including partial or noisy relabellings that no
permutation would capture. Nothing at any position, p = 0.78.

**Change-point agreement.** Four detectors sharing no machinery. CUSUM says
question 11, binary segmentation says 12, the coherence scan says 31, the pair
HMM finds none. A spread of 20 questions, and the two that agree share a
principle.

## 5. Other registration errors

Shift is one member of a larger family. Each was tested and corrected for the
size of its own search space.

| Family | Best result | Corrected p |
|---|---|---|
| Displacement, full range | offset +12, 15/34 | 0.59 |
| Symbol relabelling combined with displacement | 12/19 | 0.57 |
| Reversal, sheet scanned upside down | 9/46 | 0.85 |
| Rotation, wrap-around | 17/46 | 0.89 |
| Option relabelling, mis-registered bubble columns | 16/46 | 0.90 |
| Block or column transposition | 17/46 | 0.998 |

## 6. Robustness to the ability assumption

The conclusion does not depend on how able the candidate is assumed to be. The
evidence ratio was computed across the full admissible range, and the Monte Carlo
p-value uses no ability estimate at all.

Pushing the assumption further, the evidence gate can be made to pass by
asserting the equivalent of 5000 prior items of evidence for the candidate's
competence. The Monte Carlo criterion does not move in any condition, and the
overall verdict is unchanged.

The premise is itself testable and does not hold. A candidate performing at 85%
cannot produce 7 out of 46 under any admissible reading of this sheet. The best
available reading yields 14 out of 46 against roughly 11 expected by chance.

## 7. Data provenance

Physical rows 1 to 46 of the answer sheet reproduce the graded spreadsheet
exactly, 46 out of 46. The correct rows were graded.

Sliding the mathematics key across all 105 possible 46-row windows of the full
sheet gives a best match of 20 out of 46 at row 76, corrected p = 0.39. The
mathematics answers are not recorded in another subject's block.

These two checks use the candidate's full 150-answer record, which is not in the
public files. See section 9.

## 8. Answer sequence structure

The mathematics answers compress to a Lempel-Ziv complexity of 13, against a
random-marking baseline of 17.3, p = 0.0004. Equal-length windows elsewhere on
the same sheet give 16 to 18, so the pattern is specific to the mathematics
section.

Lempel-Ziv complexity is invariant under displacement. Moving a sequence does not
compress it. If these marks were displaced genuine answers, their complexity
would match the candidate's genuine answers elsewhere on the sheet. Together with
the answer counts in section 4, this is a second position-invariant test pointing
the same way.

## 9. Limitations of this analysis

The answer keys for the other 104 questions were not available, so the reported
performance in other subjects could not be checked. This is the single most
useful piece of missing information.

Rows 1 to 46 of the sheet are verified against the graded spreadsheet. Rows 47 to
150 were transcribed from a photograph with no independent check, and sections 7
and 8 depend on them. They should be re-run against the board's own digitisation.
Those rows are held separately and are available on request.

## 10. Recommendation

The evidence does not support correcting this sheet. Under the board's own
stated framework, the re-examination is the appropriate route, and the board has
confirmed it is available free and for a single subject.

Two requests would improve on this. The answer keys for the remaining 104
questions, which would settle whether the premise holds. And a re-scan of the
physical sheet, since a material share of suspected shift errors are scanner
misreads of faint or partially erased marks, which are resolved more cheaply and
more certainly by inspection than by inference.
