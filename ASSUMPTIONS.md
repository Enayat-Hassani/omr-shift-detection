# Assumptions tested

Six assumptions made during the design of the detector were stated in advance and then measured against benchmark data. None was supported as stated. Each entry gives the assumption, the test, the result, and the effect on the project.

---

## A1. Candidate ability can be estimated from the sheet under dispute

**Assumption.** The response model requires a per-item probability of a correct answer. The natural estimate is the one that best explains the marks on the sheet being examined.

**Test.** Ability was fitted to disputed sheets by maximum likelihood and the resulting log-likelihood ratio examined across the admissible range.

**Result.** Not supported. The fit drove the estimate to $0.23$, below the chance level of $0.25$ for a four-option paper. Below chance the log-likelihood ratio inverts: a matching answer becomes evidence against alignment, and the break-even quantity turns negative. A high fitted ability is also the direct product of a successful re-registration, so the estimate and the conclusion are not independent.

**Effect.** Ability is floored at chance ($\theta \ge 0.25$) and taken from evidence external to the sheet, normally the candidate's other subjects. It is never fitted to the paper under appeal.

---

## A2. The acceptance criteria resist an asserted candidate ability

**Assumption.** Ability enters the detector from outside and cannot be checked against the sheet. The acceptance criteria were assumed to constrain how far an asserted ability could move the outcome.

**Test.** Two measurements were taken. First, claimed prior evidence of candidate ability was escalated across eight levels up to 5,000 items, observing each criterion separately. Second, the benchmark was re-run with true generating ability withheld and replaced by a neutral prior and by a wrong value.

**Result.** Not supported for one criterion, supported for another. The evidence ratio criterion is movable by assertion: it passes at a claim of 1,000 prior items of evidence. The Monte Carlo criterion did not move across any of the eight levels because it uses no ability model. Withholding true ability cost about 4 points of detection ($0.67$ against $0.63$) and cost nothing in false positives. A wrong ability cost substantially more ($0.27$).

**Effect.** The Monte Carlo criterion is treated as the binding one. Reported detection figures are read as an upper bound, because the benchmark supplies the detector with the true generating ability.

---

## A3. Score gain is a suitable test statistic

**Assumption.** A genuine displacement returns lost marks, so the number of marks gained under the best alignment should identify it.

**Test.** Four candidate statistics were run against a planted displacement and a Monte Carlo null on the same sheets.

| Statistic | p-value on a genuine displacement |
|---|---|
| Score gain | 0.143 |
| Viterbi score gain | 0.018 |
| Forward evidence ratio | 0.025 |
| Coherence scan | 0.0003 |

**Result.** Not supported. Score gain reported a real displacement as a $14\%$ chance finding ($p = 0.143$). The discriminating feature is the contiguity of the gained marks, not their total number.

**Effect.** The binding criterion is the coherence scan, which measures the most surprising contiguous block of correct answers at any non-zero displacement. Score gain is not used in any acceptance decision.

---

## A4. Detection depends on the candidate response model

**Assumption.** The detector contains an explicit model of how candidates answer. Making that model more realistic was expected to improve detection, and an unrealistic model was expected to inflate the false positive rate.

**Test.** Three separate refinements were measured against the simpler model on identical sheets:
1. A candidate model with correlated question difficulty, an attractive wrong option taking $65\%$ of errors, and ability declining across the paper.
2. Three-parameter item response theory emissions with difficulties calibrated from a 400-sheet cohort by joint maximum likelihood.
3. The same difficulty information moved into the coherence statistic in place of emissions, compared over 220 sheets with an exact McNemar test.

**Result.** Not supported in any of the three. The false positive rate did not move under the realistic candidate model ($0.000$ in both), and detection fell from $0.975$ to $0.887$. Item response theory emissions gave $26$ of $40$ detected against $25$ of $40$ without, despite calibration recovering true item difficulty at $r = +0.992$. Moving difficulty into the coherence statistic gave $0.768$ against $0.773$ ($4$ discordant pairs against $5$, $p = 1.000$).

**Effect.** The detector retains the single-ability response model. At a displaced position the marks bear no relation to the questions they are compared against, so the statistic depends on chance alignment and very little on candidate or question models. Raising recovery requires information not present in the answer sequence; a more complex model of that sequence does not supply it.

---

## A5. Mechanisms outside the model family cannot be handled

**Assumption.** Two mechanisms were expected to be undetectable or unrepresentable: a displacement starting in the first few questions (leaving no aligned block before it) and a question deferred and answered last (violating monotone ordering).

**Test.** Both were measured on the mechanisms designed to produce them, `early_full_shift` (shift starting at question 3) and `deferred_question`.

**Result.** Not supported as stated. A displacement starting at question 3 is detected at $0.971$ with zero localization error, leaving an uncorrected loss of $2.21$ marks against $19.76$ without correction. The coherence statistic requires a contiguous correct block after the change point and does not require one before it. The deferred question cannot be represented exactly, but the best representable alignment treats it as unanswered and recovers the remainder, leaving an uncorrected loss of $3.77$ marks against $13.29$ ($\approx 72\%$ recovery), with a median localization error of one question.

**Effect.** Both mechanisms are retained in the benchmark suite and reported. Early displacement is the most costly mechanism in the suite ($\approx 20$ marks) and is handled by the detector. The single deferred question is the only part that remains unrecoverable.

---

## A6. The configured minimum segment length sets the detection floor

**Assumption.** A parameter specifying the shortest displaced block considered evidential (set to 5) was assumed to determine the smallest displacement the detector could accept.

**Test.** Every gate outcome was recorded across 150 genuine single-row skips at sheet length 20 and ability $0.55$ to $0.95$, sweeping the acceptance level across two orders of magnitude. Separately, reported $p$-values were compared against nominal levels on 300 error-free sheets.

**Result.** Not supported. The configured value decided no case. The permutation null sets the floor well above 5: the fewest correct marks in an accepted block was $11$ at the tightest acceptance level and $8$ at the loosest. The reported $p$-value is a conservative bound: it falls below nominal $0.5$ in $8.3\%$ of error-free sheets, below $0.25$ in $2.7\%$, and below $0.1$ in $0.7\%$. The binding null in $185$ of $300$ cases was the one preserving candidate answer runs.

**Effect.** The acceptance profile is the only live control and is exposed as three named profiles. Minimum segment length is no longer offered as a policy input. The detection floor is published as a measured capability ($8$ to $11$ correct marks in a contiguous block on a 20-question sheet, and $10$ at the default level).

---

## Conclusions supported by these measurements

* **Permissive scoring is exploitable.** Longest common subsequence scoring awards a random sheet $27.3$ of $46$ marks (chance baseline $11.5$), averaging $4.26$ unearned marks per sheet across the benchmark.
* **Contiguity discriminates and totals do not.** Mark contiguity identifies shifts ($p = 0.0003$) while raw total gains do not ($p = 0.143$).
* **The Monte Carlo criterion is the binding one.** It did not move under any asserted ability or candidate model variation.
* **Ordering and injectivity hold by construction.** Path constraints enforce monotonicity and injectivity in the searched path set.
* **Detection depends strongly on candidate ability.** Recovery reaches $0.80$ for a strong candidate ($\theta \ge 0.85$) with a one-row shift, $0.20$ for a middling candidate, and $0.017$ for a middling candidate with a two-row shift.
* **Single misplaced answers are correctly ignored.** Worth 1 mark, where correcting causes more harm than leaving alone. Fire rate is $0.000$.

---

## Assumptions carried forward untested

* **The base rate of registration errors, taken as 1.8%.** Measured by Skiena and Sumazin across 101,265 papers, not measured here. Sets the bar a correction must clear.
* **Maximum displacement of three rows.** An operational assumption not calibrated against confirmed historical cases.
* **The relative frequency of each error mechanism.** Frequencies of individual mechanisms are unmeasured; per-mechanism results are reported separately.
* **Clean digitization.** Initial corpus assumes correct mark reading. Scanner artifacts (faint bubbles, erasures, double marks) are evaluated separately on the second corpus.
* **Cohort-scale behavior.** Criteria apply to individual sheets. Cohort screening requires false discovery rate control, which is not implemented.
