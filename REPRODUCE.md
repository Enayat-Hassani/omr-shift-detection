# Reproducing the published results

Every table and figure in the documents comes from these commands, in this
order, from the repository root. All output is written to `results/`, which is
committed, so this is a check. Reading the documents needs none of it.

```bash
python3 omr_shift.py                              # case reports, positive control, stress test, validation, figures
python3 benchmark/omrbench.py --n 12              # comparison table in REPORT.md section 4, and the README figure
python3 benchmark/mechanisms.py --n 14            # per-mechanism table in REPORT.md section 4.3
python3 benchmark/verify_corpus.py                # controls; exits non-zero on failure
python3 benchmark/policy_profiles.py              # profile table and figure in REPORT.md section 6.1, and A6
python3 benchmark/cohort_screen.py                # cohort false discovery screen in REPORT.md sections 6.2 and 6.3
python3 benchmark/figures.py                      # README comparison figure, redrawn from the committed corpus 2 summary
python3 analysis/error_families.py                # error family table in CASE_REPORT.md section 5
python3 analysis/latent_structure.py              # structure tests in CASE_REPORT.md sections 4 and 8
```

The second corpus is separated because it is the long one: about 1.6 hours on
eight cores.

```bash
python3 benchmark/large_synthetic.py --full --jobs 8
```

This reproduces REPORT.md section 6.5, at the shipped default. `--level`
overrides the acceptance level for the metrics arm and leaves every other
threshold at its shipped value, so runs at different levels differ in exactly one
quantity and generate identical sheets from seed 20260804. Nothing published is
built from a non-default level; the level comparison in section 6.4 comes from
the profile study, which is smaller and made for that purpose.

Use `--quick` for an eight-second smoke run, or `--n` to set sheets per cell.
Output goes to `results/large_synthetic/` unless `--out` is given.

Two further scripts reproduce the negative results recorded in
[ASSUMPTIONS.md](ASSUMPTIONS.md):

```bash
python3 extensions/irt_model.py --n 40
python3 extensions/weighted_scan.py --n 220 --perm 1200
```

`policy_profiles.py` takes `--clean`, `--shift` and `--certify` to vary the three
sample sizes; the committed output uses the defaults.

## Controls

`verify_corpus.py` takes about a minute and exits non-zero on failure. It checks
four properties: the generator plants the error its label records, a deliberately
permissive rule produces false positives on the same error-free sheets the
detector clears, a rule accepting nothing scores zero, and ten known-answer
sheets adjudicate as stated.

The controls apply to the synthetic corpora. No confirmed historical case is
involved. README.md describes the validation methodology in full.

## Determinism

Results are reproducible. Seeding is fixed in `AdjudicationConfig.seed`, and the
benchmark derives per-generator seeds with `zlib.crc32`. Python's `hash` is
randomised per process and cannot be used for this. Running any command twice produces
identical output, which was checked from a clean clone.

## What is not reproducible from here

`CASE_REPORT.md` sections 7 and 8 use the candidate's marks for all 150 rows of
the sheet, transcribed from the published image. Rows 1 to 46 agree exactly with
the marking record; rows 47 to 150 have no independent check and are kept out of
the repository, since publishing them as verified would misstate what they are.
They are available on
request.
