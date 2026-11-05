# Common synthetic-data evaluation

`run.py benchmark` accepts one or more synthetic CSV paths from any generator.
The real path is fixed in `src/evaluation/benchmark.py` to
`data/raw/hmda_2024_delaware.csv`, resolved relative to the project, not the shell.
It compares the 13 shared HMDA fields across all institutions and actions 1/2/3.
For real data only, actions 1/2 map to loanapproved=1 and action 3 maps to 0.
Synthetic CSVs must already contain the shared 13 fields, including loanapproved.
Extra columns are ignored; column order does not matter. Missing required columns
and empty files fail. Numerical string formatting is canonicalized for matching.

```bash
cd /home/corelabtq/Desktop/FYP/hmda_tabddpm
source .venv-ctgan/bin/activate
python run.py benchmark outputs/ctgan/synthetic.csv --out outputs/benchmark_ctgan
python run.py benchmark outputs/tabddpm/synthetic.csv outputs/ctgan/synthetic.csv --out outputs/benchmark_comparison
```

Each input gets a numbered report folder; `comparison.csv` summarizes the inputs.
Existing output folders are protected. Use a new `--out` for a repeated run.
The evaluator trains only small detector classifiers, not data generators.
No new dependencies beyond the existing CTGAN environment are needed.

## Reports

- `report.md`: readable results and interpretation.
- `summary.json`: metrics, group coverage, settings, reference/input paths and SHA-256.
- `marginals.csv`: numerical KS, raw Wasserstein and Wasserstein divided by real IQR;
  missingness differences; category total variation and unseen category frequencies.
- `dependencies.csv`: numerical Spearman gaps, joint categorical TV, and numerical
  KS within institution/approval groups. Joint TV includes marginal differences.
- `conditional_approval.csv`: rates within banks, income groups, and bank/income
  combinations. Quartile boundaries come from the fixed real reference. Sparse
  groups remain visible; summary errors require at least 30 rows in both groups.
- `tails.csv`: P1/P99 exceedance rates and min/P0.1/P1/median/P99/P99.9/max.
  Missing numerical entries are excluded from these rates and assessed separately.
- `detectors.csv`: logistic regression and random forest ROC-AUC per random seed,
  including warnings and train/test sizes.
- `detector_summary.csv`: mean, seed standard deviation and valid run count.

For distribution distances/gaps, lower is better. For detector AUC, around 0.5
means the particular detector did not distinguish sources; near 1 indicates a
clear difference. AUC is not a generator approval-prediction score. A weak detector
cannot prove distribution equality. Convergence warnings are recorded; inspect
them before interpreting logistic results. Undefined metrics remain blank/null,
not zero. No arbitrary overall score or pass/fail thresholds are applied.

## Detector protocol

Default: up to 5,000 records per source, 3 seeds, random forest with 150 trees.
Both detectors see all 13 fields and numerical missing-token indicators.
Numerical signed-log transforms preserve negative/zero inputs; imputation,
scaling and categorical encoding are fitted on detector training folds only.
Canonical identical rows stay together across detector train/test partitions,
including exact matches occurring in both sources. Group partitioning can produce
unequal label shares; counts and proportions are reported. AUC needs both labels.
Degenerate inputs produce an explicit unavailable status rather than a false score.
Each repeat also compares disjoint real samples at the same sample size to show
ordinary sample variation. Group splits prevent shared exact rows crossing folds.

```bash
python run.py benchmark path/to/synthetic.csv --out outputs/quick_check --detector-rows 1000 --repeats 1 --trees 50
python tests/check_benchmark.py
```

The test checks identical inputs, a shuffled column preserving marginals while
breaking dependence, an obvious detector signal, schema rejection and all-missing
numerical fields.

## Interpretation limits

This uses the FULL real reference, which includes generator training records.
It measures similarity to that population, not held-out generator generalization.
The older `run.py evaluate ctgan|tabddpm` still compares to the prepared test split.
Do not compare its metrics directly to these without accounting for that scope.
No institutions are silently removed to match synthetic coverage. A generator
trained on a subset will be penalized for not representing the complete reference.

Exact full-reference matches are not labelled training memorization. Duplicates
are reported for synthetic and real records, including a size-matched real subset.
The suite does not perform privacy attacks, TSTR utility evaluation or model-ranking
preservation. Repeated-seed deviations are not formal confidence intervals.
Rare-tail estimates can be very uncertain. Pairwise checks do not characterize the
entire joint distribution; detectors complement them but do not prove equality.
