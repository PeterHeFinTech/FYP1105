# Synthetic HMDA evaluation
Input: `/home/corelabtq/Desktop/FYP/hmda_tabddpm/outputs/ctgan/synthetic.csv`
Reference: `/home/corelabtq/Desktop/FYP/hmda_tabddpm/data/raw/hmda_2024_delaware.csv`
**Full-reference similarity: the reference includes generator training records.**
Real rows: 33,896; synthetic rows: 10,000.
Approval-rate gap: 2.20 percentage points.
Synthetic duplicates: 0.02%.
## Detector AUC (0.5: indistinguishable to this detector; 1: distinguishable)
| Comparison | Detector | Mean AUC | Seed SD |
|---|---|---:|---:|
| real_vs_real | logistic | 0.5044 | 0.0127 |
| real_vs_real | random_forest | 0.5014 | 0.0054 |
| real_vs_synthetic | logistic | 0.7567 | 0.0053 |
| real_vs_synthetic | random_forest | 0.8193 | 0.0033 |
## Detailed files
- `marginals.csv`: numerical KS/Wasserstein, missingness, category TV.
- `dependencies.csv`: numerical correlations, categorical joint distributions, conditional numerical KS.
- `conditional_approval.csv`: approval within banks, income groups and bank/income groups.
- `tails.csv`: extreme quantiles and tail frequencies.
- `detectors.csv`: every detector run, sample sizes and convergence warnings.
## Limits
- Full-reference similarity, not held-out generator generalization or privacy proof.
- Detector AUC near 0.5 means no distinction found by that detector; near 1 means distinguishable.
- Detector test folds are disjoint by canonical exact-row hash; preprocessing fitted only on detector training folds.
- Repeated-seed standard deviations measure split/sample variability, not formal confidence intervals.
- Group scores omit groups with fewer than min_group valid observations in either dataset; coverage is reported.
- No scalar overall quality score or universal pass threshold; no prediction utility or membership attack is implemented.