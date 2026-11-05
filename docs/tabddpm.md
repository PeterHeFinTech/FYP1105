> Model notes from the original layout. For current paths and runnable commands, use [the project README](../README.md). Old command examples below are historical.

# HMDA TabDDPM

Generates loan-application rows, including `loanapproved`, jointly. No approval
classifier is trained. This is an HMDA adaptation of TabDDPM, not a reproduction
of the paper's hyperparameter search or benchmark results.

## Execute on the DGX Spark

The existing conda **base** environment has working CUDA-enabled PyTorch. The
`pytorch_arm` environment was incomplete when inspected. Prefer the existing
working torch build; do not install the original repository's CUDA 11 wheels.

```bash
cd /home/corelabtq/Desktop/FYP/hmda_tabddpm
conda activate base
python check.py

# Once: preprocess the original CSV, preserving it unchanged.
python hmda.py prepare --csv ../hmda_2024_delaware.csv --out prepared

# Train: 10,000 optimizer steps, with 1,000 diffusion timesteps.
python -u hmda.py train --device cuda --data prepared --out runs/first

# Generate 10,000 complete applications.
python hmda.py sample --device cuda --data prepared --checkpoint runs/first/checkpoint.pt --rows 10000 --out synthetic.csv

# Compare generated rows with held-out real records.
python hmda.py evaluate --data prepared --synthetic synthetic.csv --out evaluation
```

For a detached training job (optional, instead of the train command above):

```bash
mkdir -p logs
nohup python -u hmda.py train --device cuda --data prepared --out runs/first > logs/train.log 2>&1 &
tail -f logs/train.log
```

The default 10,000 steps is a starting budget, not a guarantee of convergence.
`--iterations`, `--timesteps`, `--hidden`, `--batch-size`, and `--lr` are configurable.
Output folders must be new: use a new `--out` to repeat an experiment. Checkpoints
are saved every 1,000 optimizer steps and at completion; interrupted runs can be
sampled from the last saved checkpoint. Automatic training resume is not provided.

## Environment

Checked environment: ARM64, NVIDIA GB10, Python 3.13, PyTorch 2.10.0+cu130.
The torch build emits a capability warning for GB10 (12.1); run `check.py` before
training. It verifies this project's mixed diffusion operations on the GPU, not
just CUDA discovery. If ancillary packages are missing, install requirements
in an environment that retains a compatible torch build:

```bash
python -m pip install -r requirements.txt
```

No package installation is required if the commands already work. Never load
untrusted pickle transformers or torch checkpoints.

## Data scope and representation

- Default population: the 10 largest LEIs in this CSV, action_taken in {1,2,3}.
  This is a US regional mortgage subset, not Hong Kong or nationwide data.
- Thirteen output fields: income, loan_amount, property_value, loan_term, lei,
  applicant_age, applicant_sex, debt_to_income_ratio, loan_purpose, loan_type,
  occupancy_type, lien_status, loanapproved.
- `loanapproved=1` for actions 1/2; 0 for action 3. Original action and denial
  reasons are excluded. LEI is a categorical coordinate, not a unique row ID.
- Numerical values: training-fitted Gaussian quantile transform. Missing numeric
  values are median-filled internally and their original token (NA, Exempt,
  empty, etc.) is modelled as an additional categorical status coordinate.
- Age and debt-to-income bins remain categorical; no fake exact age or DTI is
  assigned. Numeric income may legitimately be zero or negative.
- Amount and property-value outputs retain the public $10,000 midpoint grid.
  Other numeric outputs are rounded to whole units. Ranges are restricted to
  observed training ranges. This limits extrapolation.
- Exact duplicate selected rows stay together in an 80/20 train/test split.
  Preprocessing is fitted only on training rows. This does not guarantee
  borrower-level separation because no public stable borrower ID is available.
- The current model generates the joint institution distribution. Direct
  institution conditioning and external marginal constraints are not implemented.

## Model and outputs

The official mixed diffusion core performs Gaussian noise prediction for numerical
features and multinomial diffusion/variational loss for categorical features.
Our time-conditioned MLP uses three hidden linear layers including its input
projection, SiLU activations, and a sinusoidal timestep embedding. Both approval
and institution participate in diffusion. The upstream sampler's dummy `y` is ignored.

`prepared/`: train/test CSVs, training array, metadata and fitted transformer.
`runs/first/`: configuration, loss log and model checkpoint.
`synthetic.csv`: generated rows with the original selected field names.
`evaluation/`: numeric KS/Wasserstein statistics, categorical total variation,
missing-rate gaps, per-institution shares/approval rates, numerical Spearman
correlation differences, exact training-row matches and duplicate rates.

No classifier is used in evaluation. These are initial checks, not a complete
joint-distribution evaluation, privacy guarantee, or evidence of fair lending.
Logical multi-column constraints are not enforced; inspect synthetic records
before downstream use. Smaller groups can be poorly reproduced.
Do not tune repeatedly on the reserved test set; create a separate validation
split before model selection or systematic hyperparameter comparisons.

## Upstream attribution

TabDDPM, Kotelnikov et al., ICML 2023:
https://proceedings.mlr.press/v202/kotelnikov23a.html

Official repository: https://github.com/yandex-research/tab-ddpm
Pinned source commit: b476257dd460b778ba09eb97f7a51d6490fa17f8.
The two files in `vendor/tab_ddpm` are copied without modification from that commit.
`vendor/tab_ddpm/__init__.py` is empty to avoid importing unrelated legacy dependencies.
The upstream MIT license is preserved in `vendor/LICENSE-TabDDPM.md`.

## Rare and extreme observations

No numerical outlier filtering or winsorization is applied. All finite values in
the selected institution population are retained, including zero/negative income.
The default population is the ten largest institutions; this is a scope choice,
not an outlier filter. Use `--top-institutions 507` to include all institutions
in this CSV. Train/test splitting can place a unique extreme in the test set;
it must remain there rather than leak into training.

The quantile transform preserves ranks but compresses tail distances. Its inverse
cannot extrapolate beyond training minima/maxima, so novel held-out extremes
are a known limitation. Float64 inverse conversion reduces tail precision loss;
monetary midpoints are rounded to the nearest valid grid point rather than floored.
Raw train/test CSV values are preserved unchanged. Float32 model coordinates
are still approximate, not a lossless encoding of every original value.

Evaluation additionally writes `tail_coverage.csv` (counts and proportions below
training P1 and above P99, plus values outside the training range) and
`quantiles.csv` (min, P0.1, P1, median, P99, P99.9, max). These compare training,
held-out real and synthetic data separately. A few rare observations cannot
establish a reliable tail distribution; do not interpret matching maxima as
proof of learning realistic rare combinations. No oversampling is applied: naive
oversampling would inflate rare-event probabilities. If tail fidelity is poor,
a signed-log numerical transform or explicitly weighted tail model is a future
experiment, with out-of-range generation validated separately.
