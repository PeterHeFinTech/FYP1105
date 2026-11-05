> Model notes from the original layout. For current paths and runnable commands, use [the project README](../README.md). Old command examples below are historical.

# Compatible CTGAN baseline

Uses the official sdv-dev/CTGAN package (https://github.com/sdv-dev/CTGAN), pinned
at 0.13.0. Shares the existing `prepared/` train/test split, 13 output fields,
numeric missing-token indicators, and `hmda.py evaluate` with TabDDPM.
Does not use test data to fit CTGAN. Do not rerun preparation for this comparison.

CTGAN learns its own mixture-based numeric transform from raw training values;
it does not consume TabDDPM's Gaussian quantile coordinates. All finite training
extremes remain. Output clipping to training bounds and monetary-grid rounding
match TabDDPM for comparison. The raw generated values and clipping counts are
saved too, so postprocessing cannot hide out-of-range outputs or loss of diversity.
This comparison uses model-appropriate preprocessing, not identical internal
coordinates. Neither method is guaranteed to reproduce rare cases or enforce
cross-column business rules.

## On the remote machine

Dependencies are already installed in `.venv-ctgan`, which inherits the working
CUDA PyTorch from conda base without changing that environment.

```bash
cd /home/corelabtq/Desktop/FYP/hmda_tabddpm
source .venv-ctgan/bin/activate
python -u hmda_ctgan.py train --device cuda --epochs 300
python hmda_ctgan.py sample --device cuda --rows 10000
python hmda.py evaluate --data prepared --synthetic synthetic_ctgan.csv --out evaluation_ctgan
```

Training saves `runs/ctgan_first/checkpoint.pkl`, `config.json`, and `loss.csv`.
Sampling saves `synthetic_ctgan.csv`, `synthetic_ctgan.raw.csv`, and
`synthetic_ctgan.diagnostics.json`. Evaluate only the decoded CSV; raw includes
internal missing-status columns. Existing outputs cause an error rather than
being overwritten. To rerun, choose a new `--out` for training and pass that run's
`--checkpoint` to sampling, with new sample/evaluation output paths.

300 epochs is a starting budget, not a tuned setting. CTGAN epochs are not
TabDDPM optimizer iterations; do not directly compare their loss values or call
these equal-compute experiments. Check marginal and joint fidelity, tails and
duplicates on the same test data. Add validation before systematic tuning.
No resume support: the checkpoint and loss CSV are saved after fit completes.
Checkpoints are pickle files and must only be loaded from trusted sources.

## Recreate dependencies if needed

From conda base with the working CUDA torch:

```bash
python -m venv --system-site-packages .venv-ctgan
.venv-ctgan/bin/python -m pip install -r requirements-ctgan.txt
```

## Functional check

```bash
python check_ctgan.py --device cuda
```

Runs one epoch on fabricated toy records in a temporary folder, checks numeric
and missing-token conversion, checkpoint save/reload, deterministic seeded
sampling, shared evaluation and overwrite protection. This is a functionality
check, not evidence of HMDA model quality. No full HMDA training is started by setup.
