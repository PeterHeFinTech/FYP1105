# FYP1105

HMDA synthetic data generation with TabDDPM and CTGAN.

## Quick start

Run synthetic generation from a CSV:

```bash
python -m hmda_synth --input hmda.csv --output hmda_synth.csv --model ctgan --samples 1000
python -m hmda_synth --input hmda.csv --output hmda_synth.csv --model tabddpm --samples 1000
```

If CTGAN/TabDDPM packages are unavailable, the generator falls back to bootstrap-based sampling so the pipeline still runs.
