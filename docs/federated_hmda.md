# Two-party HMDA experiment

Run from the repository root on Spark with the project's Python environment. Prepare the aligned partitions from the local HMDA CSV before launching the two-worker run.

```bash
python two_party_trial.py --full-schema --epochs 100 --out outputs/two_party_full
python two_party_quality_candidate.py --full-schema --epochs 100 --out outputs/two_party_quality_candidate
python federated_hmda.py --source outputs/two_party_full --out outputs/federated_reproduction --epochs 100 --rows 10000
for file in heldout.csv party_a_train.csv party_b_train.csv; do
  ln -s "../two_party_full/$file" "outputs/federated_reproduction/$file"
done
python evaluate_two_party.py --run outputs/federated_reproduction --screen-outcomes
```

The published aggregate results are under `outputs/federated_full/`. The fidelity script expects that run name and the improved two-party reference. For a reproduction, point its `federated` path at the new run directory before invoking `python assess_federated.py`.

Each worker reads its own feature partition. The coordinator receives decision labels, embeddings and gradients. This experiment uses separate processes on one Spark host and provides no formal privacy guarantee. The evaluation step joins the real partitions locally to compute the real-data baseline.
