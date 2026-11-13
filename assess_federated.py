"""Compare full-schema generation against the same held-out records."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

root = Path("outputs")
baseline = root / "two_party_quality_candidate"
federated = root / "federated_full"
read = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)
heldout = read(federated / "heldout.csv")
reference = pd.read_csv(baseline / "conditional_metrics.csv")
synthetic = read(federated / "synthetic.csv")
assert set(heldout) == set(synthetic) and len(synthetic) == 10000
rows = []
for entry in reference.itertuples():
    label = str(entry.label)
    actual = heldout.loc[heldout.action_taken == label, entry.field]
    generated = synthetic.loc[synthetic.action_taken == label, entry.field]
    if entry.metric == "TV":
        frequencies = pd.concat([s.value_counts(normalize=True)
                                 for s in (actual, generated)], axis=1).fillna(0)
        value = frequencies.diff(axis=1).iloc[:, 1].abs().sum() / 2
    else:
        a, b = (pd.to_numeric(s, errors="coerce") for s in (actual, generated))
        if entry.metric == "KS":
            value = ks_2samp(a.dropna(), b.dropna()).statistic if a.count() and b.count() else np.nan
        else:
            value = abs(a.isna().mean() - b.isna().mean())
    rows.append((label, entry.field, entry.metric, abs(entry.value), value))
gaps = pd.DataFrame(rows, columns=["action", "field", "metric", "previous", "federated"])
gaps.to_csv(federated / "fidelity_comparison.csv", index=False)
print(gaps.dropna().groupby(["action", "metric"])[["previous", "federated"]].agg(["mean", "count"]))
print("Missing comparisons:", gaps[["previous", "federated"]].isna().sum().to_dict())
print("Duplicate fraction:", synthetic.duplicated().mean())
print(pd.read_csv(federated / "tstr_binary.csv").to_string(index=False))
