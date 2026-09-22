"""
Leave-one-domain-out test: train catch22 on 4 systems, test on the 5th,
completely unseen. This tests whether the benchmark teaches genuine
chaos-detection, or just memorizes quirks of each specific system.
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"

import numpy as np
import pandas as pd
import pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(str(_DATA / "chaos_benchmark_v2.csv"))
series_cols = [c for c in df.columns if c.startswith("t")]
# NOTE: fixed 2026-09 -- the old check `label == "chaotic"` silently
# mislabelled every EEG positive as negative, since EEG positives are
# labelled "nonlinear_contested" (not "chaotic"), reflecting genuine
# scientific dispute over whether EEG shows true chaos. The correct
# target is class_type, which is consistent across all systems.
y_all = (df["class_type"] == "positive").astype(int).values

print("Extracting catch22 features once for the whole dataset...")
feature_rows = []
for _, row in df.iterrows():
    s = row[series_cols].values.astype(float).tolist()
    feature_rows.append(pycatch22.catch22_all(s)["values"])
X_all = np.array(feature_rows)
df["_row_idx"] = np.arange(len(df))

systems = df["system"].unique()
print(f"\nSystems: {list(systems)}\n")

results = {}
for held_out in systems:
    train_mask = df["system"] != held_out
    test_mask = df["system"] == held_out

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[held_out] = acc
    n_other = len(systems) - 1
    print(f"Held out: {held_out:24s} | trained on the other {n_other} systems | test accuracy: {acc:.3f}")

print("\n" + "="*60)
print(f"Average leave-one-domain-out accuracy: {np.mean(list(results.values())):.3f}")
print("Compare to same-distribution 5-fold CV accuracy (catch22): 0.875")
print("="*60)
