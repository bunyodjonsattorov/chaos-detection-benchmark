"""
Close the loop: apply our full pipeline (surrogate testing, catch22, and
the independently implemented Chaos Decision Tree method) to real
biomedical data -- the Bonn University EEG dataset.

y=5: healthy, eyes open      y=1: seizure activity (ictal)
We test: healthy vs seizure, and also validate with the Toker et al. method.
"""

import numpy as np
import pandas as pd
import pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys
sys.path.insert(0, '.')
from chaos_decision_tree import stochasticity_test, zero_one_test

RNG = np.random.default_rng(11)

df = pd.read_csv("eeg_raw.csv", index_col=0)
feature_cols = [c for c in df.columns if c.startswith("X")]

healthy = df[df["y"] == 5][feature_cols].values  # eyes open
seizure = df[df["y"] == 1][feature_cols].values  # ictal

print(f"Healthy (eyes open) segments: {len(healthy)}")
print(f"Seizure segments: {len(seizure)}")

# Balance and subsample for speed
n = 150
rng_idx = RNG.choice(len(healthy), n, replace=False)
healthy_sample = healthy[rng_idx]
rng_idx = RNG.choice(len(seizure), n, replace=False)
seizure_sample = seizure[rng_idx]

def zscore(x):
    return (x - x.mean()) / x.std()

rows = []
for s in healthy_sample:
    rows.append({"label": "healthy", **{f"t{i}": v for i, v in enumerate(zscore(s))}})
for s in seizure_sample:
    rows.append({"label": "seizure", **{f"t{i}": v for i, v in enumerate(zscore(s))}})

edf = pd.DataFrame(rows)
series_cols = [c for c in edf.columns if c.startswith("t")]

# Naive baseline
def lag1_autocorr(row):
    s = row[series_cols].values.astype(float)
    return np.corrcoef(s[:-1], s[1:])[0, 1]

edf["mean_val"] = edf[series_cols].mean(axis=1)
edf["std_val"] = edf[series_cols].std(axis=1)
edf["lag1_autocorr"] = edf.apply(lag1_autocorr, axis=1)

X_naive = edf[["mean_val", "std_val", "lag1_autocorr"]].values
y = (edf["label"] == "seizure").astype(int).values
clf = RandomForestClassifier(n_estimators=300, random_state=42)
naive_scores = cross_val_score(clf, X_naive, y, cv=5)

# catch22
feature_rows = []
for _, row in edf.iterrows():
    s = row[series_cols].values.astype(float).tolist()
    result = pycatch22.catch22_all(s)
    feature_rows.append(result["values"])
X_catch22 = np.array(feature_rows)
feature_names = result["names"]
catch22_scores = cross_val_score(clf, X_catch22, y, cv=5)

print("\n" + "="*60)
print("RESULTS: REAL EEG DATA (healthy vs seizure)")
print("="*60)
print(f"Naive baseline (mean/std/lag-1-autocorr): {naive_scores.mean():.3f} +/- {naive_scores.std():.3f}")
print(f"catch22 feature-based classifier:         {catch22_scores.mean():.3f} +/- {catch22_scores.std():.3f}")
print("="*60)

clf.fit(X_catch22, y)
importances = pd.Series(clf.feature_importances_, index=feature_names)
print("\nTop 5 catch22 features distinguishing healthy from seizure EEG:")
for name, imp in importances.sort_values(ascending=False).head(5).items():
    print(f"  {name}: {imp:.3f}")

# Chaos Decision Tree method on a few examples of each
print("\n--- Chaos Decision Tree verdicts on individual EEG segments ---")
for i in range(3):
    x = zscore(healthy_sample[i])
    is_stoch, pe, _ = stochasticity_test(x, rng=RNG)
    verdict = "stochastic" if is_stoch else f"deterministic, K={zero_one_test(x, rng=RNG):.3f}"
    print(f"Healthy #{i}: {verdict}")
for i in range(3):
    x = zscore(seizure_sample[i])
    is_stoch, pe, _ = stochasticity_test(x, rng=RNG)
    verdict = "stochastic" if is_stoch else f"deterministic, K={zero_one_test(x, rng=RNG):.3f}"
    print(f"Seizure #{i}: {verdict}")
