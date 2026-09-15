"""
Fixed version: balanced classes using overlapping sliding windows on both
the real laser series and a single matched surrogate series.
"""

import numpy as np
import pandas as pd
import pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

RNG = np.random.default_rng(7)

def iaaft_surrogate(x, n_iter=300, rng=None):
    n = len(x)
    sorted_x = np.sort(x)
    target_amplitudes = np.abs(np.fft.fft(x))
    surrogate = rng.permutation(x).astype(float)
    for _ in range(n_iter):
        s_fft = np.fft.fft(surrogate)
        phases = np.angle(s_fft)
        new_fft = target_amplitudes * np.exp(1j * phases)
        surrogate = np.real(np.fft.ifft(new_fft))
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_x[ranks]
    return surrogate

raw = np.loadtxt("laser_raw.txt")
series = raw[:1000]
series = (series - series.mean()) / series.std()

surrogate = iaaft_surrogate(series, rng=RNG)

WINDOW = 60
STRIDE = 10  # overlapping windows -> more examples, balanced across classes

def make_windows(x, label):
    rows = []
    for start in range(0, len(x) - WINDOW, STRIDE):
        seg = x[start:start+WINDOW]
        rows.append({"label": label, **{f"t{t}": v for t, v in enumerate(seg)}})
    return rows

rows = make_windows(series, "real_laser") + make_windows(surrogate, "surrogate_decoy")
df = pd.DataFrame(rows)
print(f"Balanced dataset: {df['label'].value_counts().to_dict()}")

series_cols = [c for c in df.columns if c.startswith("t")]

def lag1_autocorr(row):
    s = row[series_cols].values.astype(float)
    return np.corrcoef(s[:-1], s[1:])[0, 1]

df["mean_val"] = df[series_cols].mean(axis=1)
df["std_val"] = df[series_cols].std(axis=1)
df["lag1_autocorr"] = df.apply(lag1_autocorr, axis=1)

X_naive = df[["mean_val", "std_val", "lag1_autocorr"]].fillna(0).values
y = (df["label"] == "real_laser").astype(int).values

clf = RandomForestClassifier(n_estimators=300, random_state=42)
naive_scores = cross_val_score(clf, X_naive, y, cv=5)

feature_rows = []
for _, row in df.iterrows():
    s = row[series_cols].values.astype(float).tolist()
    result = pycatch22.catch22_all(s)
    feature_rows.append(result["values"])
X_catch22 = np.array(feature_rows)
feature_names = result["names"]

catch22_scores = cross_val_score(clf, X_catch22, y, cv=5)

print("\n" + "="*60)
print("RESULTS ON REAL SANTA FE LASER DATA (balanced, corrected)")
print("="*60)
print(f"Naive baseline (mean/std/lag-1-autocorr): {naive_scores.mean():.3f} +/- {naive_scores.std():.3f}")
print(f"catch22 feature-based classifier:         {catch22_scores.mean():.3f} +/- {catch22_scores.std():.3f}")
print("="*60)

clf.fit(X_catch22, y)
importances = pd.Series(clf.feature_importances_, index=feature_names)
print("\nTop 5 catch22 features distinguishing real laser data from its surrogate:")
for name, imp in importances.sort_values(ascending=False).head(5).items():
    print(f"  {name}: {imp:.3f}")
