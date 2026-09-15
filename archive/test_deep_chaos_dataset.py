import numpy as np
import pandas as pd
import pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("/mnt/user-data/outputs/deep_chaos_benchmark.csv")
series_cols = [c for c in df.columns if c.startswith("t")]
y = (df["label"] == "chaotic").astype(int).values

def lag1_autocorr(row):
    s = row[series_cols].values.astype(float)
    return np.corrcoef(s[:-1], s[1:])[0, 1]

df["mean_val"] = df[series_cols].mean(axis=1)
df["std_val"] = df[series_cols].std(axis=1)
df["lag1_autocorr"] = df.apply(lag1_autocorr, axis=1)

X_naive = df[["mean_val", "std_val", "lag1_autocorr"]].fillna(0).values
clf = RandomForestClassifier(n_estimators=300, random_state=42)
naive_scores = cross_val_score(clf, X_naive, y, cv=5)

feature_rows = []
for _, row in df.iterrows():
    s = row[series_cols].values.astype(float).tolist()
    feature_rows.append(pycatch22.catch22_all(s)["values"])
X_catch22 = np.array(feature_rows)
catch22_scores = cross_val_score(clf, X_catch22, y, cv=5)

print("="*60)
print("FINAL COMBINED DATASET (5 systems, sim + real, near-boundary)")
print("="*60)
print(f"Naive baseline: {naive_scores.mean():.3f} +/- {naive_scores.std():.3f}")
print(f"catch22:        {catch22_scores.mean():.3f} +/- {catch22_scores.std():.3f}")

# Break down catch22 performance per system -- where is it still strong/weak?
print("\nPer-system breakdown (catch22 trained on everything, checked per system):")
clf.fit(X_catch22, y)
preds = clf.predict(X_catch22)
df["correct"] = (preds == y)
print(df.groupby("system")["correct"].mean())
