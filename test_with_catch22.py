"""
Test whether catch22 (Ben Fulcher's own feature set) can crack the hard
chaos-vs-surrogate-decoy benchmark, where naive linear stats (mean/std/
lag-1 autocorrelation) failed at 53.5% accuracy (~random guessing).
"""

import numpy as np
import pandas as pd
import pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../Chaos Dataset/hard_chaos_benchmark.csv")
series_cols = [c for c in df.columns if c.startswith("t")]

print("Extracting catch22 features for all 200 series...")
feature_rows = []
for _, row in df.iterrows():
    series = row[series_cols].values.astype(float).tolist()
    result = pycatch22.catch22_all(series)
    feature_rows.append(result["values"])

feature_names = result["names"]
X = np.array(feature_rows)
y = (df["label"] == "chaotic").astype(int).values

print(f"Extracted {X.shape[1]} catch22 features per series.\n")

clf = RandomForestClassifier(n_estimators=300, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Naive baseline (mean/std/lag-1-autocorr):  0.535 +/- 0.051")
print(f"catch22 feature-based classifier:          {scores.mean():.3f} +/- {scores.std():.3f}")
print("=" * 60)

# Which features actually mattered? Fit once on everything to inspect importances.
clf.fit(X, y)
importances = pd.Series(clf.feature_importances_, index=feature_names)
top5 = importances.sort_values(ascending=False).head(5)
print("\nTop 5 most useful catch22 features for telling chaos from decoy:")
for name, imp in top5.items():
    print(f"  {name}: {imp:.3f}")

# Confusion matrix for interpretability
y_pred = cross_val_predict(clf, X, y, cv=5)
cm = confusion_matrix(y, y_pred)
print(f"\nConfusion matrix (rows=true, cols=predicted) [decoy, chaotic]:")
print(cm)
