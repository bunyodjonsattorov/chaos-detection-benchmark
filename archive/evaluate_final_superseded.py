import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
import numpy as np, pandas as pd, pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

df = pd.read_csv(str(_DATA / "chaos_benchmark_v2.csv"))
tc = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]

# Binary task: does the series carry genuine nonlinear/deterministic structure?
y = (df["class_type"] == "positive").astype(int).values

print("Extracting catch22 features for 1360 series...")
X = np.array([pycatch22.catch22_all(r[tc].values.astype(float).tolist())["values"]
              for _, r in df.iterrows()])

def lag1(r):
    s = r[tc].values.astype(float)
    return np.corrcoef(s[:-1], s[1:])[0,1]
df["m"]=df[tc].mean(axis=1); df["s"]=df[tc].std(axis=1); df["a"]=df.apply(lag1,axis=1)
Xn = df[["m","s","a"]].fillna(0).values

cv = StratifiedKFold(5, shuffle=True, random_state=1)
clf = lambda: RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

print("\n" + "="*66)
print("BENCHMARK v2  —  1360 series, 5 systems, 4 class types")
print("="*66)
n = cross_val_score(clf(), Xn, y, cv=cv)
c = cross_val_score(clf(), X,  y, cv=cv)
print(f"Naive baseline (mean/std/lag-1):  {n.mean():.3f} +/- {n.std():.3f}")
print(f"catch22:                          {c.mean():.3f} +/- {c.std():.3f}")

# Per-negative-class difficulty: how well can we separate positives from EACH negative?
print("\nDifficulty by negative class (positives vs that negative only):")
for neg in ["iaaft_surrogate","coloured_noise","periodic"]:
    m = df["class_type"].isin(["positive", neg]).values
    yy = (df.loc[m,"class_type"]=="positive").astype(int).values
    sc = cross_val_score(clf(), X[m], yy, cv=cv)
    sn = cross_val_score(clf(), Xn[m], yy, cv=cv)
    print(f"  vs {neg:18s}  naive {sn.mean():.3f}   catch22 {sc.mean():.3f}")

# Leave-one-system-out generalisation
print("\nLeave-one-system-out (train on other systems, test on unseen):")
accs=[]
for sysname in df["system"].unique():
    tr = (df["system"]!=sysname).values; te = ~tr
    m = clf().fit(X[tr], y[tr])
    a = accuracy_score(y[te], m.predict(X[te])); accs.append(a)
    print(f"  held out {sysname:14s}  {a:.3f}")
print(f"\n  Average across held-out systems: {np.mean(accs):.3f}")
print("="*66)
