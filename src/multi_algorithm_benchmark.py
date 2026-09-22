"""
Benchmark the chaos dataset against a RANGE of algorithm families, not just catch22.

Four algorithms, each answering a different question:
  1. Naive statistical baseline   - is this dataset trivially easy?
  2. Linear AR(5) model           - is the structure explainable linearly?
                                     (the honest control: IAAFT surrogates
                                     preserve everything linear)
  3. catch22 (feature-based)      - can a physics-aware feature set detect it?
  4. ROCKET (convolution-based)   - can a generic SOTA method detect it?

Why the AR baseline matters: IAAFT surrogates preserve everything a LINEAR process
can explain. A properly-fitted linear model is therefore the honest control -- if it
fails while nonlinear methods succeed, that is direct evidence the separability comes
from nonlinear structure.
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import pycatch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = pd.read_csv(str(_DATA / "chaos_benchmark_v2.csv"))
tc = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]
X_raw = df[tc].values.astype(float)
y = (df["class_type"] == "positive").astype(int).values
print(f"Dataset: {X_raw.shape[0]} series x {X_raw.shape[1]} points, {y.sum()} positive / {(1-y).sum()} negative\n")

cv = StratifiedKFold(5, shuffle=True, random_state=1)

# ---------- feature sets ----------
def lag_ac(x, k):
    if k >= len(x): return 0.0
    a, b = x[:-k], x[k:]
    if a.std() == 0 or b.std() == 0: return 0.0
    return np.corrcoef(a, b)[0, 1]

# 1. Naive
X_naive = np.array([[x.mean(), x.std(), lag_ac(x,1)] for x in X_raw])

# 2. Linear AR(5) coefficients -- the proper linear control
def ar_coeffs(x, p=5):
    Xd = np.column_stack([x[p-i-1:len(x)-i-1] for i in range(p)])
    yd = x[p:]
    try:
        beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        resid = yd - Xd @ beta
        return np.concatenate([beta, [resid.std()]])
    except Exception:
        return np.zeros(p+1)
X_ar = np.array([ar_coeffs(x) for x in X_raw])

# 3. catch22
X_c22 = np.array([pycatch22.catch22_all(x.tolist())["values"] for x in X_raw])

# 4. ROCKET (convolution-based)
from sktime.transformations.panel.rocket import Rocket
rocket = Rocket(num_kernels=2000, random_state=42)
X_3d = X_raw.reshape(X_raw.shape[0], 1, X_raw.shape[1])
X_rocket = rocket.fit_transform(X_3d)
X_rocket = np.asarray(X_rocket)

results = {}
def run(name, X, clf):
    sc = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    results[name] = (sc.mean(), sc.std())
    print(f"  {name:34s} {sc.mean():.3f} +/- {sc.std():.3f}")

rf = lambda: RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

print("ACCURACY BY ALGORITHM FAMILY (5-fold CV)")
print("-"*60)
run("Naive statistics (mean/std/ac1)", X_naive, rf())
run("Linear AR(5) model", X_ar, rf())
run("Convolution-based (ROCKET)", X_rocket,
    make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3,3,10))))
run("Feature-based (catch22)", X_c22, rf())

# ---------- difficulty ladder across the best few ----------
print("\n" + "-"*60)
print("DIFFICULTY LADDER: positives vs each control type")
print("-"*60)
sets = {"Naive": X_naive, "Linear AR(5)": X_ar, "ROCKET": X_rocket, "catch22": X_c22}
print(f"{'':16s}" + "".join(f"{k:>14s}" for k in sets))
for neg in ["periodic", "coloured_noise", "iaaft_surrogate"]:
    m = df["class_type"].isin(["positive", neg]).values
    yy = (df.loc[m, "class_type"] == "positive").astype(int).values
    row = f"{neg:16s}"
    for k, Xs in sets.items():
        c = rf() if k != "ROCKET" else make_pipeline(StandardScaler(with_mean=False),
                                                     RidgeClassifierCV(alphas=np.logspace(-3,3,10)))
        sc = cross_val_score(c, Xs[m], yy, cv=cv, n_jobs=-1).mean()
        row += f"{sc:14.3f}"
    print(row)
