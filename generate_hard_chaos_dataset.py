"""
HARD chaos-vs-decoy benchmark dataset generator
--------------------------------------------------
Upgrades the first version using a real technique from nonlinear dynamics
research: "surrogate data testing" (Theiler et al. 1992; Schreiber & Schmitz
1996, IAAFT method).

The idea: for every genuinely chaotic sequence, generate a "decoy twin" that
has the IDENTICAL average value, spread, and frequency content (power
spectrum) -- but with the real chaotic mechanism destroyed and replaced by
randomized phases. The decoy LOOKS statistically the same by every standard
linear measure. Only genuine nonlinear/deterministic structure can tell them
apart.

This directly tests whether an algorithm detects real chaos, or is secretly
relying on linear shortcuts (autocorrelation, spectrum, variance) that we've
now deliberately equalized across classes.
"""

import numpy as np
import pandas as pd

def logistic_map(r, x0, n_steps, burn_in=200):
    x = x0
    for _ in range(burn_in):
        x = r * x * (1 - x)
    traj = np.empty(n_steps)
    for i in range(n_steps):
        x = r * x * (1 - x)
        traj[i] = x
    return traj

def lyapunov_exponent_logistic(r, x0, n_steps=2000, burn_in=500):
    x = x0
    for _ in range(burn_in):
        x = r * x * (1 - x)
    total = 0.0
    for _ in range(n_steps):
        x = r * x * (1 - x)
        total += np.log(abs(r * (1 - 2 * x)) + 1e-12)
    return total / n_steps

def iaaft_surrogate(x, n_iter=300, rng=None):
    """Iterative Amplitude Adjusted Fourier Transform (Schreiber & Schmitz 1996).
    Produces a surrogate series with the SAME power spectrum and the SAME
    amplitude distribution as x, but randomized phase relationships --
    destroying genuine nonlinear/deterministic structure while preserving
    every linear statistic."""
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

def build_hard_dataset(n_pairs=100, n_steps=80, noise_std=0.02, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # Near-boundary chaotic r values -> WEAKLY chaotic, small positive
    # Lyapunov exponent (harder than the strongly chaotic r=3.9-3.99 before)
    chaotic_rs = [3.60, 3.62, 3.65, 3.68]

    for i in range(n_pairs):
        r = rng.choice(chaotic_rs)
        x0 = rng.uniform(0.05, 0.95)
        traj = logistic_map(r, x0, n_steps)
        traj = traj + rng.normal(0, noise_std, n_steps)
        traj = (traj - traj.mean()) / traj.std()
        lam = lyapunov_exponent_logistic(r, x0)

        rows.append({
            "id": f"chaotic_{i:03d}", "label": "chaotic",
            "r_param": r, "lyapunov_exponent": lam,
            **{f"t{t}": v for t, v in enumerate(traj)}
        })

        # Decoy twin: same spectrum + same amplitude distribution, no real chaos
        decoy = iaaft_surrogate(traj, rng=rng)
        rows.append({
            "id": f"decoy_{i:03d}", "label": "surrogate_decoy",
            "r_param": np.nan, "lyapunov_exponent": np.nan,
            **{f"t{t}": v for t, v in enumerate(decoy)}
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    df = build_hard_dataset(n_pairs=100, n_steps=80)
    df.to_csv("hard_chaos_benchmark.csv", index=False)

    series_cols = [c for c in df.columns if c.startswith("t")]
    print(f"Dataset shape: {df.shape}")
    print(f"Class balance:\n{df['label'].value_counts()}\n")

    # --- Confound check: mean, std, and lag-1 autocorrelation ---
    def lag1_autocorr(row):
        s = row[series_cols].values.astype(float)
        return np.corrcoef(s[:-1], s[1:])[0, 1]

    df["mean_val"] = df[series_cols].mean(axis=1)
    df["std_val"] = df[series_cols].std(axis=1)
    df["lag1_autocorr"] = df.apply(lag1_autocorr, axis=1)

    print("Confound check across classes (mean / std / lag-1 autocorrelation):")
    print(df.groupby("label")[["mean_val", "std_val", "lag1_autocorr"]].agg(["mean", "std"]))

    # --- Naive baseline classifier using ONLY these simple linear features ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X = df[["mean_val", "std_val", "lag1_autocorr"]].fillna(0).values
    y = (df["label"] == "chaotic").astype(int).values
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)

    print(f"\nNaive baseline (mean/std/lag-1-autocorr only) cross-val accuracy:")
    print(f"{scores.mean():.3f} +/- {scores.std():.3f}  (0.5 = random guessing)")
