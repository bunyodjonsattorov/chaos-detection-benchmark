"""
Chaos vs non-chaos benchmark dataset generator (logistic map)
---------------------------------------------------------------
Generates labeled time series for a physics-relevant TSC benchmark problem:
distinguishing CHAOTIC from NON-CHAOTIC (periodic) dynamics.

Key design choices to make this a genuine "process" problem rather than
a trivial "pattern" problem an algorithm could cheat on:
  1. Random starting points -> no fixed phase/shape to memorize
  2. Matched observation noise added to BOTH classes
  3. Z-normalization -> mean and variance can't be used as a shortcut
  4. Ground truth Lyapunov exponent computed analytically, not just
     inferred from the label -> lets you grade difficulty later, not
     just binary classify
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def logistic_map(r, x0, n_steps, burn_in=200):
    """Iterate the logistic map, discarding a burn-in period so we're
    sampling from the attractor, not the transient."""
    x = x0
    traj = np.empty(n_steps)
    for i in range(burn_in):
        x = r * x * (1 - x)
    for i in range(n_steps):
        x = r * x * (1 - x)
        traj[i] = x
    return traj

def lyapunov_exponent_logistic(r, x0, n_steps=2000, burn_in=500):
    """Analytical Lyapunov exponent for the logistic map:
    lambda = average of log|f'(x)| = log|r(1-2x)| along the trajectory.
    Positive => chaotic. Negative => stable/periodic."""
    x = x0
    for _ in range(burn_in):
        x = r * x * (1 - x)
    total = 0.0
    for _ in range(n_steps):
        x = r * x * (1 - x)
        deriv = abs(r * (1 - 2 * x))
        total += np.log(deriv + 1e-12)
    return total / n_steps

def make_series(r, n_steps, noise_std, rng):
    x0 = rng.uniform(0.05, 0.95)  # random starting point -> no fixed phase
    traj = logistic_map(r, x0, n_steps)
    traj = traj + rng.normal(0, noise_std, n_steps)  # measurement noise
    traj = (traj - traj.mean()) / traj.std()          # z-normalize
    lam = lyapunov_exponent_logistic(r, x0)
    return traj, lam

def build_dataset(n_per_class=100, n_steps=150, noise_std=0.02, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # Chaotic class: r values known to be chaotic (positive Lyapunov exponent)
    chaotic_rs = [3.9, 3.95, 3.99, 3.7]
    # Non-chaotic class: r values giving fixed points / periodic cycles
    periodic_rs = [2.8, 3.2, 3.45, 3.5]

    for i in range(n_per_class):
        r = rng.choice(chaotic_rs)
        traj, lam = make_series(r, n_steps, noise_std, rng)
        rows.append({
            "id": f"chaotic_{i:03d}",
            "label": "chaotic",
            "r_param": r,
            "lyapunov_exponent": lam,
            **{f"t{t}": v for t, v in enumerate(traj)}
        })

    for i in range(n_per_class):
        r = rng.choice(periodic_rs)
        traj, lam = make_series(r, n_steps, noise_std, rng)
        rows.append({
            "id": f"periodic_{i:03d}",
            "label": "non_chaotic",
            "r_param": r,
            "lyapunov_exponent": lam,
            **{f"t{t}": v for t, v in enumerate(traj)}
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

if __name__ == "__main__":
    df = build_dataset(n_per_class=100, n_steps=150)
    df.to_csv("chaos_benchmark_logistic_map.csv", index=False)

    print(f"Dataset shape: {df.shape}")
    print(f"\nClass balance:\n{df['label'].value_counts()}")
    print(f"\nLyapunov exponent by class (should be clearly separated even though")
    print(f"the raw normalized series stats are matched):")
    print(df.groupby("label")["lyapunov_exponent"].describe()[["mean", "std", "min", "max"]])

    # Sanity check: confirm the "confound control" worked -- means/stds of the
    # actual normalized series should be near-identical across classes
    series_cols = [c for c in df.columns if c.startswith("t")]
    means = df.groupby("label")[series_cols].mean().mean(axis=1)
    stds = df.groupby("label")[series_cols].std().mean(axis=1)
    print(f"\nConfound check (should be near-equal across classes):")
    print(f"Mean of series values per class:\n{means}")
    print(f"Std of series values per class:\n{stds}")
