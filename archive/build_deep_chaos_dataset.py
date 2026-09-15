"""
Deeper, harder chaos benchmark combining multiple chaotic systems and real
world data, all in one standardized format for the team's shared database.

Systems included:
  - Logistic map, near the boundary of chaos (harder than before)
  - Lorenz system (classic 3-variable chaotic attractor, "butterfly effect")
  - Mackey-Glass equation (delay differential equation, classic chaos
    benchmark from physiology - widely used in the ML literature)
  - Real Santa Fe laser data
  - Real Bonn EEG data (seizure segments, known to show nonlinear structure)

Every series is resampled to the same fixed length and z-normalized, so
they're directly comparable. Every simulated chaotic example gets an
IAAFT surrogate decoy twin, same as before.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(123)
LENGTH = 100  # unified window length for the whole database

def resample_to_length(x, length=LENGTH):
    idx = np.linspace(0, len(x) - 1, length)
    return np.interp(idx, np.arange(len(x)), x)

def zscore(x):
    return (x - x.mean()) / x.std()

# ---------- System 1: Logistic map, near the boundary of chaos ----------
def logistic_map(r, x0, n, burn=300):
    x = x0
    for _ in range(burn): x = r*x*(1-x)
    out = np.empty(n)
    for i in range(n):
        x = r*x*(1-x)
        out[i] = x
    return out

def lyapunov_logistic(r, x0, n=3000, burn=500):
    x = x0
    for _ in range(burn): x = r*x*(1-x)
    total = 0.0
    for _ in range(n):
        x = r*x*(1-x)
        total += np.log(abs(r*(1-2*x)) + 1e-12)
    return total / n

# ---------- System 2: Lorenz system (classic chaotic attractor) ----------
def lorenz(sigma=10, rho=28, beta=8/3, n=4000, dt=0.01, x0=None, rng=None):
    x0 = x0 if x0 is not None else rng.uniform(-10, 10, 3)
    traj = np.empty((n, 3))
    x, y, z = x0
    for i in range(n):
        dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
        x += dx*dt; y += dy*dt; z += dz*dt
        traj[i] = [x, y, z]
    return traj[:, 0]  # observe only the x-component (typical in real experiments)

# ---------- System 3: Mackey-Glass delay differential equation ----------
def mackey_glass(tau, beta=0.2, gamma=0.1, n=1200, dt=1.0, rng=None):
    history_len = int(tau / dt)
    x = np.ones(history_len + n) * 1.2 + rng.uniform(-0.05, 0.05, history_len + n)
    for t in range(history_len, history_len + n - 1):
        x_tau = x[t - history_len]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**10) - gamma * x[t])
    return x[history_len+300:]  # drop transient

# ---------- Surrogate builder (same as before) ----------
def iaaft_surrogate(x, n_iter=250, rng=None):
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.fft(x))
    surrogate = rng.permutation(x).astype(float)
    for _ in range(n_iter):
        phases = np.angle(np.fft.fft(surrogate))
        surrogate = np.real(np.fft.ifft(target_amp * np.exp(1j*phases)))
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_x[ranks]
    return surrogate

rows = []

# --- Logistic map, near boundary (harder r values than before) ---
near_boundary_rs = [3.57, 3.59, 3.60, 3.61]  # right at/just past onset of chaos
for i in range(30):
    r = RNG.choice(near_boundary_rs)
    x0 = RNG.uniform(0.05, 0.95)
    series = resample_to_length(logistic_map(r, x0, 300))
    series = zscore(series)
    lam = lyapunov_logistic(r, x0)
    rows.append({"id": f"logistic_{i:03d}", "domain": "simulated", "system": "logistic_map_boundary",
                 "label": "chaotic", "lyapunov_exponent": lam, **{f"t{t}": v for t,v in enumerate(series)}})
    decoy = zscore(iaaft_surrogate(series, rng=RNG))
    rows.append({"id": f"logistic_decoy_{i:03d}", "domain": "simulated", "system": "logistic_map_boundary",
                 "label": "surrogate_decoy", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(decoy)}})

# --- Lorenz system (genuinely chaotic, classic rho=28) ---
for i in range(30):
    series = resample_to_length(lorenz(rho=28, n=3000, rng=RNG))
    series = zscore(series)
    rows.append({"id": f"lorenz_{i:03d}", "domain": "simulated", "system": "lorenz",
                 "label": "chaotic", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(series)}})
    decoy = zscore(iaaft_surrogate(series, rng=RNG))
    rows.append({"id": f"lorenz_decoy_{i:03d}", "domain": "simulated", "system": "lorenz",
                 "label": "surrogate_decoy", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(decoy)}})

# --- Mackey-Glass (tau=17 -> chaotic; classic physiological benchmark) ---
for i in range(30):
    series = resample_to_length(mackey_glass(tau=17, n=900, rng=RNG))
    series = zscore(series)
    rows.append({"id": f"mackeyglass_{i:03d}", "domain": "simulated", "system": "mackey_glass",
                 "label": "chaotic", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(series)}})
    decoy = zscore(iaaft_surrogate(series, rng=RNG))
    rows.append({"id": f"mackeyglass_decoy_{i:03d}", "domain": "simulated", "system": "mackey_glass",
                 "label": "surrogate_decoy", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(decoy)}})

# --- Real Santa Fe laser data ---
raw_laser = np.loadtxt("laser_raw.txt")[:2000]
raw_laser = zscore(raw_laser)
for i in range(30):
    start = RNG.integers(0, len(raw_laser)-150)
    seg = resample_to_length(raw_laser[start:start+150])
    rows.append({"id": f"laser_{i:03d}", "domain": "real", "system": "laser",
                 "label": "chaotic", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(seg)}})
    decoy = zscore(iaaft_surrogate(seg, rng=RNG))
    rows.append({"id": f"laser_decoy_{i:03d}", "domain": "real", "system": "laser",
                 "label": "surrogate_decoy", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(decoy)}})

# --- Real EEG seizure data ---
eeg = pd.read_csv("eeg_raw.csv", index_col=0)
seizure_segs = eeg[eeg["y"]==1].iloc[:, :178].values
for i in range(30):
    seg = resample_to_length(zscore(seizure_segs[i]))
    rows.append({"id": f"eeg_{i:03d}", "domain": "real", "system": "eeg_seizure",
                 "label": "chaotic", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(seg)}})
    decoy = zscore(iaaft_surrogate(seg, rng=RNG))
    rows.append({"id": f"eeg_decoy_{i:03d}", "domain": "real", "system": "eeg_seizure",
                 "label": "surrogate_decoy", "lyapunov_exponent": np.nan, **{f"t{t}": v for t,v in enumerate(decoy)}})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/mnt/user-data/outputs/deep_chaos_benchmark.csv", index=False)
print(f"Final combined dataset: {df.shape}")
print(df.groupby(["domain","system","label"]).size())
