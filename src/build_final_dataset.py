"""
FINAL chaos benchmark dataset (v2) — addressing limitations of v1.

Fixes applied:
  1. 80 examples per class per system (was 30) -> larger, more reliable
  2. Multiple negative classes, not just IAAFT surrogates:
       - iaaft_surrogate : spectrum + amplitude matched, structure destroyed
       - periodic        : genuinely periodic/quasi-periodic dynamics
       - coloured_noise  : stochastic with matched spectral slope
     -> tests discrimination against several kinds of "looks similar"
  3. Lyapunov ground truth computed numerically for ALL simulated systems
     (logistic, Lorenz, Mackey-Glass) via the Benettin/Wolf tangent-space
     method, not just the logistic map's analytic formula.
  4. EEG label honesty: seizure segments are labelled "nonlinear_contested",
     NOT "chaotic", with a provenance column recording how each label was
     established.

Columns:
  id, system, domain, label, class_type, label_source, lyapunov_exponent,
  t0..t99
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
import numpy as np, pandas as pd

RNG = np.random.default_rng(2024)
LEN = 100
N_PER = 80

def z(x): 
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)

def resample(x, n=LEN):
    return np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x)

# ---------- generators ----------
def logistic(r, x0, n, burn=400):
    x = x0
    for _ in range(burn): x = r*x*(1-x)
    o = np.empty(n)
    for i in range(n):
        x = r*x*(1-x); o[i] = x
    return o

def logistic_lyap(r, x0, n=5000, burn=1000):
    x = x0
    for _ in range(burn): x = r*x*(1-x)
    tot = 0.0
    for _ in range(n):
        x = r*x*(1-x)
        tot += np.log(abs(r*(1-2*x)) + 1e-15)
    return tot/n

def lorenz_traj(n, dt=0.01, sigma=10, rho=28, beta=8/3, x0=None, rng=None, burn=2000):
    v = x0 if x0 is not None else rng.uniform(-10,10,3)
    for _ in range(burn):
        x,y,zz = v
        v = v + dt*np.array([sigma*(y-x), x*(rho-zz)-y, x*y-beta*zz])
    out = np.empty((n,3))
    for i in range(n):
        x,y,zz = v
        v = v + dt*np.array([sigma*(y-x), x*(rho-zz)-y, x*y-beta*zz])
        out[i] = v
    return out

def lorenz_lyap(dt=0.01, n=20000, sigma=10, rho=28, beta=8/3, rng=None):
    """Largest Lyapunov exponent via tangent-space (Benettin) method."""
    v = rng.uniform(-10,10,3)
    for _ in range(3000):
        x,y,zz = v
        v = v + dt*np.array([sigma*(y-x), x*(rho-zz)-y, x*y-beta*zz])
    w = rng.normal(size=3); w /= np.linalg.norm(w)
    tot = 0.0
    for _ in range(n):
        x,y,zz = v
        J = np.array([[-sigma, sigma, 0],[rho-zz, -1, -x],[y, x, -beta]])
        w = w + dt*(J @ w)
        v = v + dt*np.array([sigma*(y-x), x*(rho-zz)-y, x*y-beta*zz])
        nw = np.linalg.norm(w)
        tot += np.log(nw); w /= nw
    return tot/(n*dt)

def mackey_glass(tau, n, beta=0.2, gamma=0.1, dt=1.0, rng=None, burn=500):
    h = int(tau/dt)
    total = h + burn + n
    # NOTE: must initialise h+1 entries, not h -- x[h] is read on the first
    # iteration. Using np.empty and filling only x[:h] leaves x[h] as
    # uninitialised memory, which intermittently produced NaN trajectories.
    x = np.zeros(total)
    x[:h+1] = 1.2 + rng.uniform(-0.05,0.05,h+1)
    for t in range(h, total-1):
        xt = x[t-h]
        x[t+1] = x[t] + dt*(beta*xt/(1+xt**10) - gamma*x[t])
    return x[h+burn:]

def mackey_glass_lyap(tau, rng, n=6000, eps=1e-8):
    """Largest Lyapunov exponent by trajectory-separation (Wolf-style)."""
    h = int(tau)
    def step(hist):
        xt = hist[0]; xn = hist[-1]
        return xn + (0.2*xt/(1+xt**10) - 0.1*xn)
    a = list(1.2 + rng.uniform(-0.05,0.05,h+1))
    for _ in range(2000):
        a = a[1:] + [step(a)]
    b = a.copy(); b[-1] += eps
    tot = 0.0; cnt = 0
    for _ in range(n):
        a = a[1:] + [step(a)]
        b = b[1:] + [step(b)]
        d = abs(b[-1]-a[-1])
        if d > 0:
            tot += np.log(d/eps); cnt += 1
            b = [ai + (bi-ai)*eps/d for ai,bi in zip(a,b)]
    return tot/cnt if cnt else np.nan

def iaaft(x, n_iter=300, rng=None):
    sx = np.sort(x); amp = np.abs(np.fft.fft(x))
    s = rng.permutation(x).astype(float)
    for _ in range(n_iter):
        ph = np.angle(np.fft.fft(s))
        s = np.real(np.fft.ifft(amp*np.exp(1j*ph)))
        s = sx[np.argsort(np.argsort(s))]
    return s

def coloured_noise(target, rng):
    """Stochastic surrogate with the same spectral slope: phase-randomised
    Gaussian noise shaped to the target's power spectrum (no rank-ordering,
    so its amplitude distribution stays Gaussian -- a different negative
    from IAAFT)."""
    n = len(target)
    amp = np.abs(np.fft.rfft(target))
    ph = rng.uniform(0, 2*np.pi, len(amp)); ph[0] = 0
    return np.fft.irfft(amp*np.exp(1j*ph), n=n)

rows = []
def add(sysname, domain, label, ctype, src, lyap, series):
    if not np.all(np.isfinite(series)):
        raise ValueError(f"non-finite values in {sysname}/{ctype}")
    rows.append({"system":sysname, "domain":domain, "label":label,
                 "class_type":ctype, "label_source":src,
                 "lyapunov_exponent":lyap,
                 **{f"t{i}":v for i,v in enumerate(series)}})

# ---------- 1. Logistic map (near boundary) ----------
rs = [3.57, 3.59, 3.60, 3.62]
periodic_rs = [3.20, 3.40, 3.50, 3.55]   # period-2/4/8 windows
for i in range(N_PER):
    r = RNG.choice(rs); x0 = RNG.uniform(.05,.95)
    ser = z(resample(logistic(r, x0, 400)))
    lam = logistic_lyap(r, x0)
    add("logistic_map","simulated","chaotic","positive",
        "analytic_lyapunov", lam, ser)
    add("logistic_map","simulated","not_chaotic","iaaft_surrogate",
        "constructed_surrogate", np.nan, z(iaaft(ser, rng=RNG)))
for i in range(N_PER):
    r = RNG.choice(periodic_rs); x0 = RNG.uniform(.05,.95)
    ser = z(resample(logistic(r, x0, 400)) + RNG.normal(0,0.02,LEN))
    add("logistic_map","simulated","not_chaotic","periodic",
        "analytic_lyapunov", logistic_lyap(r, x0), ser)

# ---------- 2. Lorenz ----------
lor_lam = lorenz_lyap(rng=RNG)
print(f"Lorenz largest Lyapunov exponent (computed): {lor_lam:.4f}  [literature ~0.906]")
for i in range(N_PER):
    ser = z(resample(lorenz_traj(3000, rng=RNG)[:,0]))
    add("lorenz","simulated","chaotic","positive","numerical_lyapunov", lor_lam, ser)
    add("lorenz","simulated","not_chaotic","iaaft_surrogate",
        "constructed_surrogate", np.nan, z(iaaft(ser, rng=RNG)))
    add("lorenz","simulated","not_chaotic","coloured_noise",
        "constructed_stochastic", np.nan, z(coloured_noise(ser, RNG)))

# ---------- 3. Mackey-Glass ----------
mg_lam = mackey_glass_lyap(17, RNG)
print(f"Mackey-Glass (tau=17) largest Lyapunov exponent (computed): {mg_lam:.4f}  [literature ~0.006 per step]")
for i in range(N_PER):
    ser = z(resample(mackey_glass(17, 1200, rng=RNG)))
    add("mackey_glass","simulated","chaotic","positive","numerical_lyapunov", mg_lam, ser)
    add("mackey_glass","simulated","not_chaotic","iaaft_surrogate",
        "constructed_surrogate", np.nan, z(iaaft(ser, rng=RNG)))
    add("mackey_glass","simulated","not_chaotic","coloured_noise",
        "constructed_stochastic", np.nan, z(coloured_noise(ser, RNG)))
# quasi-periodic Mackey-Glass (tau=10 is non-chaotic) as a real negative
for i in range(N_PER):
    ser = z(resample(mackey_glass(10, 1200, rng=RNG)) + RNG.normal(0,0.02,LEN))
    add("mackey_glass","simulated","not_chaotic","periodic",
        "known_regime_tau10", np.nan, ser)

# ---------- 4. Real: Santa Fe laser ----------
laser = z(np.loadtxt(str(_DATA / "laser_raw.txt"))[:3000])
for i in range(N_PER):
    st = RNG.integers(0, len(laser)-200)
    seg = z(resample(laser[st:st+200]))
    add("laser","real","chaotic","positive","published_domain_knowledge", np.nan, seg)
    add("laser","real","not_chaotic","iaaft_surrogate",
        "constructed_surrogate", np.nan, z(iaaft(seg, rng=RNG)))
    add("laser","real","not_chaotic","coloured_noise",
        "constructed_stochastic", np.nan, z(coloured_noise(seg, RNG)))

# ---------- 5. Real: Bonn EEG (label honesty!) ----------
eeg = pd.read_csv(str(_DATA / "eeg_raw.csv"), index_col=0)
sez = eeg[eeg["y"]==1].iloc[:, :178].values
hlt = eeg[eeg["y"]==5].iloc[:, :178].values
for i in range(N_PER):
    seg = z(resample(sez[i]))
    add("eeg_seizure","real","nonlinear_contested","positive",
        "contested_literature", np.nan, seg)
    add("eeg_seizure","real","not_chaotic","iaaft_surrogate",
        "constructed_surrogate", np.nan, z(iaaft(seg, rng=RNG)))
for i in range(N_PER):
    add("eeg_healthy","real","nonlinear_contested","positive",
        "contested_literature", np.nan, z(resample(hlt[i])))
    add("eeg_healthy","real","not_chaotic","iaaft_surrogate",
        "constructed_surrogate", np.nan, z(iaaft(z(resample(hlt[i])), rng=RNG)))

df = pd.DataFrame(rows)
df.insert(0, "id", [f"{r.system}_{r.class_type}_{i:04d}" for i, r in enumerate(df.itertuples())])
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv(str(_DATA / "chaos_benchmark_v2.csv"), index=False)

print(f"\nFinal dataset: {df.shape[0]} series x {LEN} points")
print("\nBreakdown by system and class type:")
print(df.groupby(["domain","system","label","class_type"]).size().to_string())
