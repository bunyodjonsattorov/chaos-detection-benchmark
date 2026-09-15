import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
import math
"""
Simplified implementation of the core ideas from Toker, Sommer & D'Esposito
(2020) "A simple method for detecting chaos in nature" - the Chaos
Decision Tree Algorithm.

Step 1: test for stochasticity using permutation entropy + surrogates
         (this is the same surrogate idea we already built)
Step 2: if not stochastic, run the 0-1 test for chaos (K statistic)

We test this against series we already know the ground truth for:
chaotic logistic map, periodic logistic map, a surrogate decoy, and
the real Santa Fe laser data.
"""

import numpy as np
from itertools import permutations

RNG = np.random.default_rng(1)

def permutation_entropy(x, order=5, delay=1):
    """Bandt & Pompe (2002) permutation entropy: how unpredictable the
    up/down ordering pattern of the signal is."""
    n = len(x)
    perms = {p: 0 for p in permutations(range(order))}
    for i in range(n - (order - 1) * delay):
        window = x[i:i + order * delay:delay]
        pattern = tuple(np.argsort(window))
        perms[pattern] += 1
    counts = np.array(list(perms.values()))
    probs = counts[counts > 0] / counts.sum()
    return -np.sum(probs * np.log(probs)) / np.log(math.factorial(order))

def iaaft_surrogate(x, n_iter=200, rng=None):
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

def stochasticity_test(x, n_surrogates=25, order=5, rng=None):
    """Is the signal's permutation entropy consistent with its own
    surrogates? If yes -> stochastic. If no -> deterministic, proceed."""
    real_pe = permutation_entropy(x, order=order)
    surrogate_pes = []
    for _ in range(n_surrogates):
        surr = iaaft_surrogate(x, n_iter=150, rng=rng)
        surrogate_pes.append(permutation_entropy(surr, order=order))
    surrogate_pes = np.array(surrogate_pes)
    lo, hi = np.percentile(surrogate_pes, [2.5, 97.5])
    is_stochastic = lo <= real_pe <= hi
    return is_stochastic, real_pe, (lo, hi)

def zero_one_test(phi, n_c=50, rng=None):
    """Gottwald-Melbourne 0-1 test for chaos. K near 1 = chaotic,
    K near 0 = periodic."""
    N = len(phi)
    n_max = N // 10
    Ks = []
    for _ in range(n_c):
        c = rng.uniform(np.pi/5, 4*np.pi/5)  # avoid resonances near 0
        j = np.arange(1, N + 1)
        p = np.cumsum(phi * np.cos(j * c))
        q = np.cumsum(phi * np.sin(j * c))
        M = np.array([
            np.mean((p[n:] - p[:-n])**2 + (q[n:] - q[:-n])**2)
            for n in range(1, n_max + 1)
        ])
        K = np.corrcoef(np.arange(1, n_max + 1), M)[0, 1]
        Ks.append(K)
    return np.median(Ks)

def classify(x, rng, label=""):
    x = (x - x.mean()) / x.std()
    is_stoch, pe, (lo, hi) = stochasticity_test(x, rng=rng)
    if is_stoch:
        verdict = "STOCHASTIC"
        K = None
    else:
        K = zero_one_test(x, rng=rng)
        verdict = "CHAOTIC" if K > 0.5 else "PERIODIC"
    print(f"{label:20s} | PE={pe:.3f} (surrogate range {lo:.3f}-{hi:.3f}) "
          f"| stochastic={is_stoch} | K={K if K is None else f'{K:.3f}'} -> {verdict}")

# --- Test on known ground-truth systems ---
def logistic_map(r, x0, n, burn=200):
    x = x0
    for _ in range(burn): x = r*x*(1-x)
    out = np.empty(n)
    for i in range(n):
        x = r*x*(1-x)
        out[i] = x
    return out

rng = np.random.default_rng(0)
chaotic = logistic_map(3.9, rng.uniform(0.1,0.9), 800)
periodic = logistic_map(3.5, rng.uniform(0.1,0.9), 800)
decoy = iaaft_surrogate(chaotic, n_iter=300, rng=rng)

print("=== Testing the Chaos Decision Tree method on known ground truth ===\n")
classify(chaotic, rng, "Chaotic (r=3.9)")
classify(periodic, rng, "Periodic (r=3.5)")
classify(decoy, rng, "Surrogate decoy")

raw = np.loadtxt(str(_DATA / "laser_raw.txt"))
laser = raw[:800]
print()
classify(laser, rng, "Real laser data")
