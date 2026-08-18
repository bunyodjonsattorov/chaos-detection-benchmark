# Chaos Detection Benchmark: Can Time-Series Features Spot Real Chaos?

A small, self-contained benchmark testing whether automated time-series
feature extraction (specifically [catch22](https://github.com/DynamicsAndNeuralSystems/catch22))
can detect genuine deterministic chaos — even when an adversarial control
strips away every linear statistical shortcut (mean, variance, power
spectrum, lag-1 autocorrelation) that a naive classifier could otherwise
exploit.

## TL;DR

| Classifier | Features used | Accuracy |
|---|---|---|
| Naive baseline | mean, std, lag-1 autocorrelation | 53.5% (≈ chance) |
| catch22 + random forest | 22 canonical time-series features | **97.0% ± 2.9%** |

catch22 recovers the chaotic/non-chaotic distinction almost perfectly using
features tied to genuine nonlinear structure (motif statistics, transition
matrices, time-reversal asymmetry) — on a dataset explicitly constructed so
that simple linear statistics carry *no* signal.

## Why this is a meaningful test

Distinguishing "chaotic" from "not chaotic" is trivial if the two classes
also happen to differ in mean, variance, or spectral content — a classifier
can cheat on those shortcuts without learning anything about the actual
dynamics. This project rules that out in two stages:

1. **Easy dataset** — chaotic vs. periodic logistic-map trajectories, both
   z-normalized, with randomized starting points and matched observation
   noise so amplitude/phase can't be memorized.
2. **Hard dataset** — every chaotic series is paired with a **surrogate
   decoy** generated via IAAFT (Iterative Amplitude Adjusted Fourier
   Transform; Schreiber & Schmitz, 1996). The decoy has the *identical*
   mean, variance, and power spectrum as its chaotic twin, but its phases
   are randomized — destroying the actual chaotic mechanism while leaving
   every linear statistic untouched. Only real nonlinear/deterministic
   structure can tell the twins apart.

If a feature set can still separate the classes on the hard dataset, it's
detecting genuine dynamics, not exploiting a confound.

## Repository structure

```
generate_chaos_dataset.py        # builds the "easy" chaotic vs. periodic dataset
generate_hard_chaos_dataset.py   # builds the "hard" dataset with IAAFT surrogate decoys
test_with_catch22.py             # extracts catch22 features and benchmarks a classifier
chaos_benchmark_logistic_map.csv # output of generate_chaos_dataset.py
hard_chaos_benchmark.csv         # output of generate_hard_chaos_dataset.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn pycatch22
```

## Usage

**1. Generate the easy dataset**

```bash
python3 generate_chaos_dataset.py
```

Creates `chaos_benchmark_logistic_map.csv` — 200 logistic-map series (100
chaotic, r ∈ {3.7, 3.9, 3.95, 3.99}; 100 periodic, r ∈ {2.8, 3.2, 3.45,
3.5}), each with its analytically computed Lyapunov exponent as ground
truth.

**2. Generate the hard dataset**

```bash
python3 generate_hard_chaos_dataset.py
```

Creates `hard_chaos_benchmark.csv` — 100 weakly chaotic series (r ∈ {3.60,
3.62, 3.65, 3.68}, near the edge of chaos) each paired with an IAAFT
surrogate decoy. Also prints a naive-feature baseline, which should land
close to random (~53–55%), confirming the adversarial control worked.

**3. Test catch22 on the hard dataset**

```bash
python3 test_with_catch22.py
```

Extracts 22 catch22 features per series and trains a random forest with
5-fold cross-validation. Should land around 95–98% accuracy, and prints
the most important individual features plus a confusion matrix.

## Things to try

- Change `chaotic_rs` in either generator script — values closer to 3.57
  (the accumulation point / edge of chaos) make the problem harder.
- Change `n_steps` — shorter series are harder to classify.
- Change `n_iter` in `iaaft_surrogate()` — fewer iterations makes the decoy
  a worse spectral/amplitude match, making the problem easier again.
- In `test_with_catch22.py`, restrict to a single feature (e.g.
  `CO_trev_1_num`, the time-reversal asymmetry statistic) to see how much
  of the ~97% accuracy comes from one feature alone.

## Background reading

- Theiler, J. et al. (1992). *Testing for nonlinearity in time series: the
  method of surrogate data.* Physica D.
- Schreiber, T. & Schmitz, A. (1996). *Improved surrogate data for
  nonlinearity tests.* Physical Review Letters.
- Lubba, C. H. et al. (2019). *catch22: CAnonical Time-series
  CHaracteristics.* Data Mining and Knowledge Discovery.

## License

MIT — see [LICENSE](LICENSE).
