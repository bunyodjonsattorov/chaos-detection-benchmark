# A benchmark for detecting dynamical structure in time series

Part of an Interdisciplinary Special Project (PHYS3888, University of Sydney)
building a new benchmark database for time series classification, aimed at
problems that cannot be solved by shape matching alone.

Supervised by A/Prof Ben Fulcher (School of Physics) with Eli Muller
(Faculty of Medicine and Health).

---

## The problem

The standard benchmark used to evaluate time series classification
algorithms consists largely of **pattern problems**: fixed-length windows
where classes differ by shape, and where that shape sits in the window
matters. Leaf outlines, gestures, heartbeats.

Physical systems produce **process problems**: the recording is an
arbitrary sample from a system that is already running. Where you started
recording carries no information. Classes differ by the *rule generating
the data*, not by any shape in the window.

Existing benchmarks contain very few of these, so it is not known how well
current algorithms handle them.

## What this repository contains

A confound-controlled dataset for one such property: detecting genuine
nonlinear / deterministic structure. Every positive signal is paired with
control signals constructed to match it on progressively more properties,
so that classification cannot succeed through simple shortcuts.

| Control type | Matches the positive on | Difficulty |
|---|---|---|
| `periodic` | broad signal character | easiest |
| `coloured_noise` | power spectrum | medium |
| `iaaft_surrogate` | power spectrum **and** amplitude distribution | hardest |

All series are z-normalised and resampled to 100 points, so mean, variance
and length carry no information.

## Dataset

`data/chaos_benchmark_v2.csv` — 1360 series of 100 points.

| Domain | System | Positive class | Controls |
|---|---|---|---|
| simulated | logistic map (r ≈ 3.57–3.62) | chaotic | iaaft, periodic |
| simulated | Lorenz (ρ = 28) | chaotic | iaaft, coloured noise |
| simulated | Mackey-Glass (τ = 17) | chaotic | iaaft, coloured noise, periodic |
| real | Santa Fe far-infrared laser | chaotic | iaaft, coloured noise |
| real | Bonn EEG, seizure | nonlinear (contested) | iaaft |
| real | Bonn EEG, healthy | nonlinear (contested) | iaaft |

80 series per class per system.

**Columns**: `id`, `system`, `domain`, `label`, `class_type`,
`label_source`, `lyapunov_exponent`, `t0`–`t99`.

### Label provenance

The `label_source` column records how each label was established, because
they are not equally certain:

- `analytic_lyapunov` — closed-form Lyapunov exponent (logistic map)
- `numerical_lyapunov` — computed via tangent-space (Lorenz: 1.04,
  literature ≈ 0.91) or trajectory separation (Mackey-Glass τ=17: 0.0095,
  literature ≈ 0.006)
- `published_domain_knowledge` — laser data, an established chaotic benchmark
- `contested_literature` — EEG. Whether EEG reflects deterministic chaos is
  **actively disputed**, so these are labelled `nonlinear_contested`, not
  `chaotic`, and should not be used as chaos ground truth
- `constructed_surrogate` / `constructed_stochastic` — the controls

## Results

Seven algorithm families, 5-fold cross-validation:

| Algorithm | Family | Accuracy |
|---|---|---|
| ROCKET | convolution | **0.899** |
| catch22 | feature-based | 0.875 |
| Linear AR(5) | linear baseline | 0.760 |
| Raw series + Random Forest | shape / interval proxy | 0.721 |
| Naive statistics | baseline | 0.695 |
| 1-NN Euclidean | distance-based | 0.674 |
| MLP on raw series | neural network | 0.633 |

### Difficulty ladder

| Control | Naive | Linear AR | ROCKET | catch22 |
|---|---|---|---|---|
| periodic | 0.942 | 0.970 | 1.000 | 0.995 |
| coloured noise | 0.774 | 0.700 | 0.990 | 0.975 |
| IAAFT surrogate | 0.643 | 0.751 | 0.867 | 0.822 |

### Findings

1. **The confound controls work.** Every method drops monotonically as the
   controls match more properties, and every method finds IAAFT surrogates
   hardest. The difficulty is engineered, not accidental.

2. **Shape-based and distance-based methods fail.** 1-NN Euclidean (0.674)
   and raw-series Random Forest (0.721) perform poorly. These rely on
   comparing shapes and on phase-dependent position, which is exactly what
   a process problem removes.

3. **Evidence of nonlinear structure.** IAAFT surrogates preserve
   everything a linear process can explain. A fitted AR(5) model reaches
   0.751 against surrogates while ROCKET reaches 0.867 and catch22 0.822.

### Limitations

- ROCKET, a mainstream convolution method not designed for dynamics,
  outperforms catch22. The claim "physics-aware methods win" is **not**
  supported; the supported claim is that methods capturing temporal
  structure succeed while simple statistics and shape comparison fail.
- The AR(5) baseline scores above chance on surrogates (0.751), higher
  than expected if surrogate matching were exact. IAAFT may not fully
  converge on 100-point windows. Unresolved.
- No nonlinear but non-chaotic positives, so "nonlinear" and "chaotic"
  are not separated.
- Real-data positives (laser, EEG) have no computed Lyapunov ground truth.
- Leave-one-system-out generalisation has not been re-run on this version.

## Usage

```bash
pip install -r requirements.txt
python download_data.py              # fetches third-party raw data
python src/build_final_dataset.py    # writes data/chaos_benchmark_v2.csv
python src/multi_algorithm_benchmark.py
```

Scripts expect to run from the repository root, and read/write raw data
under `data/`.

## Repository layout

```
src/     current pipeline
  build_final_dataset.py        dataset generator (seeded, reproducible)
  multi_algorithm_benchmark.py  seven-family comparison
  evaluate_final.py             catch22 vs naive baseline
  chaos_decision_tree.py        independent reimplementation of Toker et al. (2020)
  leave_one_domain_out_test.py  cross-system generalisation test

data/    dataset and raw inputs (EEG fetched by download_data.py)
docs/    concept glossary, full write-up, status briefing
archive/ superseded earlier versions, kept for history
```

## Data sources

Raw datasets are **not redistributed here**. `download_data.py` fetches
them from their public sources.

- Santa Fe laser — Hübner, U., Abraham, N. B. & Weiss, C. O.
  *Phys. Rev. A* **40**, 6354 (1989)
- Bonn EEG — Andrzejak, R. G. et al. *Phys. Rev. E* **64**, 061907 (2001)

## References

- Middlehurst, M., Schäfer, P. & Bagnall, A. Bake off redux: a review and
  experimental evaluation of recent time series classification algorithms.
  *Data Mining and Knowledge Discovery* (2024)
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J. D.
  Testing for nonlinearity in time series: the method of surrogate data.
  *Physica D* **58**, 77–94 (1992)
- Schreiber, T. & Schmitz, A. Improved surrogate data for nonlinearity
  tests. *Phys. Rev. Lett.* **77**, 635–638 (1996)
- Toker, D., Sommer, F. T. & D'Esposito, M. A simple method for detecting
  chaos in nature. *Communications Biology* **3**, 11 (2020)
- Lubba, C. H. et al. catch22: CAnonical Time-series CHaracteristics.
  *Data Mining and Knowledge Discovery* **33**, 1821–1852 (2019)
- Dempster, A., Petitjean, F. & Webb, G. I. ROCKET: exceptionally fast and
  accurate time series classification using random convolutional kernels.
  *Data Mining and Knowledge Discovery* **34**, 1454–1495 (2020)
- Owens, N. & Fulcher, B. Parameter inference from a non-stationary unknown
  process. *Chaos* **34**, 101501 (2024)
- Strogatz, S. *Nonlinear Dynamics and Chaos*. Westview Press (2014)

## Status

Work in progress, semester 2 2026. Results are preliminary and the dataset
is expected to change.
