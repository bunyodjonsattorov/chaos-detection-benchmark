# Chaos Benchmark v2

A confound-controlled time series classification benchmark for detecting
genuine nonlinear/deterministic structure, built for the PHYS3888 ISP
Group C interdisciplinary dynamics database.

## Files
- `chaos_benchmark_v2.csv` — the dataset (1360 series x 100 points)
- `build_final_dataset.py` — generator (fully reproducible, seeded)
- `evaluate_final.py` — benchmark evaluation script

## Task
Binary classification: does this series carry genuine nonlinear /
deterministic structure, or is it a statistically-matched lookalike?

## Design
Every positive example is paired with negatives that match it on
progressively more properties, so simple shortcuts are removed:

| Negative class | Matches the positive on | Purpose |
|---|---|---|
| `periodic` | broad signal character | easiest control |
| `coloured_noise` | power spectrum | removes spectral shortcuts |
| `iaaft_surrogate` | power spectrum **and** amplitude distribution | hardest control; only nonlinear phase structure differs |

All series are z-normalised and resampled to 100 points, so mean, variance
and length carry no information.

## Contents

| Domain | System | Positive class | Negatives |
|---|---|---|---|
| simulated | logistic map (r≈3.57–3.62, near onset) | chaotic | iaaft, periodic |
| simulated | Lorenz (ρ=28) | chaotic | iaaft, coloured noise |
| simulated | Mackey-Glass (τ=17) | chaotic | iaaft, coloured noise, periodic (τ=10) |
| real | Santa Fe far-infrared laser | chaotic | iaaft, coloured noise |
| real | Bonn EEG, seizure | nonlinear (contested) | iaaft |
| real | Bonn EEG, healthy | nonlinear (contested) | iaaft |

80 series per class per system.

## Label provenance
The `label_source` column records how each label was established, because
they are **not** equally certain:

- `analytic_lyapunov` — closed-form Lyapunov exponent (logistic map)
- `numerical_lyapunov` — computed via tangent-space / trajectory-separation
  (Lorenz: 1.04, literature ≈0.91; Mackey-Glass τ=17: 0.0095, literature ≈0.006)
- `published_domain_knowledge` — laser data, an established chaotic benchmark
- `contested_literature` — EEG. Whether EEG reflects deterministic chaos is
  **actively disputed**. These are labelled `nonlinear_contested`, not
  `chaotic`, and should not be treated as chaos ground truth.
- `constructed_surrogate` / `constructed_stochastic` — negatives we built

## Baseline results

| Test | Naive (mean/std/lag-1) | catch22 |
|---|---|---|
| 5-fold CV, all systems | 0.579 | 0.869 |

Difficulty ladder (positives vs one negative class at a time):

| Negative | Naive | catch22 |
|---|---|---|
| periodic | 0.912 | 0.995 |
| coloured noise | 0.653 | 0.964 |
| IAAFT surrogate | 0.510 | 0.830 |

Leave-one-system-out (train on other systems, test on an unseen one):

| Held-out system | catch22 |
|---|---|
| Lorenz | 0.679 |
| laser | 0.667 |
| Mackey-Glass | 0.662 |
| EEG seizure | 0.588 |
| EEG healthy | 0.500 |
| logistic map | 0.438 |
| **average** | **0.589** |

## Headline findings
1. **The negatives form a clean difficulty ladder.** IAAFT surrogates are far
   harder than periodic controls (naive drops 0.912 → 0.510), confirming the
   confound controls work as intended.
2. **catch22 scores 0.869 within-distribution but 0.589 across systems**, and
   below chance on the held-out logistic map. It is largely learning
   system-specific fingerprints rather than a transferable notion of chaos.
3. Cross-system generalisation, not within-distribution accuracy, is the
   discriminating test — and no method tested here passes it.

## Known limitations
- Binary task only; no graded difficulty by Lyapunov magnitude yet
- Real-data positives (laser, EEG) have no computed Lyapunov ground truth
- Only one real physical system (laser) and one real biological system (EEG)
- Surrogate quality on short 100-point windows is not separately validated

## Reproducing
```
pip install numpy pandas scikit-learn pycatch22
python3 build_final_dataset.py     # writes chaos_benchmark_v2.csv
python3 evaluate_final.py          # reproduces the tables above
```
