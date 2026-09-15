# Chaos Dataset — Where Things Stand

Everything done so far on the chaos strand of the ISP, written so I can
explain it to teammates or answer questions without looking anything up.

---

## 1. The one-sentence version

I built a labelled dataset where each "real" signal is paired with fake
signals designed to look statistically identical, so that an algorithm can
only tell them apart by detecting genuine structure in the dynamics rather
than by using simple shortcuts.

---

## 2. Why the pairing matters

The whole point of our project is that existing benchmarks are too easy,
because an algorithm can succeed by recognising a shape rather than
understanding the system.

So for every real signal in my dataset, I generate a control signal that
matches it on everything simple. If an algorithm can still separate them,
it must be detecting something real.

The controls come in three levels of difficulty:

| Control type | What it matches the real signal on | How hard |
|---|---|---|
| `periodic` | broad character only | easiest |
| `coloured_noise` | the power spectrum (how jagged or smooth it is) | medium |
| `iaaft_surrogate` | power spectrum **and** the exact set of values | hardest |

All signals are z-normalised (mean 0, standard deviation 1) and resampled
to 100 points, so average, spread and length carry no information at all.

---

## 3. What is actually in the dataset

**1360 series, 100 points each.**

| Domain | System | Positive class | Controls used |
|---|---|---|---|
| simulated | logistic map (r ≈ 3.57–3.62) | chaotic | surrogate, periodic |
| simulated | Lorenz (ρ = 28) | chaotic | surrogate, coloured noise |
| simulated | Mackey-Glass (τ = 17) | chaotic | surrogate, coloured noise, periodic |
| real | Santa Fe far-infrared laser | chaotic | surrogate, coloured noise |
| real | Bonn EEG, seizure | nonlinear (contested) | surrogate |
| real | Bonn EEG, healthy | nonlinear (contested) | surrogate |

80 series per class per system.

**Columns**: `id`, `system`, `domain`, `label`, `class_type`,
`label_source`, `lyapunov_exponent`, then `t0`–`t99`.

---

## 4. How the labels were established (this matters)

The `label_source` column records provenance, because the labels are **not
equally certain**:

- `analytic_lyapunov` — closed-form Lyapunov exponent (logistic map)
- `numerical_lyapunov` — computed by tangent-space / trajectory-separation
  methods. Lorenz came out at 1.04 (literature ≈ 0.91); Mackey-Glass at
  0.0095 (literature ≈ 0.006). Close enough to confirm the numerics work.
- `published_domain_knowledge` — the laser data, a long-established
  chaotic benchmark
- `contested_literature` — the EEG. Whether EEG shows genuine chaos is
  **actively disputed**, so those are labelled `nonlinear_contested`, not
  `chaotic`. Do not treat them as chaos ground truth.
- `constructed_surrogate` / `constructed_stochastic` — the controls

---

## 5. Results: seven algorithm families tested

5-fold cross-validation on the full dataset.

| Algorithm | Family | Accuracy |
|---|---|---|
| ROCKET | convolution-based | **0.902** |
| catch22 | feature-based | 0.869 |
| Linear AR(5) | linear baseline | 0.774 |
| Raw series + Random Forest | shape/interval proxy | 0.709 |
| Naive statistics | baseline | 0.696 |
| 1-NN Euclidean | distance-based | 0.657 |
| MLP on raw series | neural network | 0.623 |

### The difficulty ladder

| Control | Naive | Linear AR | ROCKET | catch22 |
|---|---|---|---|---|
| periodic | 0.942 | 0.969 | 1.000 | 0.995 |
| coloured noise | 0.776 | 0.717 | 0.994 | 0.964 |
| IAAFT surrogate | 0.662 | 0.756 | 0.865 | 0.830 |

**Every method drops as the controls get harder, and every method finds
surrogates hardest.** That is the strongest evidence that the design works,
and it holds regardless of which algorithm you use.

---

## 6. The three findings worth reporting

**Finding 1 — the confound controls work.**
The difficulty ladder is consistent across all four method types. Each
control matches the real signal on more properties, and each is measurably
harder. That is engineered difficulty, not accidental.

**Finding 2 — shape-based and distance-based methods fail here.**
1-NN Euclidean (0.657) and raw-series Random Forest (0.709) do badly.
These rely on comparing shapes and on where features sit in the window,
which is exactly what a process problem removes. This is direct support
for the pattern-versus-process argument.

**Finding 3 — the nonlinearity evidence.**
IAAFT surrogates preserve everything a *linear* process can explain. A
properly fitted AR(5) model gets 0.756 against surrogates, while ROCKET
gets 0.865 and catch22 0.830. That gap is evidence of genuinely nonlinear
structure being detected.

---

## 7. Things to be honest about

**ROCKET beats catch22 (0.902 vs 0.869).**
ROCKET is a mainstream convolution method, not a dynamics tool. So the
simple story "physics-aware methods win" does not hold as stated. The more
accurate version is: *methods that capture temporal structure succeed,
while simple statistics, distance measures and raw shape comparison fail.*

**The linear baseline is not at chance.**
AR(5) reaching 0.756 on surrogates is higher than it theoretically should
be if the surrogate matching were perfect. Likely the IAAFT does not fully
converge on short 100-point windows. This needs checking.

**Generalisation is untested in the current numbers.**
An earlier run on the older dataset showed catch22 dropping from 0.87
within-distribution to 0.59 on an unseen system. That test has not been
re-run on this larger dataset.

**Do not mix numbers across dataset versions.**
The 53% vs 98% figures came from the earlier logistic-map-only dataset.
On this harder combined dataset nothing reaches 98%.

---

## 8. Likely questions and short answers

**"Is this testing chaos or nonlinearity?"**
Strictly nonlinearity. IAAFT preserves everything a linear process could
produce, so separating real from surrogate shows nonlinear structure.
Chaos is the *source* of that structure in my positives, verified by
Lyapunov exponents, but the test itself is not chaos-specific.

**"Could a nonlinear but non-chaotic system pass the test?"**
Yes. That is a genuine gap — there are currently no nonlinear non-chaotic
positives in the dataset.

**"How did you make the fakes?"**
IAAFT (Iterative Amplitude Adjusted Fourier Transform), a published method
from Schreiber & Schmitz 1996. It keeps the exact set of values and the
power spectrum while randomising the phase relationships that carry the
deterministic structure.

**"How do you know the labels are right?"**
For the simulated systems, I computed Lyapunov exponents and checked them
against published values. For the laser it is an established benchmark.
For EEG I do not claim chaos, which is why those are labelled contested.

**"Why 100 points?"**
Short windows make the problem harder and are more realistic for real
recordings. It is also short enough that all systems can be resampled to a
common length, so everything is directly comparable.

**"Is the dataset big enough?"**
1360 series, 80 per class per system. Reasonable for a proof of concept.
A published benchmark would want more, and more systems.

---

## 9. What comes next

- Re-run the leave-one-system-out generalisation test on this dataset
- Investigate why the linear AR baseline performs above chance on
  surrogates (possible surrogate convergence issue on short windows)
- Add nonlinear but non-chaotic positives, to separate "nonlinear" from
  "chaotic" properly
- Check whether the same paired design transfers to the other four
  properties in the group's database

---

## 10. Files

- `chaos_benchmark_v2.csv` — the dataset
- `build_final_dataset.py` — generator, fully seeded and reproducible
- `evaluate_final.py` — catch22 vs naive evaluation
- `multi_algorithm_benchmark.py` — the seven-family comparison
- `BENCHMARK_README.md` — full documentation

Reproduce with:
```
pip install numpy pandas scikit-learn pycatch22 sktime numba
python3 build_final_dataset.py
python3 multi_algorithm_benchmark.py
```
