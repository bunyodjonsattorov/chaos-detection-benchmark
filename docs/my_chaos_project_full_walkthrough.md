# My Chaos Project — Full Walkthrough (Plain English, Zero Assumed Knowledge)

A complete record of everything done so far, written so I can explain any
part of it to anyone who asks, without needing to look anything up.

---

## 1. The overall group project, in one paragraph

There's a well-known standard test set used across the world to check if
new algorithms are good at analyzing time series (an ordered list of
numbers, usually measured over time). But that standard test set mostly
tests "shape recognition" — like matching a heartbeat shape — not whether
an algorithm actually understands the deeper *behavior* of a system, like
whether it's chaotic, or drifting over time, or near some kind of tipping
point. Our team's job is to build a new, better test set that specifically
challenges algorithms on these deeper behavioral properties.

## 2. My specific piece of the project

My assigned topic is **chaos** — specifically, testing whether an algorithm
can tell the difference between a system that's genuinely chaotic and one
that just looks messy or irregular but isn't.

---

## 3. Everything I did, in order

### Stage 0 — Learning the basics

I started by understanding two ideas that everything else depends on:

- **Ground truth**: a label you can trust, because it's calculated
  mathematically, not guessed.
- **Confound**: an accidental, irrelevant difference between two groups of
  data that an algorithm could use as a cheap shortcut instead of learning
  the real thing you want it to detect.

### Stage 1 — Building the first (too easy) dataset

I used a very simple formula called the **logistic map**:

```
next_value = r * current_value * (1 - current_value)
```

You pick a number `r` (a "dial setting") and repeatedly feed the result
back into the formula. Depending on where you set that dial, the resulting
sequence either settles into a boring repeating pattern, or becomes
completely unpredictable — even though it's the exact same formula every
time. That unpredictable version is chaos.

I generated 200 example sequences: 100 with a chaotic dial setting, 100
with a non-chaotic setting. For the label, I didn't guess — I calculated
the **Lyapunov exponent** for each one, a number that mathematically
measures whether nearby starting points diverge over time (positive number
= chaos, negative = not chaos).

**Problem found**: this first dataset was too easy. A very basic method —
just checking the average value, the spread of values, and how correlated
each number was with the one right before it — could tell the two groups
apart almost perfectly. That's a bad benchmark, because the algorithm isn't
detecting chaos, it's cheating off a shortcut.

### Stage 2 — Making it genuinely hard (surrogate data)

To fix this, I used a real, published technique called **surrogate data
testing**. For every chaotic sequence, I built a "decoy twin" — a fake
sequence with the exact same average value, spread, and frequency content
(how smooth/jagged it looks overall), but with the real chaotic mechanism
scrambled out. This decoy is called a **surrogate**, and the specific
method I used to build it is called **IAAFT** (Iterative Amplitude Adjusted
Fourier Transform). It works by repeatedly alternating between two steps:
matching the frequency content, then matching the exact value distribution,
back and forth, until both match closely.

**Result**: on this harder version, the basic shortcut method dropped to
**53.5% accuracy** — essentially random guessing (50% would be a coin
flip). The benchmark was now genuinely hard.

### Stage 3 — Testing a smarter method (catch22)

I then tested a proper tool called **catch22**, built by our supervisor
Ben Fulcher. It's a set of 22 numbers, each specifically designed to
capture subtle patterns in time series data.

**Result**: catch22 scored **98.0% accuracy** on the same hard dataset. So
the benchmark was hard for a naive approach but solvable for a genuinely
smart, dynamics-aware one — exactly the property we want in a good
benchmark. The features that mattered most were about the *ordering
pattern* of ups and downs, and something called **time-reversal
asymmetry** — checking if the signal looks different played backwards.

### Stage 4 — Testing on real physical data (not just simulation)

Everything so far was simulated. Ben's instruction was not to just have
data generated for me — I needed to explore real-world examples. So I
found and downloaded a real dataset: the **Santa Fe laser dataset**, actual
measurements from a real laser experiment behaving chaotically (this is a
well-known, classic dataset in the field).

I ran the same test: naive shortcut method vs. catch22.

**Result**: naive method got **72.4%** (higher than on simulation — a real
signal is apparently harder to fully "disguise" than an artificial one),
catch22 got **94.2%**. The gap still held up on real data, which is an
important validation — it's not just a trick that works on clean simulated
numbers.

*(Side note: my first attempt at this had a bug — I accidentally created
an imbalanced dataset, with far more decoy examples than real ones, which
made the naive method look artificially good. I caught this and fixed it
by balancing the classes properly before trusting the result.)*

### Stage 5 — Researching chaos in other fields (biology and medicine)

Ben specifically wanted me to explore beyond physics. I researched how
chaos and "irregular-looking" signals are studied in other fields:

- **Cardiology**: for decades, some researchers believed heart rate
  variability (the tiny gaps between heartbeats) was chaotic. Rigorous
  surrogate testing has repeatedly failed to confirm this, and it remains
  a debated topic.
- **Neuroscience/epilepsy (EEG, brainwave data)**: similarly debated.
  Studies using surrogate testing found mixed evidence — some found
  epileptic seizure activity does show genuine nonlinear structure, while
  some healthy brain activity looks statistically similar to simple random
  processes.
- **Ecology**: researchers actually deliberately induced chaos in a real
  population of flour beetles by tuning their death rate in a lab
  experiment — one of very few cases of directly demonstrated chaos in a
  living population.

**Key realization**: surrogate data testing (the exact method I'd already
built) is literally the standard tool scientists use to settle these
real, unresolved debates across multiple fields.

### Stage 6 — Independently implementing a published method

I found and read a paper (Toker, Sommer & D'Esposito, 2020, published in
*Communications Biology*) that built a formal "Chaos Decision Tree
Algorithm" — a step-by-step method for taking any real signal and deciding
if it's random, periodic, or chaotic, robust to noisy real-world data. It
uses:

- **Permutation entropy**: a way of measuring how unpredictable the
  up/down ordering pattern of a signal is.
- The same surrogate-testing idea I'd already used, to first check "is
  this just randomness?"
- **The 0-1 test for chaos**: if the signal isn't random, this test
  produces a number between 0 and 1 (called `K`) — close to 1 means
  chaotic, close to 0 means periodic.

I built this method myself from scratch (not using a pre-made package) and
tested it on my known ground-truth examples.

**Result**: it correctly identified my chaotic sequence, my surrogate
decoy, and the real laser data. It incorrectly called my periodic sequence
"random" — but this exact failure mode is explicitly described as a known
limitation in the original paper, so it wasn't a mistake in my code, it's
a genuine, documented edge case of this type of method.

This gave me **two independent methods agreeing with each other** on the
same ground-truth cases — a real, meaningful cross-validation.

### Stage 7 — Testing on real biomedical data (EEG)

To close the loop between physics and biology, I found and downloaded a
real, publicly available dataset: EEG (brainwave) recordings from Bonn
University, with clear labels including healthy (eyes open) and seizure
activity.

I ran my full pipeline: naive shortcut method vs. catch22, plus the
independently-built Chaos Decision Tree method.

**Result**: naive method got **60.3%**, catch22 got **94.3%** — the same
pattern held a third time, now on real biological/medical data, not just
physics. The Chaos Decision Tree method gave genuinely mixed, uncertain
verdicts on the EEG data — which actually matches the real scientific
literature, since whether EEG shows true chaos is still an open question.
That's a *good* sign, not a failure — it means the method is being
appropriately honest about a genuinely uncertain case, rather than forcing
a false-confident answer.

---

## 4. Every dataset I used, summarized

| Dataset | Real or simulated | What it is | What I used it for |
|---|---|---|---|
| Logistic map (easy) | Simulated | Chaotic vs. non-chaotic sequences, simple version | First proof of concept — found it was too easy |
| Logistic map + surrogates (hard) | Simulated | Chaotic sequences + statistically matched decoys | Genuinely hard benchmark; naive 53.5% vs. catch22 98.0% |
| Santa Fe laser dataset | Real | Real chaotic laser intensity measurements | Validated the method generalizes beyond simulation; naive 72.4% vs. catch22 94.2% |
| Bonn EEG dataset | Real | Real brainwave recordings, healthy vs. seizure | Validated the method on real biomedical data; naive 60.3% vs. catch22 94.3% |

## 5. Every reading/paper I used, summarized

| Source | What it gave me |
|---|---|
| Middlehurst, Schäfer & Bagnall (2024), "Bake off redux" | Understanding of the existing UCR/UEA benchmark and its algorithm families — showed the "shape recognition" bias my project is meant to fix |
| Theiler et al. (1992) | The original method behind surrogate data testing |
| Schreiber & Schmitz (1996) | The specific IAAFT surrogate-building technique I implemented |
| Owens & Fulcher (2024), on PINUP (parameter inference) | Showed that "obvious" benchmark systems (including the logistic map!) are often too easy — same lesson I found myself, in a different sub-topic |
| Toker, Sommer & D'Esposito (2020), "A simple method for detecting chaos in nature" | The Chaos Decision Tree Algorithm — I implemented its core logic myself and validated it against my own ground truth |
| Cardiology/EEG chaos-debate research (various) | Showed that chaos detection is a real, unresolved question outside physics, and that surrogate testing is the standard tool used to investigate it |

## 6. Key concepts, in one line each

- **Time series**: an ordered list of numbers, usually over time
- **Ground truth**: a trustworthy, calculated (not guessed) correct label
- **Confound**: an accidental shortcut an algorithm could cheat with
- **Chaos**: deterministic behavior that's still extremely sensitive to starting conditions
- **Lyapunov exponent**: a number measuring how fast nearby trajectories diverge; positive = chaos
- **Surrogate**: a statistically matched decoy signal with the real structure destroyed
- **IAAFT**: the specific algorithm used to build a surrogate
- **catch22**: a set of 22 numbers built to detect subtle dynamical patterns in time series (built by our supervisor, Ben Fulcher)
- **Permutation entropy**: a measure of how unpredictable a signal's up/down ordering is
- **0-1 test for chaos**: a method producing a number (K) from 0 to 1 indicating how chaotic a signal is

## 7. Headline findings, all in one place

| Domain | Naive method accuracy | catch22 accuracy |
|---|---|---|
| Simulated (hardened, with surrogates) | 53.5% | 98.0% |
| Real laser data | 72.4% | 94.2% |
| Real EEG data | 60.3% | 94.3% |

**The consistent finding across all three**: naive statistical shortcuts
fail to detect real chaotic/dynamical structure, while a purpose-built,
dynamics-aware method succeeds — every time, across simulation, physics,
and biology. This is validated two ways: internally (naive vs. catch22)
and externally (my own implementation of a separate published method
agreeing with catch22 wherever ground truth was known).

## 8. What's still ahead

- Explaining more precisely why the naive method performs differently on
  real data vs. simulated data (still an open question)
- Testing closer to the actual boundary of chaos (where the Lyapunov
  exponent is close to zero), which should be much harder even for catch22
- Contributing this dataset and method as a reusable template for
  teammates working on other properties (non-stationarity, criticality,
  etc.)
