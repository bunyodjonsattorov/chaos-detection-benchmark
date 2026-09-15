# Project Concepts Glossary — Plain English, Zero Assumed Knowledge

A guide for explaining our benchmark project to teammates, written with math/CS
analogies instead of physics jargon.

---

## Part 1 — The absolute basics

**Time series**
A list of numbers in order, usually measured over time. Example: your room
temperature recorded every minute. In code terms: just an array/list of numbers.

**Dataset**
A collection of examples used to test or train something. In our case: a
collection of time series, each with a label attached.

**Label / ground truth**
The "correct answer" attached to an example. If we say a time series is
"chaotic," that's its label. "Ground truth" means we're confident this label
is actually correct — not guessed, but proven or calculated.

**Algorithm / classifier**
A program that looks at a time series (without being told the label) and
tries to guess what category it belongs to. Think of it as a function:
`data in -> guess out`.

**Feature**
A single number that summarizes something about a time series — e.g., "the
average value," or "how bumpy it is." Algorithms often work by computing a
bunch of features first, then making a decision based on those numbers
instead of the raw data.

**Benchmark**
A standardized test — a fixed set of problems with known answers — that lets
you fairly compare different algorithms against each other.

---

## Part 2 — Why this project exists

There's an existing standard benchmark (the **UCR/UEA archive**, ~150
datasets) that the time-series research community uses to test new
algorithms. Problem: it's mostly made of "shape recognition" tasks (like
matching an ECG heartbeat shape), which don't really test whether an
algorithm understands **how a system behaves over time** — only whether it
can spot a familiar shape.

**Our job**: build a new benchmark that specifically tests whether
algorithms can detect subtle *behavioral* structure, not just shapes.

---

## Part 3 — "Process" vs "Pattern" (the core distinction)

This is the single most important idea in the whole project.

- **Pattern problem**: a fixed, finite chunk of data where *where* something
  happens matters. Like recognizing a hand-drawn digit — the shape and its
  position on the page matter.
- **Process problem**: the data is a snapshot of an ongoing system governed
  by some underlying rule. The system keeps running whether or not you're
  watching. *Where* you started measuring shouldn't matter — what matters is
  the *rule* driving it.

**Analogy**: Pattern = recognizing a specific photo. Process = recognizing
the *rules of the video game* just from watching a short clip, regardless of
which frame you started watching from.

We want **process** problems — ones that test understanding of the
underlying rule, not shape memorization.

---

## Part 4 — The 11 candidate dynamical properties, explained simply

Each of these is a *type of behavior* a system's rule can produce. Any of
these could become a sub-topic for a teammate.

| Term | Plain-English meaning |
|---|---|
| **Chaos / Lyapunov exponent** | A rule is deterministic (no randomness) but two nearly-identical starting points end up wildly different after enough steps. Lyapunov exponent = a number measuring *how fast* that difference grows. Positive = chaotic, negative = not. |
| **Nonlinearity** | The rule isn't of the simple form `output = a × input + b`. Small tweaks to input can cause disproportionate changes in output. |
| **Non-stationarity** | The statistics of the data (average, spread, etc.) change over time, instead of staying constant. Like data drift in machine learning. |
| **Intermittency** | The signal switches between calm stretches and bursty stretches, unpredictably. |
| **Criticality** | The system sits at a "tipping point" where triggers of any size can cause effects of any size — no typical/average effect size. |
| **Non-Gaussianity** | The values don't follow a normal/bell-curve distribution — could be skewed, or have unusually frequent extreme values. |
| **Time reversibility** | If you played the sequence backwards, would it look statistically the same? Most real dynamical processes are NOT reversible — they look "wrong" backwards. |
| **Non-chaotic strange attractors** | The long-term trajectory has a complicated, intricate geometric shape, but nearby points stay nearby over time (unlike chaos). |
| **Fractal / multifractal scaling** | The statistical pattern looks similar at different zoom levels (self-similarity). "Multifractal" = different zoom levels behave differently rather than all the same way. |
| **Phase-amplitude coupling** | A fast rhythm's "loudness" depends on where a slow rhythm currently is in its own cycle — like amplitude modulation (AM radio). |
| **Attractor dimension** | A number (often not a whole number!) describing how much "space" the system's long-term behavior fills. Low = collapses to a simple curve. High = complex, space-filling. |

---

## Part 5 — Worked example: the Logistic Map (our chaos experiment)

The simplest possible chaos-generating rule:

```
next_value = r * current_value * (1 - current_value)
```

You pick one number, `r` (a "dial setting"), and repeatedly feed the output
back in as the next input.

- `r ≈ 2.5` → settles into a fixed, boring value
- `r ≈ 3.2` → bounces between a couple of values forever (periodic)
- `r ≈ 3.9` → looks completely random, but is 100% deterministic — this is
  chaos

**Why this is useful**: one formula, one number to tweak, and you get
perfectly labeled example data — you *know* the truth because you wrote the
rule.

---

## Part 6 — Confounds (the "biggest challenge" from meeting 1)

**Confound**: an accidental, irrelevant difference between your example
groups that an algorithm could use as a shortcut instead of learning the
real thing you wanted it to detect.

**Example**: if your "chaotic" examples happen to have bigger average values
than your "non-chaotic" ones, an algorithm could get 100% accuracy just by
checking "is the average big?" — without ever detecting chaos at all. That
would be a false, misleading benchmark.

**Fix**: force all groups to match on the easy/shallow statistics (same
average, same spread, etc.), so the algorithm is forced to find the *real*
underlying difference.

---

## Part 7 — Surrogate data testing (how we made our benchmark genuinely hard)

**Surrogate**: a fake, decoy version of a real signal, built to match it
statistically while removing the "real" structure you're trying to detect.

**Why it's useful**: it's not enough to match simple stats like average and
spread — two very different processes can still share those. Surrogates go
further: they match even the frequency content (how jagged/smooth the
signal is overall) while destroying the actual deterministic mechanism.

**IAAFT** (Iterative Amplitude Adjusted Fourier Transform): the specific
recipe we used to build surrogates. In plain terms: it repeatedly
scrambles the "timing/ordering logic" of a signal while forcing it to keep
the exact same value distribution and frequency content as the original.
Result: a decoy that "looks" the same by every simple measure, but has no
real chaos underneath.

This is a real, published, respected technique (Schreiber & Schmitz, 1996)
— not something we invented.

---

## Part 8 — catch22 / hctsa (Ben's tools)

**catch22**: a curated set of 22 numbers ("features"), each specifically
chosen because it's useful for telling different kinds of time series
apart — including subtle dynamical ones like the properties in Part 4. Built
by our supervisor, Ben Fulcher.

**hctsa**: a much bigger version of the same idea (thousands of features
instead of 22) — also built by Ben.

**Why they matter to us**: they're specifically designed to detect the kind
of subtle structure our project cares about, as opposed to generic/simple
statistics.

---

## Part 9 — Our actual experiment, recapped simply

1. Generated 200 time series from the logistic map: half genuinely chaotic,
   half not — with **known, calculated ground truth** (the Lyapunov
   exponent).
2. Made sure both groups had matching average value and spread, so a lazy
   algorithm couldn't cheat.
3. Tested a "naive" method (just average, spread, and lag-1 correlation) →
   near-perfect accuracy. **Too easy.**
4. Built a harder version: for every chaotic series, created a **surrogate
   decoy twin** with identical spectrum/statistics but no real chaos.
5. Re-tested the naive method on the harder version → **53.5% accuracy**
   (~random guessing). Now genuinely hard.
6. Tested catch22 (Ben's tool) on the same hard version → **98.0%
   accuracy**. Proves genuine dynamics-aware methods succeed where naive
   ones fail — the exact result our whole project is trying to demonstrate.

---

## Part 10 — Quick glossary cheat-sheet

| Term | One-line definition |
|---|---|
| Time series | An ordered list of numbers, usually over time |
| Ground truth | A verified, trustworthy correct answer/label |
| Confound | An accidental shortcut an algorithm could cheat with |
| Process vs pattern | Behavior-of-a-rule vs shape-of-a-snapshot |
| Chaos | Deterministic but extremely sensitive to starting conditions |
| Lyapunov exponent | Number measuring how fast nearby paths diverge; positive = chaos |
| Nonlinearity | Output not proportional to input |
| Non-stationarity | Statistics change over time |
| Surrogate | A statistically-matched decoy with the real structure removed |
| IAAFT | The specific method used to build surrogates |
| catch22 / hctsa | Curated feature sets built by Ben Fulcher for detecting dynamical structure |
