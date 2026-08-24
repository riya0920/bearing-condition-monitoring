# ML-3 — Multivariate Sensor Anomaly Detection & Health Indexing

**Status: complete.** The physics feature layer, the health index with a
hysteretic alarm state machine, the three-way detector comparison at a matched
false-alarm budget, and the lead-time-vs-false-alarm operating curve are built.
The fleet dashboard, the process-side (Tennessee-Eastman-style) multivariate case,
and real bearing data are not.

```bash
python run_cm.py            # ~90 s
python run_cm.py --quick    # smaller fleet
python run_cm.py --report-only
```

## Data provenance — read this first

**The vibration is synthesised, not measured.** `src/bearing.py` generates it:
impulse trains at the kinematic fault frequencies, each exciting an exponentially
decaying structural resonance, with rolling-element slip, cycle-to-cycle shaft
speed jitter, 1× and 2× machine content, and a degradation trajectory with a knee.

It is **not CWRU and not IMS**, and no number here is comparable to a paper using
those. What the simulation buys is ground truth — fault type, onset cycle, failure
cycle, and a set of assets that genuinely never fail — which is what turns
"detected 79 cycles early" and "0.33 false alarms per asset-life" into scoreable
claims instead of a screenshot of red dots.

## The physics, because that is the differentiator

Fault frequencies come from bearing geometry (SKF 6205-like: 9 elements, ⌀7.94 mm
ball, ⌀39.04 mm pitch, 0° contact angle), not from feature selection:

| | orders of shaft speed | at 29.95 Hz |
|---|---|---|
| BPFO (outer race) | 3.5848× | 107.4 Hz |
| BPFI (inner race) | 5.4152× | 162.2 Hz |
| BSF (rolling element) | 2.3568× | 70.6 Hz |
| FTF (cage) | 0.3982× | 11.9 Hz |

**Envelope analysis, not an FFT of the raw signal.** A defect impulse is broadband
and small; what the accelerometer records is a structural resonance being rung by
it. The fault information is in the *amplitude modulation* of a high-frequency
carrier, not in any line at BPFO — and the low-frequency end of the raw spectrum is
dominated by 1× imbalance, which is 10–100× larger than anything the bearing is
doing. So: band-pass around the resonance → Hilbert envelope → FFT the envelope.

### The collision, and how it is resolved

For this geometry, **BPFO × 3 = 10.754× shaft and BPFI × 2 = 10.830× shaft — 0.70%
apart**, which is inside the slip tolerance any real detector must allow. Harmonic
energy alone therefore *cannot* separate an outer-race fault from an inner-race
one on this bearing.

The separator is **sidebands**, and it is geometry rather than statistics: an outer
race defect is stationary in the load zone, so its impulse train has constant
amplitude and a clean comb at BPFO. An inner race defect rotates with the shaft,
carrying the defect in and out of the load zone once per revolution — the impulse
train is amplitude-modulated at shaft rate, which puts sidebands at BPFI ± 1× shaft.

That is why `diagnose_with_baseline()` uses sidebands as a **tie-break** and not as
an override. An earlier version let any sideband energy above threshold win
outright, and on a severe outer-race fault it flipped the diagnosis to BPFI at the
very end of life: a strong fault lifts the whole envelope floor, so BPFI's
exceedance crosses the detection threshold on leakage alone and its sideband ratio
then gets computed on noise. Sidebands are evidence about *which of two contenders*
it is, not evidence that there are two.

## Results

**Diagnosis** — 98% correct race on developed faults; 591/600 healthy cycles
correctly called healthy:

| true fault | healthy | BPFO | BPFI | BSF | indeterminate |
|---|---|---|---|---|---|
| BPFO | 362 | **235** | 0 | 1 | 2 |
| BPFI | 353 | 0 | **245** | 0 | 2 |
| BSF | 375 | 0 | 0 | **224** | 1 |
| healthy | **591** | 3 | 1 | 5 | 0 |

**The operating curve** — the deliverable of the whole project:

| health-index threshold | median lead (cycles) | P05 lead | false alarms per healthy asset-life | assets missed |
|---|---|---|---|---|
| 95 | 199 | 178 | 8.00 | 0 |
| 90 | 102 | 81 | 8.00 | 0 |
| 85 | 78 | 74 | 1.33 | 0 |
| **80** | **76** | **70** | **0.00** | **0** |
| 70 | 70 | 67 | 0.00 | 0 |
| 50 | 67 | 60 | 0.00 | 0 |

At a budget of ≤1 false alarm per asset-lifetime: **76 cycles of median warning,
70 at P05, nothing missed.** A lead time quoted *at* a false-alarm budget is the
operational contract; a lead time quoted without one is a number chosen after
seeing the answer.

**Alarm flapping: 0** across the whole fleet. Hysteresis (separate enter and exit
thresholds per state) plus 3-of-5 persistence is what produces that. This is the
answer to "operators disabled the last vendor's system in six weeks": the vendor
optimised detection and never measured flapping.

**The detector bake-off, at a matched false-alarm budget:**

| detector | median lead | worst-case lead | false alarms per asset-life | detected |
|---|---|---|---|---|
| Hotelling T² (statistical) | 79 | 75 | 0.33 | 9/9 |
| IsolationForest (ML) | 79 | 72 | 0.33 | 9/9 |
| Autoencoder (deep) | 78 | 68 | 0.33 | 9/9 |

**There is no winner, and that is the result.** A 1-cycle spread across 9
trajectories is the sampling noise of a median over 9 numbers. What I would ship is
the T²: thirty lines, no training step and therefore no retraining pipeline, a
score that decomposes into per-feature contributions so the alarm can be explained
to the operator who receives it, and failure modes that are a hundred years old and
documented. The reason deep does not win is not that deep is bad — it is that
**the features already contain the physics**. Once envelope energy at BPFO is a
column, the remaining problem is "is this column unusually large", which is a job
for a covariance.

## Two things I got wrong, kept because they are the interesting part

**1. The demodulation band cannot be learned from healthy data.** The first version
chose it adaptively by kurtogram over each asset's baseline period, and every asset
came back with the same meaningless 500–1062 Hz band. The reason is physical: the
baseline period is healthy, a healthy bearing produces no impulses, and a kurtogram
with nothing impulsive to find returns whichever band the noise favoured.

The band is a property of the *structure* — the housing resonance that defect
impulses ring — established by an impact test at commissioning. So it is
configuration. What a kurtogram is good for is auditing it, and §2 of RESULTS.md
shows the asymmetry: on failing assets the kurtogram recovers the commissioned band
from **degraded** data far more reliably than from healthy data.

**2. Absolute thresholds do not work, and the numbers say why.** A healthy
bearing's envelope-energy ratio in this simulation is already ~12× the broadband
floor, because the peak of a noisy spectrum over a search window is several times
its median by construction. My first threshold of "4× the floor" fired on
everything and classified healthy bearings as inner-race faults with total
confidence. Every feature is now expressed as **exceedance over that asset's own
healthy baseline** in robust (IQR) units — which is also what makes the cold-start
policy honest: an asset with too little history gets the `LOW_CONFIDENCE` state
explicitly, not silence and not a number pretending to be calibrated.

## The claim that did not survive measurement

The textbook says kurtosis rises before RMS on a degrading bearing. Measured here:
**median lead 14 cycles, positive on 8 of 9 assets** — weaker than the textbook
implies, and I report it that way rather than trimming the table.

The reason is a limitation of my simulator, not a refutation of the physics: the
degradation model raises a single amplitude parameter smoothly, so the impulse
train gains energy and impulsiveness together. A real bearing passes through a
phase where one small spall produces sharp impulses at almost constant total
energy, and that phase is where the kurtosis lead is won. What actually delivers
the lead here is the envelope band energy at the fault frequency, which is
physics-located rather than a generic shape statistic.

## Built in the second pass — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)

`python extend.py` — a larger fleet (18 failing + 6 healthy, up from 9 + 3) plus
three gaps this README previously named:

- **The per-alarm "why" panel.** The first build recommended shipping Hotelling T²
  *partly because its score decomposes into per-feature contributions* — and then
  did not decompose it. It does now, exactly: with `d = x − μ`, feature *j*
  contributes `d_j·(S⁻¹d)_j` and the contributions **sum to T² identically**,
  unlike SHAP or occlusion which approximate. **The leading contributor names the
  correct bearing race on 100% of failing assets**, so the alarm card carries a
  diagnosis rather than a score.
- **Cold start, exercised.** `LOW_CONFIDENCE` and the fleet prior existed and were
  never run. Three states now, and the middle one is what most systems omit:
  `PROVISIONAL` widens thresholds by **1 + 1/√(2(n−1))** — the sampling error of a
  standard-deviation estimate, not a tuning knob — so the detector becomes less
  trigger-happy exactly when its baseline is least trustworthy.
- **The P-F interval, quantified.** The vocabulary was used correctly and the
  number was never produced. It is the quantity that sets the inspection interval,
  and the distribution matters more than usual: **setting the interval from the
  mean rather than the P10 makes it 1.5× too long** (12 cycles against 8), missing
  the fast half of the failures by construction.

## Completed in the third pass — real data, see [docs/REAL_DATA.md](docs/REAL_DATA.md)

```bash
python fetch_cwru.py       # ~120 MB from the CWRU Bearing Data Center
python validate_cwru.py
```

The README's number-one gap was "no real data — everything is synthesised", and it
was the largest threat to every claim here: the envelope pipeline was designed
against a simulator I wrote and then validated against the same simulator. **It
could not have failed.** This is the non-circular test.

40 files of real accelerometer data, the same SKF 6205-2RS bearing
`src/bearing.py`'s geometry was written for, 424 non-overlapping
snapshots at four motor loads and three fault sizes. Nothing retuned.

**Run as designed: 36.8% correct race.** That is a
failure, and the diagnosis is worth more than the number would have been.

| pipeline | correct race | healthy called healthy |
|---|---|---|
| as designed (synthetic thresholds) | 36.8% | 43.8% |
| sideband rule deleted | 67.9% | 43.8% |
| sideband statistic **fixed** + 2×BSF added | 68.4% | 31.2% |
| **+ healthy gate as baseline exceedance** | **68.4%** | **87.5%** |

Scored on 20 files held out from the 20 used
for any calibration, **split by file rather than by snapshot** — snapshots from
one recording share a bearing, a mounting, a load and a speed.

### Two real bugs, both now fixed in `src/features.py`

**The sideband tie-break was inverted** — and it is the piece of reasoning this
README presents as its cleverest. `sideband_ratio` computed `side / (2 × center)`,
normalising by the carrier, so a genuine inner-race fault makes BPFI enormous and
drives its own statistic *down*. Measured medians: **0.256
on inner-race faults against 0.840 on everything else.**
Backwards. It cleared the synthetic threshold on
88% of snapshots and
overrode BPFO lines six times larger than BPFI. **A ratio normalised by its own
carrier inverts when the carrier is the signal.** The simulator hid it by
injecting sidebands in proportion to the fault, so numerator and denominator grew
together.

**The healthy gate was an absolute constant** of 4.0, and real healthy bearings
sit at 3.3–4.1, straddling it.

Both are the same mistake, and **this README already contained the lesson** —
*"absolute thresholds do not work... express every feature as exceedance over that
asset's own healthy baseline"*. It was applied to the energy features and not to
these two. Writing a lesson down is not the same as having applied it everywhere
it holds, and the only reliable way to find where it was missed is a dataset you
did not write.

### Ball faults remain unsolved, at 19%

A rolling element strikes the outer race, then half a ball revolution later the
inner race — two impacts per rotation, so the dominant line is 2×BSF rather than
BSF. The project computed BSF only. Adding 2×BSF is a genuine fix to the fault
model and it moved ball accuracy from 16% to 19%: **almost nothing.** Ball-fault
energy does not concentrate on a line the way race-fault energy does, so a
line-energy detector is close to the wrong instrument. That is the honest state,
not a tuning backlog.

### What CWRU does and does not validate

**Validated.** Envelope analysis at frequencies computed from bearing geometry —
not learned, not selected — transfers to real accelerometer data at a sampling
rate it was not designed for. Inner race is diagnosed 120/120.

**Not validated, and it is the larger half of this project.** CWRU has no
degradation trajectory: each file is one bearing at one fault size, and the three
sizes are three different bearings rather than one bearing over time. So **every
prognostic number in RESULTS.md stays simulated** — the health index, the
76-cycle lead time, the operating curve, the false-alarms-per-asset-life figure,
the detector bake-off. Nothing here supports them.

**And CWRU is the easy version of diagnosis.** Spark-eroded pits are clean,
geometric and single-point. Natural spalling is rough, spreads along the race, and
smears the signature across a band instead of putting it on a line. This accuracy
is an upper bound on the accuracy against grown faults.

## What is NOT built

1. **No degradation trajectory in the real data.** CWRU cannot test prognosis at
   all — see above. Doing so needs a run-to-failure set such as IMS or FEMTO.
2. **Ball faults at 19%**, after a correct physics fix that barely moved them. A
   line-energy detector is close to the wrong instrument for them.
3. **No process-side multivariate case.** The spec asks for a Tennessee-Eastman
   style dataset with residual-based features from per-signal regression models.
   Not built — this project is bearings only.
4. **No dashboard.** The per-alarm T² contribution decomposition exists and is
   measured (it names the correct race on 100% of failing assets); nothing
   renders it.
5. **No speed-varying case.** Shaft speed is measured per file and the CWRU rig
   runs at constant speed. No run-up, no coast-down, no order tracking — which is
   where fixed-frequency band energy stops working entirely.
6. **The synthetic fleet is still 9 failing + 3 healthy assets**, so every median
   in RESULTS.md is over 9 numbers and every false-alarm rate over 3.

## Layout

```
src/bearing.py     geometry, fault frequencies, run-to-failure and healthy simulators
src/features.py    time domain, spectral kurtosis band selection, envelope spectrum, sidebands
src/health.py      per-asset baselines, health index, hysteretic alarm state machine, diagnosis
src/detectors.py   T2 / IsolationForest / autoencoder at a matched false-alarm budget
run_cm.py          orchestration; writes docs/RESULTS.md
```
