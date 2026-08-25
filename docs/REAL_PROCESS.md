# The process comparison, on data I did not generate

The README named this as the project's remaining circularity and said the residual model's win was *the result most likely to be an artefact of a generator built from linear relationships*. It runs here on the Tennessee Eastman benchmark and on SKAB, with the detectors unchanged — the only new code is the loading and the calibration.

**Tennessee Eastman is still a simulation**, and calling it real data would be the overclaim this project keeps catching. What it is: a simulation *somebody else built*, of a process I did not design, with faults I did not choose, that the literature has used as its reference for thirty years. That breaks the circularity — the monitor cannot have been tuned to a generator I never saw. **SKAB is a real rig**: a water circulation loop with faults induced by hand.


## The first version of this was wrong, and wrong flatteringly

Setting each detector at its own 99% limit and comparing detection delays produced this: **all five detectors found all ten faults**, including the three the literature agrees are close to undetectable, at false-alarm rates of four to six percent. That is not a comparison of methods, it is a comparison of thresholds — the loosest detector wins every race and pays for it in a column the delay table does not show.

Everything below sets thresholds so that every detector runs at the **same false-alarm budget**, measured as alarm episodes per 1000 samples on `d00_te`, a normal run held out from fitting. And because a ranking that holds at one operating point is not a ranking, the budget is swept.


## Tennessee Eastman

52 variables, 500 normal training samples, 960 held-out normal samples for calibration, 10 fault runs, fault injected at sample 160 of 960 (3-minute sampling).

### Per fault, at 1 false alarm per 1000

| fault | univariate | T² | SPE | T² or SPE | residual | DPCA T² | DPCA SPE | DPCA T²|SPE | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | 5 | 7 | 4 | 4 | **3** | 6 | 5 | 5 | A/C feed ratio step, B composition constant |
| 03 ⚠ | **52** | never | 539 | 539 | 618 | never | 570 | 570 | D feed temperature step (widely reported as  |
| 04 | **2** | 3 | **2** | **2** | **2** | 77 | **2** | **2** | reactor cooling water inlet temperature step |
| 05 | **2** | **2** | 3 | **2** | **2** | 3 | 3 | **2** | condenser cooling water inlet temperature st |
| 06 | **2** | 10 | **2** | **2** | **2** | 8 | **2** | **2** | A feed loss (step) |
| 09 ⚠ | 232 | **6** | 490 | 490 | 255 | 734 | never | never | D feed temperature random variation (near-un |
| 11 | **7** | 10 | 9 | 9 | **7** | 12 | 8 | 8 | reactor cooling water inlet temperature, ran |
| 13 | 45 | **38** | 44 | 44 | 46 | 40 | 40 | 40 | reaction kinetics, slow drift |
| 14 | **2** | **2** | 4 | 3 | 3 | 3 | **2** | **2** | reactor cooling water valve sticking |
| 15 ⚠ | 240 | 742 | 707 | 707 | **139** | 681 | never | never | condenser cooling water valve sticking (near |

⚠ marks the three faults the literature agrees are close to undetectable. They are reported rather than dropped: a comparison that quietly excludes them flatters every method, and *whether a detector claims to find what nobody finds* is exactly the question dropping them hides.


### The seven detectable faults — no resolution at all

| false-alarm budget | univariate | T² | SPE | T² or SPE | residual | DPCA T² | DPCA SPE | DPCA T²|SPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 per 1000 | **2** | 7 | 4 | 3 | 3 | 8 | 3 | **2** |
| 5 per 1000 | **2** | 3 | 3 | **2** | **2** | 5 | **2** | **2** |
| 20 per 1000 | **2** | 3 | **2** | **2** | **2** | 5 | **2** | **2** |
| 50 per 1000 | **2** | **2** | **2** | **2** | **2** | 3 | **2** | **2** |

Every detector sits at the m-of-n floor. With 3-sample persistence the fastest possible delay is 2, and at every budget the **univariate "wall of charts" is never worse than anything else**. On this fault set the multivariate machinery buys nothing: TE's detectable faults are steps and drifts that push individual measurements clean outside their normal range, which is precisely the case a per-tag limit was already good at. The multivariate argument is about faults that break *correlations* without moving any single tag much, and these are not those.


### The three hard faults — where the methods differ, and the ranking will not sit still

| false-alarm budget | univariate | T² | SPE | T² or SPE | residual | DPCA T² | DPCA SPE | DPCA T²|SPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 per 1000 | **232** | 374 (2/3) | 539 | 539 | 255 | 708 (2/3) | 570 (1/3) | 570 (1/3) |
| 5 per 1000 | 231 | **22** | 99 | 54 | 139 | 44 | 133 | 363 |
| 20 per 1000 | 42 | 22 | 4 | 3 | **0** | 23 | 53 | 131 |
| 50 per 1000 | 7 | 22 | 2 | 2 | **0** | 23 | 6 | 3 |

**The ranking flips three times across four budgets.** At 1 per 1000 the univariate detector is fastest; at 5 it is the slowest of the five and T² is fastest; at 20 and 50 the residual model is. Nothing about the data changed — only how much nuisance the operating point tolerates.

That is the finding, and it is a criticism of the synthetic study rather than a result from it: **the synthetic comparison reported a single operating point**, and on this evidence a single operating point cannot support a ranking. The residual model's win there is not refuted so much as shown to have been unfalsifiable as stated.

The zeros at the loose budgets should be read with suspicion rather than satisfaction. A delay of 0 on a fault the literature calls undetectable, bought at 20–50 nuisance alarms per 1000 samples, is a detector alarming most of the time and being right by coincidence.


### Dynamic PCA: the textbook fix, and it does not help

Static PCA assumes the rows are independent. TE's are not — median lag-1 autocorrelation **0.55**, **58%** of variables above 0.5 — so the effective sample size behind every limit is smaller than the row count implies. Lag embedding (Ku, Storch & Georgakis 1995) is the standard answer: stack *l* lagged copies so the dynamics move inside the model.

It costs columns, and the training set does not grow:

| lags | columns | samples per column | hard faults | detectable |
|---:|---:|---:|---:|---:|
| 0 ← | 52 | 9.6 | 3 | 2 |
| 1 | 104 | 4.8 | 22 | 2 |
| 2 | 156 | 3.2 | 131 | 2 |
| 3 | 208 | 2.4 | 8 | 2 |
| 4 | 260 | 1.9 | 26 | 2 |

**Static PCA wins.** At 9.6 samples per column it detects the hard faults at a median of 3; every lag count is worse, and the ratio falls to 1.9 by four lags. The likely reason is that lag embedding spends the fix on dimensions the training set cannot afford — a covariance estimate in 156 dimensions from 500 rows is not the same object as one in 52.

**That explanation is offered, not demonstrated.** The sweep is not monotonic — three lags beats two — and with only three hard faults each median is over three numbers, so the ordering between the non-zero lag counts is noise. What the table supports is the negative result: on this data, at this training-set size, the textbook correction for autocorrelation does not improve detection. It would need a longer normal run to separate *the method does not help here* from *the method cannot be fitted here*.


## SKAB — a real rig, and a null result

12 runs, 8 tags (Accelerometer1RMS, Accelerometer2RMS, Current, Pressure…), fitted on the first half of 9405 anomaly-free samples and calibrated on the second.

| detector | detected | median delay |
|---|---:|---:|
| univariate 3-sigma (the wall of charts) | 12/12 | 0 |
| PCA T2 only | 12/12 | 0 |
| PCA SPE only | 12/12 | 0 |
| T2 or SPE | 12/12 | 0 |
| residual (model-based) | 12/12 | 0 |
| DPCA T2 | 12/12 | 0 |
| DPCA SPE | 12/12 | 0 |
| DPCA T2 or SPE | 12/12 | 0 |

**Every detector fires on the first labelled sample, so SKAB separates nothing.** That is a property of the dataset as used here, not a compliment to the detectors: the anomalies are physical interventions on a small test loop — a valve closed by hand, a rotor unbalanced — and they are large, abrupt, and already under way at the first sample the label marks. A benchmark on which everything scores identically is reported as such rather than as five methods agreeing.


## What this does and does not settle

- **It settles the circularity.** The detectors were written against a generator I built; they now have a score on a process I did not, and the score does not support what the synthetic study concluded.
- **It does not make TE real.** It is a simulation with a thirty-year literature, which is a different and better thing than a simulation with a README.
- **The synthetic study is not re-run or retracted here.** Its numbers stand as measurements of its own generator; what changes is the claim built on top of them, and `docs/RESULTS.md` now points here.
- **SKAB was a null.** It is kept because a null that is reported is worth more than a null that is dropped, and dropping it would leave the impression that both datasets agreed with TE.

