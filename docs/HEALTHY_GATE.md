# The healthy gate, which had never been calibrated

The fleet dashboard produced the finding: **nothing on the screen is green**. Four of CWRU's forty files are healthy bearings and none of them is called healthy at asset level. `RESULTS.md` reports 21.9% of healthy *snapshots* called healthy, which reads as a middling number; aggregated to assets by majority vote it is zero.


## Why: an absolute constant inside a relative measure

`features.diagnose` opens with `if top < min_ratio: return healthy`, and `min_ratio` is **4.0**. The band ratio is already normalised — it is energy at a defect frequency against that same spectrum's noise floor — so the gate looks scale-free. It is not: the healthy distribution sits right on top of it.

| percentile | healthy | faulty |
|---|---:|---:|
| p5 | 3.66 | 9.20 |
| p25 | 4.07 | 13.33 |
| p50 | 4.51 | 35.37 |
| p75 | 5.19 | 78.45 |
| p95 | 5.91 | 265.56 |

**78% of healthy snapshots sit above the gate of 4.0**, which is most of the way to explaining the zero. Pass 3's recalibration tuned the *sideband* threshold and never touched this one — visible in its own output, where healthy accuracy is 0.4375 for every variant it tried.


## The trade, swept

Raising the gate always helps healthy accuracy and always costs fault detection. There is no setting that is simply better, which is why this is a curve rather than a fix:

| gate | healthy snapshots called healthy | faulty: correct race | faulty called healthy | healthy assets green | faulty assets missed |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 0% | 62% | 0% | 0/4 | 0/36 |
| 4.0 ← old | 22% | 62% | 0% | 0/4 | 0/36 |
| 5.0 | 67% | 62% | 0% | 4/4 | 0/36 |
| 6.0 ← new | 95% | 62% | 0% | 4/4 | 0/36 |
| 7.0 | 100% | 62% | 0% | 4/4 | 0/36 |
| 8.0 | 100% | 62% | 2% | 4/4 | 0/36 |
| 9.0 | 100% | 61% | 4% | 4/4 | 0/36 |
| 10.0 | 100% | 61% | 9% | 4/4 | 3/36 |
| 12.0 | 100% | 59% | 19% | 4/4 | 9/36 |
| 14.0 | 100% | 57% | 27% | 4/4 | 11/36 |

← the gate that shipped. It gets 0/4 healthy assets green and 62% of faulty snapshots' races right.


**There is a plateau, and it is wide.** Every gate from **4.75 to 8.25** — 15 settings, a band 3.50 wide — turns *all* healthy assets green, misses *no* faulty asset, and does *not* reduce the correct-race rate. The trade this section opened by assuming would exist does not exist in that range: the old gate was not a conservative choice on a curve, it was simply below the plateau.


## Choosing it honestly: leave one healthy file out

With 4 healthy files, a threshold chosen on all of them and scored on all of them is a threshold scored on its own training set. Leave-one-out, choosing the smallest gate that calls 80% of the calibration healthy snapshots healthy:

| held out | gate chosen | held-out healthy called healthy | faulty correct race |
|---|---:|---:|---:|
| 100 | 5.50 | 100% | 62% |
| 97 | 5.50 | 100% | 62% |
| 98 | 5.25 | 75% | 62% |
| 99 | 5.25 | 56% | 62% |

**Every fold picks 5.25–5.50** — a spread of 0.25 — and all four land inside the plateau. That combination is what makes this a calibration rather than an overfit: the estimator is stable across folds *and* the answer does not depend on getting it exactly right.


## Applied

`features.diagnose`'s `min_ratio` is changed from **4.0** to **6.0**. Not the leave-one-out median: 6.0 sits above every fold's choice, so it errs toward calling a marginal spectrum healthy, and 2.25 below the top of the plateau. A margin inside a measured band, rather than an optimum on four recordings.

| | old gate | new gate |
|---|---:|---:|
| healthy snapshots called healthy | 22% | **95%** |
| healthy assets green | 0/4 | **4/4** |
| faulty: correct race | 62% | 62% |
| faulty assets called healthy | 0/36 | 0/36 |

Fault detection is unchanged. The whole of the improvement is on the healthy side, which is what a gate below the healthy distribution predicts and is the reason this was worth measuring rather than guessing.


## What this settles, and what it does not

- **The cause was an absolute constant inside a relative measure.** The band ratio is normalised to its own spectrum's noise floor, so the gate looked scale-free and was not.
- **The draft of this document concluded the opposite.** It said the change should not be applied, on the grounds that four healthy files cannot pin a threshold. That is true and it is not the question — the plateau means the threshold does not need pinning, and the leave-one-out spread being small *inside* a wide flat region is the evidence that settles it. The wrong conclusion is recorded because the reasoning that produced it is the tempting one.
- **Four healthy recordings is still four.** The plateau is measured on them, so its width is itself an estimate from n = 4. What would improve it is more healthy files at more load levels, which CWRU has.
- **62% correct-race is unchanged and still the real weakness.** This fixes the healthy side and touches nothing about telling BPFO from BPFI, which is where the remaining error is.

