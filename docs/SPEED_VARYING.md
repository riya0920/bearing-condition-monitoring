# ML-3 pass 5 — the speed-varying case

Item 9 of the not-built list said no run-up, no coast-down and no order tracking, *"which is where fixed-frequency band energy stops working entirely"*. That was an assertion. Here it is as a measurement, the fix, and — first — the correction, because **the assertion was too strong.**

## The claim was overstated

Fixed-frequency detection does not stop working at the first sign of speed variation. It keeps naming the right fault well past the point where its evidence has mostly evaporated, because every competing candidate smears too and the winner only has to win. What collapses is the MARGIN, and the margin is what an early fault has to spend. Both tables below are needed: the call rate says when the method fails outright, the ratio says when it stopped being worth trusting.

## Why it degrades, before the numbers

Every detector in this project locates energy at a FREQUENCY. BPFO is 3.585× shaft, so at 29.95 Hz shaft it is 107.4 Hz and the search window goes there with a 2% tolerance. During a run-up the line sweeps clean across that window and out the other side, so the energy is spread over a band far wider than the tolerance and **the peak the detector is looking for does not exist at any single frequency.**

Order tracking resamples onto uniform shaft ANGLE. A defect strikes once every fixed number of revolutions, not once every fixed number of seconds, so in the angle domain the line is stationary again — at an *order* rather than a frequency. `BearingGeometry.orders()` has been in this codebase since the first pass and nothing had ever used it.

## 1. The sweep sweep

12 trials per fault per width, two faults, 2-second records, at full fault severity. Correct-call rate:

| speed variation | fixed frequency | order, tacho phase | order, phase estimated from the signal |
|---|---|---|---|
| ±0% | 100% | 100% | 100% |
| ±5% | 100% | 100% | 100% |
| ±10% | 100% | 100% | 100% |
| ±25% | 100% | 100% | 100% |
| ±50% | 62% | 100% | 100% |

**The first row is the control.** At constant speed the two domains are analysing the same thing and score 100% against 100%. A table whose first row disagreed would be measuring an implementation bug and nothing else.

Now the ratio at the true fault order, which is where the mechanism shows:

| speed variation | fixed frequency | order, tacho | order, estimated |
|---|---|---|---|
| ±0% | 92.4 | 105.1 | 49.4 |
| ±5% | 38.3 | 106.0 | 38.9 |
| ±10% | 25.6 | 107.2 | 27.7 |
| ±25% | 13.5 | 107.8 | 25.1 |
| ±50% | 9.2 | 112.6 | 30.8 |

The fixed-frequency ratio falls **92 → 9**, a factor of 10, while the order-tracked ratio goes 105 → 113 — flat, or slightly up. **The energy did not go anywhere. It is exactly where it always was, in angle.** The call survives to ±50% only because a strong fault can afford to lose 10× of its evidence and still outrank the alternatives.

## 2. Which is why the next table matters more

The same comparison at severities where the fixed method starts near the healthy gate of 6 rather than a hundred times above it. That is what an EARLY fault looks like, and early faults are what the whole project exists to catch.

| severity | speed variation | fixed: correct | fixed: median ratio | order: correct | order: median ratio |
|---|---|---|---|---|---|
| 0.05 | ±0% | 0% | 4.7 | 4% | 4.9 |
| 0.05 | ±25% | 4% | 4.3 | 0% | 4.6 |
| 0.05 | ±50% | 0% | 4.4 | 4% | 5.1 |
| 0.10 | ±0% | 42% | 6.4 | 50% | 6.8 |
| 0.10 | ±25% | 4% | 4.4 | 50% | 6.5 |
| 0.10 | ±50% | 0% | 4.7 | 67% | 7.1 |
| 0.20 | ±0% | 100% | 18.2 | 100% | 19.7 |
| 0.20 | ±25% | 12% | 5.6 | 100% | 18.2 |
| 0.20 | ±50% | 4% | 5.2 | 100% | 19.8 |

Counting only the weak faults this detector **does** find at constant speed (74 paired trials - same fault, same seed, speed varied): fixed-frequency analysis loses **66 of 74** below the healthy gate, and order tracking loses **4 of 74**. A fault that falls under the gate is not misnamed. It is called healthy, and nobody looks at it again.

Read the severity-0.20 rows on their own. At constant speed the fixed method is right every time with a ratio near 18; at ±25% the same faults sit just above the gate and it names none of them, while order tracking has not moved. **That is the row the strong-fault table in section 1 was hiding.**

## 3. The cost of having no tacho

Speed tracked off the vibration signal instead of a keyphasor. Median absolute speed error across every trial: **3.83%**.

| speed variation | correct-call cost of estimating | ratio, tacho | ratio, estimated |
|---|---|---|---|
| ±0% | +0% | 105.1 | 49.4 |
| ±5% | +0% | 106.0 | 38.9 |
| ±10% | +0% | 107.2 | 27.7 |
| ±25% | +0% | 107.8 | 25.1 |
| ±50% | +0% | 112.6 | 30.8 |

The call rate barely moves and **the ratio loses about half its margin at every width, including at constant speed**. A constant speed estimated one bin off is a constant speed ERROR, and a constant speed error integrates into a phase that drifts linearly across the record — so the impulses at the end land at a different angle from the ones at the start and the line smears. Estimating the phase does not cost accuracy here; it costs the margin that keeps a weak fault above the gate, which is the same currency the previous section was spending.

## 4. Coast-down

A coast-down is not a reversed run-up — a machine losing energy to friction decays roughly exponentially, so the fast part is at the beginning and most of the record sits near the final speed. Over the same ±50% range: fixed 83%, order/tacho 100%, order/estimated 100%.

## 5. The check that decides whether any of this is real

36 snapshots from 6 real CWRU records, all at constant speed. The two methods agree on the call **97%** of the time, and their ratio vectors correlate at **r = 1.000** in log space.

Angular resampling has several ways to be silently wrong — an off-by-one in the phase integration, an order axis scaled by the wrong factor, a samples-per-rev low enough to alias one fault order onto another — and every one of them still produces a plausible-looking spectrum with a peak in it. On constant speed the angle axis is a linear function of the time axis, so the two spectra are the same measurement in different units. This is the only available check that can tell a better method from a broken one, and it is the reason the simulated tables above are worth reading at all.

## Honest limits

- The tacho phase is EXACT here because the generator knows the speed profile. A real keyphasor gives one pulse per revolution and the phase between pulses is interpolated, which adds error this study does not have.
- track_speed follows the strongest line in a speed window. On a machine whose 1x imbalance is not the dominant low-frequency content -- a gearbox with a strong mesh order, a fan with heavy blade-pass -- it will lock onto the wrong line and every angle after that is wrong.
- No real speed-varying data. CWRU runs at constant speed, so the run-up is simulated and only the CONSTANT-speed agreement check uses real measurements. A run-up rig record would test both halves at once.
- Angle is assumed monotone: no reversing drives.
- The comparison gives the fixed-frequency method the MEAN speed, which is the most favourable single number available to it. A real system would use nameplate speed or a stale tacho reading and do worse, so every gap above is a lower bound on the gap in a plant.
- Two fault types. Ball faults are excluded because this project already knows its line-energy detector is close to the wrong instrument for them (19% on real data), and adding a case that fails for an unrelated reason would muddy what these tables measure.
- The severity table's 'missed fault' count is against this project's own gate of 6.0, which was calibrated on constant-speed data. A plant running speed-varying machines would calibrate its own, and the right gate for order-tracked ratios is not the right gate for fixed-frequency ones.