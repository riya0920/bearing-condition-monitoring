"""Per-asset baselines, the health index, and the alarm state machine.

Why absolute thresholds do not work, demonstrated by having tried them: the
envelope-spectrum energy ratio of a HEALTHY bearing in this simulation is already
about 12x the broadband floor, because the peak of a noisy spectrum over a search
window is several times its median by construction. A threshold of "4x the floor"
therefore fires on everything. The first version of `diagnose()` had exactly that
bug and classified healthy bearings as inner-race faults with total confidence.

The fix is the thing real condition-monitoring systems do: every asset gets a
CALIBRATION PERIOD, its own healthy distribution is measured, and every feature is
reported as exceedance over that asset's own baseline. Mounting, load, foundation
and sensor gain all differ per asset; a fleet-wide absolute threshold is a promise
that they do not.

This also answers the cold-start question directly. A new asset with 3 days of
history has no baseline, so it gets the FLEET PRIOR and an explicit
`LOW_CONFIDENCE` state that the UI is required to show. It does not get silence,
and it does not get a number pretending to be calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

# Features the health index is built from, and the direction of "worse".
INDEX_FEATURES = ("env_BPFO_ratio", "env_BPFI_ratio", "env_BSF_ratio", "env_kurtosis")

MIN_BASELINE_CYCLES = 30  # below this the baseline is not trusted


class State(IntEnum):
    LOW_CONFIDENCE = -1
    NORMAL = 0
    WATCH = 1
    ALERT = 2
    CRITICAL = 3


@dataclass
class Baseline:
    """Healthy-period statistics for ONE asset."""
    median: dict[str, float] = field(default_factory=dict)
    p95: dict[str, float] = field(default_factory=dict)
    iqr: dict[str, float] = field(default_factory=dict)
    n_cycles: int = 0
    trusted: bool = False

    @classmethod
    def fit(cls, feats: list[dict], keys=INDEX_FEATURES) -> "Baseline":
        b = cls(n_cycles=len(feats), trusted=len(feats) >= MIN_BASELINE_CYCLES)
        for k in keys:
            v = np.array([f[k] for f in feats], dtype=float)
            b.median[k] = float(np.median(v))
            b.p95[k] = float(np.percentile(v, 95))
            q1, q3 = np.percentile(v, [25, 75])
            # Robust scale. A plain SD over the calibration window is contaminated
            # by any incipient fault already present at install, which is not rare.
            b.iqr[k] = float(max(q3 - q1, 1e-9))
        return b

    def exceedance(self, feats: dict, key: str) -> float:
        """How many robust scale units above this asset's healthy median."""
        return (feats[key] - self.median[key]) / self.iqr[key]


def health_index(feats: dict, baseline: Baseline, keys=INDEX_FEATURES,
                 saturate: float = 12.0) -> float:
    """0-100, where 100 is as-new. Built from exceedance, not from raw amplitude.

    The score is driven by the WORST feature rather than the average, because a
    bearing with one screaming fault frequency and three quiet ones is a bearing
    with a fault, and averaging is how that gets diluted into normality.
    """
    ex = max(baseline.exceedance(feats, k) for k in keys)
    ex = float(np.clip(ex, 0.0, saturate))
    return float(100.0 * (1.0 - ex / saturate))


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Trailing median smoothing. MEDIAN, not mean: a single bad acquisition (a
    hammer strike nearby, a loose sensor for one reading) should not move the
    score, and a mean lets it."""
    v = np.asarray(values, dtype=float)
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - window + 1) : i + 1])
    return out


@dataclass
class AlarmPolicy:
    """m-of-n persistence with hysteresis.

    Two separate thresholds per transition -- one to enter a state, a LOWER one to
    leave it. Without the gap, a score sitting on the threshold toggles every
    acquisition, which is the flapping that makes operators mute a system in week
    three. The gap is not a tuning nicety; it is the difference between a system
    that is used and one that is disabled.
    """
    watch_enter: float = 85.0
    watch_exit: float = 90.0
    alert_enter: float = 70.0
    alert_exit: float = 78.0
    critical_enter: float = 45.0
    critical_exit: float = 55.0
    m: int = 3   # of
    n: int = 5   # consecutive acquisitions

    def _m_of_n(self, flags: np.ndarray, i: int) -> bool:
        lo = max(0, i - self.n + 1)
        return bool(flags[lo : i + 1].sum() >= self.m)

    def run(self, score: np.ndarray, trusted: bool = True) -> np.ndarray:
        """Return the state at every cycle."""
        score = np.asarray(score, dtype=float)
        n = len(score)
        states = np.zeros(n, dtype=int)
        cur = State.NORMAL if trusted else State.LOW_CONFIDENCE
        below_w = score < self.watch_enter
        below_a = score < self.alert_enter
        below_c = score < self.critical_enter
        for i in range(n):
            if not trusted:
                states[i] = State.LOW_CONFIDENCE
                continue
            # Escalate on m-of-n below the ENTER threshold.
            if self._m_of_n(below_c, i):
                cur = State.CRITICAL
            elif self._m_of_n(below_a, i) and cur < State.CRITICAL:
                cur = max(cur, State.ALERT) if cur != State.CRITICAL else cur
            elif self._m_of_n(below_w, i) and cur < State.ALERT:
                cur = max(cur, State.WATCH)
            # De-escalate only above the higher EXIT threshold, one step at a time.
            if cur == State.CRITICAL and score[i] > self.critical_exit:
                cur = State.ALERT
            elif cur == State.ALERT and score[i] > self.alert_exit:
                cur = State.WATCH
            elif cur == State.WATCH and score[i] > self.watch_exit:
                cur = State.NORMAL
            states[i] = int(cur)
        return states


def count_flaps(states: np.ndarray, at_or_above: int = int(State.ALERT)) -> int:
    """Number of times the asset crosses INTO the alarm band and back out again.

    This is the alarm-fatigue counter. It is deliberately not "number of alarms":
    an alarm that stays on is one event a planner deals with, while an alarm that
    comes and goes six times is six interruptions and one lost user.
    """
    s = np.asarray(states) >= at_or_above
    if len(s) == 0:
        return 0
    transitions = np.diff(s.astype(int))
    entries = int((transitions == 1).sum()) + int(s[0])
    exits = int((transitions == -1).sum())
    return max(0, min(entries, exits))


def first_sustained(states: np.ndarray, at_or_above: int = int(State.ALERT)) -> int | None:
    """Index of the first alarm that is never de-asserted for the rest of the run."""
    s = np.asarray(states) >= at_or_above
    n = len(s)
    for i in range(n):
        if s[i] and s[i:].all():
            return i
    return None


def diagnose_with_baseline(feats: dict, baseline: Baseline, min_exceed: float = 3.0,
                           margin: float = 1.5,
                           sideband_min: float = 0.30) -> tuple[str, float]:
    """Name the fault in units of this asset's own healthy variation.

    1. Is anything there? The strongest candidate must exceed the asset's healthy
       median by `min_exceed` robust scale units. Otherwise: healthy.
    2. Sidebands BREAK TIES. If BPFI is contending -- within `margin` of the leader
       -- and carries shaft-rate sidebands, call inner race. Shaft-rate sidebands
       mean the defect is rotating, which means the rotating race.
    3. Otherwise the leader must beat the runner-up by `margin` exceedance units,
       or the answer is "indeterminate" -- a real answer, not a failure.

    Step 2 is deliberately a tie-break and not an override. The earlier version let
    any sideband energy above threshold win outright, and on a severe outer-race
    fault it flipped the diagnosis to BPFI at the very end of life: a strong fault
    lifts the whole envelope floor, BPFI's exceedance crosses `min_exceed` on
    leakage alone, and its sideband ratio is then computed on noise. Sidebands are
    evidence about WHICH of two contenders it is, not evidence that there are two.
    """
    keys = {"BPFO": "env_BPFO_ratio", "BPFI": "env_BPFI_ratio", "BSF": "env_BSF_ratio"}
    ex = {name: baseline.exceedance(feats, k) for name, k in keys.items()}
    order = sorted(ex.items(), key=lambda kv: -kv[1])
    top, second = order[0], order[1]
    if top[1] < min_exceed:
        return "healthy", float(top[1])
    bpfi_contending = (top[0] == "BPFI") or (top[1] - ex["BPFI"] < margin)
    if bpfi_contending and ex["BPFI"] >= min_exceed and feats.get("sb_BPFI", 0.0) > sideband_min:
        return "BPFI", float(feats["sb_BPFI"])
    if top[1] - second[1] < margin:
        return "indeterminate", float(top[1] - second[1])
    return top[0], float(top[1] - second[1])
