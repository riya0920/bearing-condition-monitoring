"""Per-alarm explanation, cold-start policy, and the P-F interval.

Three gaps the first build named, and the first is the one that decides whether
the system survives contact with an operator.

WHY AN ALARM NEEDS A REASON
---------------------------
The first build recommended shipping Hotelling T-squared partly BECAUSE its score
decomposes into per-feature contributions -- and then did not decompose it. That
is the gap this closes, and it matters more than the detector choice: an alarm
that says "health 62, investigate" gets acknowledged and ignored, because the
person receiving it cannot act on a scalar. An alarm that says "health 62, driven
by envelope energy at BPFO, which is 6.2 sigma above this asset's own baseline"
sends somebody to look at a specific bearing.

The decomposition is exact, not an attribution heuristic:

    T^2 = (x - mu)' * S^-1 * (x - mu)

Writing d = x - mu and letting the quadratic form be a sum over feature pairs,
each feature j contributes

    c_j = d_j * (S^-1 d)_j

and sum_j c_j = T^2 exactly. Unlike SHAP or occlusion this is an identity rather
than an approximation, which is worth saying out loud in a domain where "the model
said so" is not an acceptable answer to a maintenance planner.

The contributions can be NEGATIVE when features are correlated -- a feature moving
*with* its correlated partners can reduce the distance. That is real information
(it says the deviation is in the expected direction) and it is reported rather
than clipped away.
"""
from __future__ import annotations

import numpy as np


def t2_contributions(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray,
                     names: list[str]) -> list[dict]:
    """Exact per-feature decomposition of a Hotelling T-squared score."""
    d = np.asarray(x, dtype=float) - mu
    sd = inv_cov @ d
    contrib = d * sd
    total = float(contrib.sum())
    rows = [{
        "feature": names[j],
        "contribution": float(contrib[j]),
        "pct_of_total": float(100.0 * contrib[j] / total) if total else float("nan"),
        "deviation_sigma": float(d[j] / np.sqrt(1.0 / max(inv_cov[j, j], 1e-12)))
        if inv_cov[j, j] > 0 else float("nan"),
    } for j in range(len(names))]
    return sorted(rows, key=lambda r: -r["contribution"])


def explain_alarm(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray,
                  names: list[str], top_n: int = 4) -> dict:
    rows = t2_contributions(x, mu, inv_cov, names)
    total = sum(r["contribution"] for r in rows)
    top = rows[:top_n]
    return {
        "t2": total,
        "top_contributors": top,
        "top_n_pct_of_score": float(sum(r["pct_of_total"] for r in top)),
        "sentence": _sentence(top, total),
        "all_contributions": rows,
    }


def _sentence(top: list[dict], total: float) -> str:
    """The line that goes on the alarm card. One sentence, no jargon-free-ness
    for its own sake -- an operator on a rotating-equipment route knows what BPFO
    is, and dumbing it down loses the actionable part."""
    if not top:
        return "no contributors"
    lead = top[0]
    rest = ", ".join(t["feature"] for t in top[1:3])
    s = (f"T² = {total:.1f}, driven by {lead['feature']} "
         f"({lead['pct_of_total']:.0f}% of the score, "
         f"{lead['deviation_sigma']:+.1f}σ from this asset's baseline)")
    return s + (f"; then {rest}." if rest else ".")


# --------------------------------------------------------------------------
# cold start
# --------------------------------------------------------------------------

def cold_start_policy(n_cycles: int, min_cycles: int = 30,
                      fleet_prior: dict | None = None) -> dict:
    """What the system says about an asset with almost no history.

    Three states, and the middle one is the one most systems omit:

      NO_BASELINE     below `min_cycles`: no per-asset statistics exist. Use the
                      FLEET PRIOR and label the output LOW_CONFIDENCE. Not silence
                      -- silence is indistinguishable from healthy, and a new
                      machine is exactly when infant-mortality failures happen.
      PROVISIONAL     baseline exists but is short. Widen the thresholds in
                      proportion to the uncertainty in the estimated scale, so a
                      noisy baseline produces a conservative detector rather than
                      a jumpy one.
      ESTABLISHED     enough history to trust the asset's own distribution.

    The widening factor is 1 + 1/sqrt(2(n-1)), the approximate relative standard
    error of a standard-deviation estimate from n samples. It is not a tuning knob
    -- it is the sampling error of the thing being estimated, which means the
    detector automatically becomes less trigger-happy exactly when its baseline is
    least trustworthy.
    """
    if n_cycles < min_cycles:
        return {"state": "NO_BASELINE", "confidence": "LOW",
                "uses_fleet_prior": True, "threshold_widening": 1.5,
                "n_cycles": n_cycles,
                "note": ("fleet prior in use; per-asset baseline needs "
                         f"{min_cycles - n_cycles} more acquisitions")}
    if n_cycles < 3 * min_cycles:
        w = 1.0 + 1.0 / np.sqrt(2 * (n_cycles - 1))
        return {"state": "PROVISIONAL", "confidence": "MEDIUM",
                "uses_fleet_prior": False, "threshold_widening": float(w),
                "n_cycles": n_cycles,
                "note": f"thresholds widened x{w:.3f} for baseline sampling error"}
    return {"state": "ESTABLISHED", "confidence": "HIGH", "uses_fleet_prior": False,
            "threshold_widening": 1.0, "n_cycles": n_cycles,
            "note": "per-asset baseline trusted"}


def fleet_prior(baselines: list) -> dict:
    """Pooled statistics across assets, for an asset that has none of its own."""
    keys = baselines[0].median.keys()
    return {
        "median": {k: float(np.median([b.median[k] for b in baselines])) for k in keys},
        "iqr": {k: float(np.median([b.iqr[k] for b in baselines])) for k in keys},
        "n_assets": len(baselines),
    }


# --------------------------------------------------------------------------
# the P-F interval
# --------------------------------------------------------------------------

def pf_interval(health: np.ndarray, failure_index: int,
                potential_failure_threshold: float = 90.0,
                functional_failure_threshold: float = 40.0) -> dict:
    """The maintenance-planning interval this whole field is organised around.

    P = POTENTIAL failure: the first point at which the condition is detectable at
        all. Here, the first sustained crossing below the P threshold.
    F = FUNCTIONAL failure: the point at which the asset can no longer do its job.

    The P-F interval is F - P, and it is the quantity that decides the INSPECTION
    INTERVAL: to catch a fault you must inspect at least twice within P-F, so the
    interval must be at most half of it. A monitoring system that cannot state its
    P-F distribution cannot tell a planner how often to look, which is the first
    question a planner asks.

    Reporting the DISTRIBUTION and not the mean matters here more than usual: the
    inspection interval has to be set from a low percentile. Half the *mean* P-F
    interval misses the fast half of the failures by construction.
    """
    h = np.asarray(health, dtype=float)
    below_p = h < potential_failure_threshold
    below_f = h < functional_failure_threshold

    p_idx = None
    for i in range(len(h)):
        if below_p[i] and below_p[i:].mean() > 0.9:
            p_idx = i
            break
    f_idx = None
    for i in range(len(h)):
        if below_f[i] and below_f[i:].mean() > 0.9:
            f_idx = i
            break
    if f_idx is None:
        f_idx = failure_index

    return {
        "p_index": p_idx, "f_index": f_idx,
        "pf_interval_cycles": (f_idx - p_idx) if p_idx is not None else None,
        "p_before_failure": (failure_index - p_idx) if p_idx is not None else None,
        "detected": p_idx is not None,
    }


def inspection_interval(pf_intervals: list[int], percentile: float = 10.0,
                        inspections_per_pf: float = 2.0) -> dict:
    """Turn a P-F distribution into the number a planner actually needs."""
    v = np.array([x for x in pf_intervals if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    p_low = float(np.percentile(v, percentile))
    return {
        "n": len(v),
        "pf_mean": float(v.mean()), "pf_median": float(np.median(v)),
        "pf_p10": p_low, "pf_min": float(v.min()),
        "recommended_interval_cycles": p_low / inspections_per_pf,
        "interval_if_mean_used": float(v.mean()) / inspections_per_pf,
        "percentile_used": percentile,
    }
