"""Three detector families on the same features, at the same false-alarm budget.

The comparison only means something if the budget is matched. Any detector wins a
bake-off if you let it alarm more, so every method here has its threshold set to
hit the SAME false-alarm rate on healthy assets, and only then is lead time read
off. That is the same discipline as matched ARL0 in the SPC project, and it is the
single most common flaw in vendor comparisons.

  statistical  -- Hotelling T^2 / Mahalanobis distance on the baseline covariance,
                  the classic multivariate process-monitoring detector. Named as
                  such because it is not a neural network and it is not new: it is
                  what the process-monitoring literature has used since the 1940s.
  ml           -- IsolationForest on the same features.
  deep         -- a small autoencoder over a window of features, scored by
                  reconstruction error.

If the statistical baseline wins, the honest thing is to say so and ship it.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

import health

SEED = 20260818
FEATURE_KEYS = ("env_BPFO_ratio", "env_BPFI_ratio", "env_BSF_ratio",
                "env_kurtosis", "kurtosis", "crest_factor", "rms", "env_rms")


def _matrix(feats: list[dict]) -> np.ndarray:
    return np.array([[f[k] for k in FEATURE_KEYS] for f in feats], dtype=float)


# --------------------------------------------------------------------------
# scorers: each returns an ANOMALY SCORE per cycle (higher = worse)
# --------------------------------------------------------------------------

def score_t2(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Hotelling T^2 with a shrunk covariance.

    Shrinkage because the baseline window is short relative to the feature count
    and the raw covariance is near-singular; inverting it produces enormous
    distances driven by the least-observed direction. This is the standard failure
    of textbook Mahalanobis on short baselines.
    """
    mu = train.mean(axis=0)
    cov = np.cov(train, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0]) * 0.1
    inv = np.linalg.pinv(cov)
    d = x - mu
    return np.einsum("ij,jk,ik->i", d, inv, d)


def score_iforest(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    m = IsolationForest(n_estimators=300, random_state=SEED, contamination="auto")
    m.fit(train)
    return -m.score_samples(x)


class _AE(nn.Module):
    def __init__(self, n_in: int, hidden: int = 6):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, hidden))
        self.dec = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def score_autoencoder(train: np.ndarray, x: np.ndarray, epochs: int = 400) -> np.ndarray:
    """Autoencoder reconstruction error, trained on healthy data only.

    Standardisation uses the TRAIN statistics only; using the whole series would
    leak the fault into the scaler, which is the most common quiet mistake in
    unsupervised anomaly detection and always makes the method look better.
    """
    torch.manual_seed(SEED)
    mu, sd = train.mean(0), train.std(0) + 1e-9
    tr = torch.tensor((train - mu) / sd, dtype=torch.float32)
    te = torch.tensor((x - mu) / sd, dtype=torch.float32)
    model = _AE(tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(tr), tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        err = ((model(te) - te) ** 2).mean(dim=1).numpy()
    return err


SCORERS = {
    "Hotelling T² (statistical)": score_t2,
    "IsolationForest (ML)": score_iforest,
    "Autoencoder (deep)": score_autoencoder,
}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _first_sustained_above(score: np.ndarray, thr: float, m: int = 3, n: int = 5,
                           start: int = 0) -> int | None:
    """First index >= start where m-of-n scores exceed thr and it never recovers."""
    flag = score > thr
    for i in range(start, len(score)):
        lo = max(0, i - n + 1)
        if flag[lo : i + 1].sum() >= m and flag[i:].mean() > 0.85:
            return i
    return None


def _count_episodes(score: np.ndarray, thr: float, m: int = 3, n: int = 5) -> int:
    """Alarm episodes that assert and then clear -- the nuisance count."""
    flag = score > thr
    asserted = np.zeros(len(score), dtype=bool)
    for i in range(len(score)):
        lo = max(0, i - n + 1)
        asserted[i] = flag[lo : i + 1].sum() >= m
    episodes = 0
    prev = False
    for a in asserted:
        if a and not prev:
            episodes += 1
        prev = a
    # An episode that is still asserted at the end has not cleared.
    if asserted[-1] and episodes > 0:
        episodes -= 1
    return episodes


def bakeoff(fleet: list[dict], baseline_cycles: int,
            fa_budget_per_asset: float = 0.5) -> list[dict]:
    """Run all three detectors at a matched false-alarm budget on healthy assets."""
    healthy = [a for a in fleet if not a["failing"]]
    failing = [a for a in fleet if a["failing"]]
    rows = []
    for name, fn in SCORERS.items():
        # Score everything first.
        scores = {}
        for a in fleet:
            x = _matrix(a["feats"])
            scores[a["asset"]] = fn(x[:baseline_cycles], x)

        # Calibrate the threshold on HEALTHY assets to hit the budget. Sweeping
        # percentiles of the pooled healthy score, take the lowest threshold whose
        # measured nuisance rate is within budget -- lowest, because a lower
        # threshold means earlier detection, and we want the most sensitive
        # detector the budget will pay for.
        pooled = np.concatenate([scores[a["asset"]][baseline_cycles:] for a in healthy])
        chosen, chosen_fa = None, None
        for q in np.linspace(80, 99.99, 60):
            thr = float(np.percentile(pooled, q))
            fa = np.mean([_count_episodes(scores[a["asset"]], thr) for a in healthy])
            if fa <= fa_budget_per_asset:
                chosen, chosen_fa = thr, fa
                break
        if chosen is None:
            chosen = float(np.percentile(pooled, 99.99))
            chosen_fa = np.mean([_count_episodes(scores[a["asset"]], chosen) for a in healthy])

        leads, n_det = [], 0
        for a in failing:
            i = _first_sustained_above(scores[a["asset"]], chosen, start=baseline_cycles)
            if i is not None:
                leads.append(len(a["snaps"]) - 1 - i)
                n_det += 1
        rows.append({
            "name": name,
            "threshold_percentile_of_healthy": float(q),
            "median_lead": float(np.median(leads)) if leads else 0.0,
            "min_lead": float(np.min(leads)) if leads else 0.0,
            "max_lead": float(np.max(leads)) if leads else 0.0,
            "false_alarms_per_asset": float(chosen_fa),
            "n_detected": n_det, "n_failing": len(failing),
            "fa_budget": fa_budget_per_asset,
        })
    return rows


def lead_time_vs_false_alarms(fleet: list[dict], baseline_cycles: int) -> list[dict]:
    """The operating curve, on the physics health index rather than a raw score."""
    healthy = [a for a in fleet if not a["failing"]]
    failing = [a for a in fleet if a["failing"]]
    for a in fleet:
        hi = np.array([health.health_index(f, a["baseline"]) for f in a["feats"]])
        a["_hi"] = health.smooth(hi, 5)

    rows = []
    for thr in (95, 90, 85, 80, 70, 60, 50, 40):
        leads, missed = [], 0
        for a in failing:
            i = _first_sustained_above(-a["_hi"], -thr, start=baseline_cycles)
            if i is None:
                missed += 1
            else:
                leads.append(len(a["snaps"]) - 1 - i)
        fa = float(np.mean([_count_episodes(-a["_hi"], -thr) for a in healthy]))
        rows.append({
            "threshold": float(thr),
            "median_lead": float(np.median(leads)) if leads else 0.0,
            "p05_lead": float(np.percentile(leads, 5)) if leads else 0.0,
            "false_alarms_per_asset": fa,
            "n_missed": missed,
        })
    return rows


def state_machine_report(fleet: list[dict], baseline_cycles: int) -> list[dict]:
    policy = health.AlarmPolicy()
    rows = []
    for a in fleet:
        hi = np.array([health.health_index(f, a["baseline"]) for f in a["feats"]])
        his = health.smooth(hi, 5)
        states = policy.run(his, trusted=a["baseline"].trusted)
        first = health.first_sustained(states, int(health.State.ALERT))
        rows.append({
            "asset": a["asset"],
            "first_alert": first,
            "lead": (len(a["snaps"]) - 1 - first) if (first is not None and a["failing"]) else None,
            "flaps": health.count_flaps(states),
            "final_state": health.State(int(states[-1])).name,
        })
    return rows


def diagnosis_report(fleet: list[dict], baseline_cycles: int,
                     severity_threshold: float = 0.15) -> dict:
    labels = ("healthy", "BPFO", "BPFI", "BSF", "indeterminate")
    conf: dict[str, dict[str, int]] = {}
    developed_total = developed_right = developed_indet = 0
    for a in fleet:
        true = a["fault"] or "healthy"
        row = conf.setdefault(true, {k: 0 for k in labels})
        for i, f in enumerate(a["feats"]):
            if i < baseline_cycles:
                continue
            pred, _ = health.diagnose_with_baseline(f, a["baseline"])
            row[pred] = row.get(pred, 0) + 1
            if a["failing"] and a["severity"][i] > severity_threshold:
                developed_total += 1
                developed_right += int(pred == true)
                developed_indet += int(pred == "indeterminate")
    return {
        "confusion": [{"true": k, **v} for k, v in conf.items()],
        "severity_threshold": severity_threshold,
        "accuracy_when_developed": developed_right / max(1, developed_total),
        "indeterminate_when_developed": developed_indet / max(1, developed_total),
        "n_developed_cycles": developed_total,
    }
