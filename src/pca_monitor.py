"""PCA-based process monitoring: T² and SPE, and why you need both.

THE DISTINCTION THAT IS THE WHOLE POINT, and the one most "anomaly detection on
sensor data" write-ups never make.

PCA splits the measurement space in two. The first `k` principal components span
the MODEL PLANE -- the subspace the process normally moves in, which encodes the
correlation structure. Everything orthogonal to it is the RESIDUAL SPACE, which
under normal operation contains only noise.

    T²   distance from the centre, measured INSIDE the model plane.
         "The process is in a normal STATE, but an extreme one."
         Feed rate high, temperature high, pressure high -- all consistent with
         each other, just further from the operating point than usual.

    SPE  distance FROM the model plane, also called Q.
         "The process is in a state that violates its own relationships."
         Temperature normal, cooling flow normal, and that temperature at that
         cooling flow has never happened before.

A fault that shifts the operating point along the normal correlation structure
raises T² and leaves SPE flat. A fault that BREAKS a relationship raises SPE and
can leave T² flat -- and it is invisible to every univariate chart, because no
single variable left its own limits.

**Monitoring only T² therefore misses exactly the class of fault that
multivariate monitoring exists for.** It is also the easy mistake, because T² is
the one with the familiar name.

CONTROL LIMITS. Both statistics get limits from the reference data's own
distribution rather than from a chi-squared or the Jackson-Mudholkar formula.
Those are exact under multivariate normality, and process data is not normal --
it is autocorrelated, mildly nonlinear, and mixed across operating points. A
percentile of the empirical reference distribution carries a false-alarm rate by
construction, which is the property that actually has to hold.

WHAT THIS DOES NOT HANDLE, named rather than left for the reader to discover:
process data is AUTOCORRELATED, so consecutive samples are not independent and
the effective sample size behind any limit is smaller than the row count
suggests. Dynamic PCA (lagged copies of each variable) is the standard answer and
is not implemented here.
"""
from __future__ import annotations

import numpy as np


class PCAMonitor:
    """Fit on normal operation; score T² and SPE separately, always both."""

    def __init__(self, n_components: int | None = None,
                 variance_target: float = 0.90) -> None:
        self.k = n_components
        self.variance_target = float(variance_target)
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.P: np.ndarray | None = None        # loadings, (n_tags, k)
        self.eig: np.ndarray | None = None
        self.t2_limit = self.spe_limit = float("nan")
        self.explained = float("nan")

    # -- fitting -----------------------------------------------------------
    def fit(self, x: np.ndarray, alpha: float = 0.99) -> "PCAMonitor":
        x = np.asarray(x, dtype=float)
        self.mu, self.sd = x.mean(0), x.std(0)
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        z = (x - self.mu) / self.sd

        # SVD on the standardised data. Standardising is not cosmetic: without
        # it a pressure in kPa dominates a composition in mole fraction purely
        # through its units, and the "principal" components describe the choice
        # of units rather than the process.
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        var = s ** 2 / max(len(z) - 1, 1)
        ratio = var / var.sum()
        if self.k is None:
            self.k = int(np.searchsorted(np.cumsum(ratio),
                                         self.variance_target) + 1)
            self.k = max(1, min(self.k, len(var) - 1))
        self.P = vt[: self.k].T
        self.eig = np.maximum(var[: self.k], 1e-12)
        self.explained = float(np.cumsum(ratio)[self.k - 1])

        t2, spe = self._stats(z)
        # Empirical limits, not chi-squared: process data is autocorrelated and
        # mildly nonlinear, so a distributional limit carries a false-alarm rate
        # nobody measured.
        self.t2_limit = float(np.quantile(t2, alpha))
        self.spe_limit = float(np.quantile(spe, alpha))
        self.alpha = alpha
        return self

    # -- scoring -----------------------------------------------------------
    def _stats(self, z: np.ndarray) -> tuple:
        t = z @ self.P                          # scores in the model plane
        t2 = np.einsum("ij,j,ij->i", t, 1.0 / self.eig, t)
        recon = t @ self.P.T
        e = z - recon                           # residual, orthogonal to the plane
        spe = np.einsum("ij,ij->i", e, e)
        return t2, spe

    def score(self, x: np.ndarray) -> dict:
        if self.P is None:
            raise RuntimeError("fit() first")
        z = (np.asarray(x, dtype=float) - self.mu) / self.sd
        t2, spe = self._stats(z)
        return {
            "t2": t2, "spe": spe,
            "t2_alarm": t2 > self.t2_limit,
            "spe_alarm": spe > self.spe_limit,
            "any_alarm": (t2 > self.t2_limit) | (spe > self.spe_limit),
            "t2_limit": self.t2_limit, "spe_limit": self.spe_limit,
        }

    # -- fault isolation ---------------------------------------------------
    def spe_contributions(self, x: np.ndarray) -> np.ndarray:
        """Per-variable contribution to SPE. Exact: they sum to SPE.

        For SPE the contribution is simply the squared residual of each
        variable, so the decomposition is not an approximation and needs no
        apology -- unlike SHAP or occlusion, which estimate.

        The caveat that matters operationally: contributions suffer from SMEARING.
        A fault in one variable propagates through the model into the residuals of
        the variables correlated with it, so the largest contributor is not always
        the faulty sensor. It narrows the search from eighteen tags to two or
        three, and an engineer still has to look.
        """
        z = (np.asarray(x, dtype=float) - self.mu) / self.sd
        e = z - (z @ self.P) @ self.P.T
        return e ** 2

    def t2_contributions(self, x: np.ndarray) -> np.ndarray:
        """Per-variable contribution to T², via the scores."""
        z = (np.asarray(x, dtype=float) - self.mu) / self.sd
        t = z @ self.P
        w = (t / self.eig) @ self.P.T
        return w * z

    def diagnose(self, x: np.ndarray, tags: list, top: int = 3) -> list[dict]:
        """Which statistic fired, and which tags drove it."""
        sc = self.score(x)
        spe_c = self.spe_contributions(x)
        t2_c = self.t2_contributions(x)
        out = []
        for i in range(len(x)):
            if not sc["any_alarm"][i]:
                continue
            which = ("SPE" if sc["spe_alarm"][i] and not sc["t2_alarm"][i]
                     else "T2" if sc["t2_alarm"][i] and not sc["spe_alarm"][i]
                     else "both")
            c = spe_c[i] if which in ("SPE", "both") else t2_c[i]
            idx = np.argsort(-c)[:top]
            out.append({
                "sample": i, "statistic": which,
                "t2": float(sc["t2"][i]), "spe": float(sc["spe"][i]),
                "top_tags": [{"tag": tags[j], "contribution": float(c[j])}
                             for j in idx],
                "reading": ("relationships broken -- a state that violates the "
                            "process's own correlations"
                            if which == "SPE" else
                            "normal state, extreme magnitude" if which == "T2"
                            else "both magnitude and relationships have moved"),
            })
        return out


# ---------------------------------------------------------------------------
# scoring a run
# ---------------------------------------------------------------------------

def detection_delay(alarm: np.ndarray, fault_start: int, m: int = 3,
                    n: int = 5) -> int | None:
    """Samples from fault onset to a SUSTAINED alarm (m of the last n).

    Persistence, not a first crossing. A single sample over the limit is how a
    monitoring system earns the reputation that gets it muted, and the m-of-n
    rule is the same one the bearing side uses -- deliberately, so the two halves
    of this project are scored on the same policy.
    """
    a = np.asarray(alarm, dtype=bool)
    for i in range(fault_start, len(a)):
        lo = max(0, i - n + 1)
        if a[lo:i + 1].sum() >= m:
            return i - fault_start
    return None


def false_alarm_rate(alarm: np.ndarray, fault_start: int, m: int = 3,
                     n: int = 5) -> float:
    """Sustained alarms per 1000 samples during the PRE-FAULT period."""
    a = np.asarray(alarm, dtype=bool)[:fault_start]
    if len(a) == 0:
        return 0.0
    hits, i = 0, 0
    while i < len(a):
        lo = max(0, i - n + 1)
        if a[lo:i + 1].sum() >= m:
            hits += 1
            i += n                       # one episode, not one per sample
        else:
            i += 1
    return 1000.0 * hits / len(a)


# ---------------------------------------------------------------------------
# dynamic PCA
# ---------------------------------------------------------------------------

def lag_embed(x: np.ndarray, lags: int) -> np.ndarray:
    """Stack l lagged copies alongside the current sample.

    Row t becomes [x_t, x_{t-1}, ..., x_{t-l}], so a model fitted on this sees
    the process's DYNAMICS rather than only its instantaneous cross-section.

    Why it matters, and it is not a refinement. Static PCA assumes the rows are
    independent. Process data is autocorrelated, so the effective sample size
    behind every control limit is smaller than the row count implies -- a limit
    set from an empirical quantile of 500 correlated samples is a limit set from
    rather fewer than 500 pieces of information, and it comes out too tight. The
    visible symptom is a false-alarm rate above the nominal one, which reads as
    a sensitive detector rather than as a miscalibrated one.

    Lag embedding is the standard answer (Ku, Storch & Georgakis 1995). It does
    not remove the autocorrelation; it moves the dynamics INSIDE the model, so
    that what is left over is closer to independent.

    The cost is real and is not hidden: l lags multiply the column count by
    (l + 1), and the first l rows have no history and are dropped.
    """
    x = np.asarray(x, dtype=float)
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if lags == 0:
        return x
    n, p = x.shape
    if n <= lags:
        raise ValueError(f"{n} samples cannot support {lags} lags")
    return np.concatenate([x[lags - i: n - i] for i in range(lags + 1)], axis=1)


def autocorrelation(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """Per-column lag-`lag` autocorrelation. The evidence that DPCA is needed
    at all -- if this is near zero, lag embedding buys nothing and costs
    columns."""
    x = np.asarray(x, dtype=float)
    z = x - x.mean(0)
    sd = z.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = z / sd
    a, b = z[lag:], z[:-lag]
    return (a * b).mean(0)
