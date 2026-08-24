"""Tests for the process-side (multivariate) half."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pca_monitor as PCA  # noqa: E402
import process as PROC  # noqa: E402


# ---------------------------------------------------------------------------
# the simulator
# ---------------------------------------------------------------------------

def test_relationships_are_real_not_imposed():
    """Derived tags are computed FROM their drivers, so the covariance structure
    is a consequence of the process."""
    x = PROC.simulate(1500, "none", seed=0).x
    c = np.corrcoef(x, rowvar=False)
    i_press = PROC.TAGS.index("reactor_press")
    i_feed = PROC.TAGS.index("feed_d_flow")
    assert abs(c[i_press, i_feed]) > 0.5, "pressure must track feed rate"


def test_every_fault_stays_inside_the_marginal_limits():
    """The regime the whole argument lives in.

    An earlier version pushed cooling flow 115 sigma from its training mean; at
    that size every univariate chart fires instantly and there is nothing
    multivariate to demonstrate.
    """
    train = PROC.simulate(1500, "none", seed=3)
    mu, sd = train.x.mean(0), train.x.std(0)
    for fault in ("feed_composition_step", "heat_exchanger_fouling",
                  "kinetics_drift"):
        te = PROC.simulate(1500, fault, seed=503)
        z = np.abs((te.x[-200:].mean(0) - mu) / sd)
        assert z.max() < 4.0, f"{fault} moves a tag {z.max():.1f} sigma"


def test_production_rate_gives_cooling_flow_a_wide_natural_range():
    """Without a varying load, marginal limits are tight and univariate charts
    are sufficient for everything."""
    x = PROC.simulate(1500, "none", seed=1).x
    cf = x[:, PROC.TAGS.index("cool_water_flow")]
    assert np.ptp(cf) > 5.0


def test_an_unknown_fault_is_refused():
    with pytest.raises(ValueError, match="unknown fault"):
        PROC.simulate(100, "not_a_fault")


# ---------------------------------------------------------------------------
# PCA: T2 vs SPE
# ---------------------------------------------------------------------------

def _fit(seed=0, n=1500):
    train = PROC.simulate(n, "none", seed=seed)
    return PROC.simulate, PCA.PCAMonitor(variance_target=0.90).fit(train.x), train


def test_t2_and_spe_are_quiet_on_normal_data():
    _, mon, _ = _fit()
    test = PROC.simulate(1500, "none", seed=999)
    sc = mon.score(test.x)
    assert sc["t2_alarm"].mean() < 0.10
    assert sc["spe_alarm"].mean() < 0.10


def test_moving_along_the_model_plane_raises_t2_and_not_spe():
    """Moving ALONG the model plane is a T² event by construction.

    The push is scaled by sqrt(eigenvalue), because T² divides each score by its
    component's variance: a fixed push along a HIGH-variance component barely
    moves T² at all. Getting that wrong is how a test like this ends up asserting
    something the statistic never promised — my first version pushed 6 units
    along PC1 and produced T² = 8.5 against a limit of 18.8.
    """
    _, mon, train = _fit()
    direction = mon.P[:, 0] * np.sqrt(mon.eig[0])
    x = train.x[:50] + 8.0 * (direction * mon.sd)
    sc = mon.score(x)
    assert sc["t2"].mean() > mon.t2_limit, "a push along the plane must raise T²"
    assert sc["spe"].mean() < mon.spe_limit, "and must leave SPE alone"


def test_breaking_a_relationship_moves_spe():
    """Perturbing ORTHOGONALLY to the model plane is an SPE event."""
    _, mon, train = _fit()
    x = train.x[:50].copy()
    z = (x - mon.mu) / mon.sd
    e = z - (z @ mon.P) @ mon.P.T             # the residual direction
    n = np.linalg.norm(e, axis=1, keepdims=True)
    z_bad = z + 8.0 * e / np.maximum(n, 1e-9)
    sc = mon.score(z_bad * mon.sd + mon.mu)
    assert sc["spe"].mean() > mon.spe_limit


def test_spe_contributions_sum_exactly_to_spe():
    """Exact, not approximate — unlike SHAP or occlusion."""
    _, mon, train = _fit()
    x = train.x[:20]
    sc = mon.score(x)
    assert np.allclose(mon.spe_contributions(x).sum(1), sc["spe"], rtol=1e-9)


def test_components_are_chosen_by_variance_target():
    _, mon, _ = _fit()
    assert mon.explained >= 0.90
    assert 1 <= mon.k < len(PROC.TAGS)


# ---------------------------------------------------------------------------
# residual model
# ---------------------------------------------------------------------------

def test_residuals_are_small_on_normal_data_and_large_when_a_link_breaks():
    train = PROC.simulate(1500, "none", seed=5)
    rm = PROC.ResidualModel().fit(train.x)
    normal = rm.residuals(PROC.simulate(1500, "none", seed=605).x)

    # Break one relationship by hand: hold cooling flow while the load moves.
    broken = PROC.simulate(1500, "none", seed=605)
    j = PROC.TAGS.index("cool_water_flow")
    x = broken.x.copy()
    x[:, j] = x[:, j].mean()
    assert np.abs(rm.residuals(x)).mean() > np.abs(normal).mean() * 1.5


def test_r2_says_which_residuals_are_worth_monitoring():
    """At R² near zero the residual IS the signal, and charting it is a
    univariate chart with extra steps."""
    train = PROC.simulate(1500, "none", seed=7)
    r2 = PROC.ResidualModel().fit(train.x).r2(train.x)
    assert r2.max() > 0.8, "a coupled process must have predictable tags"
    assert len(r2) == len(PROC.TAGS)


def test_ridge_is_more_stable_across_refits_than_ols():
    """The actual claim, tested as a comparison rather than a magic threshold.

    Process variables are collinear by construction — that is what makes them a
    process rather than eighteen independent sensors — so OLS coefficients swing
    between refits, and residuals that move between refits move for reasons that
    have nothing to do with the plant.

    An absolute bound on the coefficient drift would be a number I tuned until it
    passed; the comparison is the thing being asserted.
    """
    x1 = PROC.simulate(1500, "none", seed=8).x
    x2 = PROC.simulate(1500, "none", seed=9).x

    def drift(alpha):
        a = PROC.ResidualModel(alpha=alpha).fit(x1)
        b = PROC.ResidualModel(alpha=alpha).fit(x2)
        wa = np.concatenate([w for _, w in a.coefs])
        wb = np.concatenate([w for _, w in b.coefs])
        return float(np.abs(wa - wb).max())

    assert drift(1.0) < drift(1e-8), "ridge must be steadier than OLS"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def test_detection_requires_persistence_not_a_single_crossing():
    alarm = np.zeros(100, dtype=bool)
    alarm[50] = True                          # one spike
    assert PCA.detection_delay(alarm, 40, m=3, n=5) is None
    alarm[50:54] = True
    assert PCA.detection_delay(alarm, 40, m=3, n=5) is not None


def test_a_never_firing_detector_reports_none_not_zero():
    assert PCA.detection_delay(np.zeros(100, dtype=bool), 30) is None


def test_false_alarms_count_episodes_not_samples():
    """One sustained excursion is one alarm, not one per sample."""
    a = np.zeros(200, dtype=bool)
    a[20:60] = True
    assert PCA.false_alarm_rate(a, fault_start=200, m=3, n=5) < 60
