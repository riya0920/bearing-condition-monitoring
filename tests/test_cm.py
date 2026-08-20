"""ML-3 tests: bearing kinematics, envelope analysis, and the alarm state machine.

The fault-frequency test is checked against published values for the SKF 6205,
which is the bearing every public test rig uses -- so it is a real external check
rather than a self-consistency one.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bearing  # noqa: E402
import features  # noqa: E402
import health  # noqa: E402

# Published order values for an SKF 6205-2RS (9 balls, 7.94 mm, 39.04 mm, 0 deg).
PUBLISHED_ORDERS = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 2.357, "FTF": 0.3983}


def test_fault_frequencies_match_published_orders():
    g = bearing.BearingGeometry()
    orders = g.orders()
    for k, v in PUBLISHED_ORDERS.items():
        assert orders[k] == pytest.approx(v, abs=2e-3), k


def test_fault_frequencies_scale_linearly_with_shaft_speed():
    g = bearing.BearingGeometry()
    a = g.fault_frequencies(30.0)
    b = g.fault_frequencies(60.0)
    for k in ("BPFO", "BPFI", "BSF", "FTF"):
        assert b[k] == pytest.approx(2 * a[k])


def test_bpfo_and_bpfi_sum_to_element_count_times_shaft():
    """A kinematic identity: BPFO + BPFI = n * f_shaft. Good arithmetic check."""
    g = bearing.BearingGeometry()
    f = g.fault_frequencies(29.95)
    assert f["BPFO"] + f["BPFI"] == pytest.approx(g.n_elements * 29.95)


def test_the_harmonic_collision_is_real():
    """BPFO*3 and BPFI*2 sit inside any usable slip tolerance for this geometry --
    which is why the diagnosis needs sidebands and not just harmonic energy."""
    o = bearing.BearingGeometry().orders()
    gap = abs(o["BPFO"] * 3 - o["BPFI"] * 2) / (o["BPFI"] * 2)
    assert gap < 0.01


def test_envelope_spectrum_finds_bpfo_in_a_bpfo_signal():
    g = bearing.BearingGeometry()
    x = bearing.simulate_snapshot(g, 29.95, "BPFO", 1.0, np.random.default_rng(3))
    freqs, spec, _ = features.envelope_spectrum(x, (3000, 4000))
    m = (freqs > 50) & (freqs < 600)
    peak = float(freqs[m][np.argmax(spec[m])])
    expected = g.fault_frequencies(29.95)["BPFO"]
    assert peak == pytest.approx(expected, rel=0.03)


def test_healthy_signal_has_no_dominant_fault_line():
    g = bearing.BearingGeometry()
    x = bearing.simulate_snapshot(g, 29.95, None, 0.0, np.random.default_rng(4))
    freqs, spec, _ = features.envelope_spectrum(x, (3000, 4000))
    m = (freqs > 50) & (freqs < 600)
    # No line should tower over the local floor the way a real fault does.
    assert float(spec[m].max() / np.median(spec[m])) < 40


def test_shaft_speed_estimator_recovers_the_true_speed():
    g = bearing.BearingGeometry()
    for hz in (25.0, 29.95, 35.0):
        x = bearing.simulate_snapshot(g, hz, None, 0.0, np.random.default_rng(5))
        assert features.estimate_shaft_speed(x) == pytest.approx(hz, abs=1.0)


def test_severity_increases_the_fault_band_energy():
    g = bearing.BearingGeometry()
    prev = -1.0
    for sev in (0.0, 0.3, 0.7, 1.0):
        x = bearing.simulate_snapshot(g, 29.95, "BPFO", sev, np.random.default_rng(6))
        f = features.snapshot_features(x, g, band=(3000, 4000))
        assert f["env_BPFO_ratio"] > prev or sev == 0.0
        prev = f["env_BPFO_ratio"]


def test_run_to_failure_is_flat_before_onset():
    g = bearing.BearingGeometry()
    _, sev, _, truth = bearing.simulate_run_to_failure(g, "BPFO", n_cycles=100,
                                                       onset=50, seed=1)
    assert (sev[:50] == 0).all()
    assert sev[-1] > sev[60] > 0
    assert truth["onset_cycle"] == 50


# ---------------------------------------------------------------- health index

def _fake_feats(vals):
    return [{k: v for k, v in zip(health.INDEX_FEATURES, row)} for row in vals]


def test_health_index_is_100_at_the_baseline_median():
    base_rows = [[1.0, 1.0, 1.0, 3.0] for _ in range(40)]
    b = health.Baseline.fit(_fake_feats(base_rows))
    assert b.trusted
    assert health.health_index(_fake_feats([[1.0, 1.0, 1.0, 3.0]])[0], b) == pytest.approx(100.0)


def test_health_index_falls_as_features_exceed_the_baseline():
    rng = np.random.default_rng(0)
    rows = [[1.0 + rng.normal(0, 0.1), 1.0, 1.0, 3.0] for _ in range(60)]
    b = health.Baseline.fit(_fake_feats(rows))
    bad = _fake_feats([[10.0, 1.0, 1.0, 3.0]])[0]
    assert health.health_index(bad, b) < 50


def test_short_baseline_is_not_trusted():
    b = health.Baseline.fit(_fake_feats([[1.0, 1.0, 1.0, 3.0]] * 5))
    assert not b.trusted


def test_untrusted_baseline_yields_low_confidence_state():
    b = health.Baseline.fit(_fake_feats([[1.0, 1.0, 1.0, 3.0]] * 5))
    states = health.AlarmPolicy().run(np.full(50, 20.0), trusted=b.trusted)
    assert (states == int(health.State.LOW_CONFIDENCE)).all()


def test_hysteresis_prevents_flapping_on_a_borderline_score():
    """A score oscillating around the enter threshold must not toggle the alarm."""
    rng = np.random.default_rng(7)
    score = 85.0 + rng.normal(0, 1.2, 400)      # sits right on watch_enter
    states = health.AlarmPolicy().run(score, trusted=True)
    assert health.count_flaps(states, int(health.State.ALERT)) == 0


def test_a_sustained_collapse_reaches_critical_and_stays():
    score = np.concatenate([np.full(60, 98.0), np.full(60, 30.0)])
    states = health.AlarmPolicy().run(score, trusted=True)
    assert states[-1] == int(health.State.CRITICAL)
    first = health.first_sustained(states, int(health.State.ALERT))
    assert first is not None and 60 <= first <= 70


def test_smoothing_is_median_not_mean():
    """One bad acquisition must not move the score."""
    v = np.full(30, 50.0)
    v[15] = 1000.0
    s = health.smooth(v, 5)
    assert s.max() < 60.0
