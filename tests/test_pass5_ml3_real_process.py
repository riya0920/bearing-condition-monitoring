"""Pass 5: the process comparison on Tennessee Eastman and SKAB.

The most important tests here guard the METHOD, not the numbers. The first
version of this comparison set each detector at its own 99% limit and reported
that all five found all ten faults including the three nobody finds — a table
that was wrong in the most flattering possible direction. What stops that
recurring is the matched false-alarm budget and the sweep, so those are what get
pinned.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("rrp", ROOT / "run_real_process.py")
RP = importlib.util.module_from_spec(_spec)
sys.modules["rrp"] = RP
_spec.loader.exec_module(RP)

_fs = importlib.util.spec_from_file_location("fp", ROOT / "fetch_process.py")
FP = importlib.util.module_from_spec(_fs)
sys.modules["fp"] = FP
_fs.loader.exec_module(FP)

TE = ROOT / "data" / "PROCESS" / "te.npz"
RESULT = ROOT / "out" / "real_process.json"


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

def test_the_transpose_trap_is_handled():
    """`d00.dat` ships 52 x 500 while every other file is n x 52. It is the
    commonest way to get this dataset wrong and it produces an array of the
    right dtype and the wrong shape."""
    tall = np.arange(52 * 7, dtype=float).reshape(7, 52)
    wide = tall.T
    a = FP._parse_dat(b"\n".join(b" ".join(b"%.6e" % v for v in row)
                                 for row in tall), "tall")
    b = FP._parse_dat(b"\n".join(b" ".join(b"%.6e" % v for v in row)
                                 for row in wide), "wide")
    assert a.shape == (7, 52) and b.shape == (7, 52)
    assert np.allclose(a, b)


def test_a_file_with_the_wrong_variable_count_is_refused():
    bad = np.zeros((10, 9))
    with pytest.raises(ValueError, match="52 variables"):
        FP._parse_dat(b"\n".join(b" ".join(b"%.1f" % v for v in r) for r in bad),
                      "bad")


def test_the_fault_onsets_are_named_constants():
    """Getting the onset wrong makes every detection delay meaningless, so it
    is not allowed to be a literal inside a loop."""
    assert FP.TE_TEST_ONSET == 160
    assert FP.TE_TRAIN_ONSET == 20
    assert FP.TE_SAMPLE_MINUTES == 3.0


def test_the_hard_faults_are_in_the_fetch_list():
    for f in (3, 9, 15):
        assert f in FP.TE_FAULTS
        assert "undetectable" in FP.TE_FAULTS[f]


def test_there_are_52_variable_names():
    assert len(FP.TE_NAMES) == 52
    assert len(set(FP.TE_NAMES)) == 52


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------

def _bank(seed=0, n=600, p=6):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    mix = rng.normal(size=(3, p))
    x = base @ mix + 0.1 * rng.normal(size=(n, p))
    return RP._Bank(x), x


def test_every_statistic_is_one_number_per_sample():
    """Otherwise the univariate detector gets 52 chances to alarm and T² gets
    one, which is not a comparison."""
    bank, x = _bank()
    for name, s in bank.stats(x).items():
        assert s.shape == (len(x),), name


def test_calibration_puts_every_detector_at_the_same_budget():
    bank, x = _bank()
    _, hold = _bank(seed=1)
    thr = RP.calibrate(bank, hold, target_per_1000=20.0)
    achieved = [v["achieved_per_1000"] for v in thr.values()]
    assert all(a <= 20.0 + 1e-9 for a in achieved), achieved
    assert len(thr) == len(RP.DETECTORS)


def test_a_looser_budget_never_raises_a_threshold():
    bank, _ = _bank()
    _, hold = _bank(seed=1)
    tight = RP.calibrate(bank, hold, 1.0)
    loose = RP.calibrate(bank, hold, 50.0)
    for n in RP.DETECTORS:
        assert loose[n]["threshold"] <= tight[n]["threshold"] + 1e-9, n


@pytest.mark.skipif(not TE.exists(), reason="run fetch_process.py first")
def test_calibration_is_what_stops_the_flattering_table():
    """The bug the first version of the script had, on the data it had it on.

    Gaussian toy data will not show this — its detectors all sit at zero false
    alarms at their own 99% quantile, which is exactly why the bug needed real
    tails to surface. So this runs on TE.
    """
    z = np.load(TE, allow_pickle=True)
    train, hold = z["train_normal"], z["test_normal"]
    bank = RP._Bank(train)

    # The buggy version's thresholds: each detector's own limit fitted on TRAIN,
    # then applied to a different normal run. That is not the same as the 99%
    # quantile of the holdout, which is ~1% by construction and would hide the
    # problem entirely.
    train_stats, hold_stats = bank.stats(train), bank.stats(hold)
    naive = {n: RP.PM.false_alarm_rate(
        hold_stats[n] > np.quantile(train_stats[n], RP.ALPHA),
        len(hold) + 1, m=RP.M_OF_N) for n in RP.DETECTORS}
    spread_before = max(naive.values()) - min(naive.values())

    cal = RP.calibrate(bank, hold, 5.0)
    got = [v["achieved_per_1000"] for v in cal.values()]
    spread_after = max(got) - min(got)

    assert spread_before > 10.0, naive
    assert spread_after < spread_before / 2, (naive, got)
    assert all(g <= 5.0 + 1e-9 for g in got), got


def test_summarise_can_exclude_faults_without_touching_the_rows():
    rows = [{"fault": f, "detectors": {n: {"delay": f, "false_per_1000": 0.0}
                                       for n in RP.DETECTORS}}
            for f in (1, 3, 9, 15)]
    everything = RP.summarise(rows)
    hard_only = RP.summarise(rows, exclude=(1,))
    assert everything[RP.DETECTORS[0]]["of"] == 4
    assert hard_only[RP.DETECTORS[0]]["of"] == 3
    assert len(rows) == 4, "summarise mutated its input"


def test_a_detector_that_never_fires_is_counted_as_never():
    rows = [{"fault": 1, "detectors": {n: {"delay": None, "false_per_1000": 0.0}
                                       for n in RP.DETECTORS}}]
    s = RP.summarise(rows)
    assert s[RP.DETECTORS[0]]["detected"] == 0
    assert s[RP.DETECTORS[0]]["median_delay"] is None


# ---------------------------------------------------------------------------
# against the fetched data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TE.exists(), reason="run fetch_process.py first")
def test_the_te_data_has_the_shape_the_literature_describes():
    z = np.load(TE, allow_pickle=True)
    assert z["train_normal"].shape[1] == 52
    assert z["train_normal"].shape[0] == 500
    assert z["test_normal"].shape == (960, 52)
    assert int(z["test_onset"]) == 160


@pytest.mark.skipif(not RESULT.exists(), reason="run run_real_process.py first")
def test_the_hard_faults_are_not_found_quickly_at_a_tight_budget():
    """The sanity check the first version failed. At one alarm per 1000, faults
    3, 9 and 15 must take a long time or never arrive — a detector that finds
    them instantly is reporting its own threshold."""
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    tight = next(s for s in d["te"]["sweep"] if s["budget_per_1000"] == 1.0)
    for n in RP.DETECTORS:
        v = tight["hard"][n]
        assert v["median_delay"] is None or v["median_delay"] > 50, (n, v)


@pytest.mark.skipif(not RESULT.exists(), reason="run run_real_process.py first")
def test_the_detectable_faults_have_no_resolution_left():
    """Everything sits at the m-of-n floor, which is why the hard faults are
    the only place the methods can be compared."""
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    for s in d["te"]["sweep"]:
        for n in RP.DETECTORS:
            v = s["detectable"][n]
            assert v["detected"] == v["of"]
            assert v["median_delay"] <= 7


@pytest.mark.skipif(not RESULT.exists(), reason="run run_real_process.py first")
def test_the_ranking_is_not_stable_across_operating_points():
    """The finding. If this ever stops holding the write-up has to change."""
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    winners = []
    for s in d["te"]["sweep"]:
        cand = [(s["hard"][n]["median_delay"], n) for n in RP.DETECTORS
                if s["hard"][n]["median_delay"] is not None]
        winners.append(min(cand)[1])
    assert len(set(winners)) > 1, f"ranking was stable: {winners}"


@pytest.mark.skipif(not RESULT.exists(), reason="run run_real_process.py first")
def test_the_univariate_detector_is_never_beaten_on_detectable_faults():
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    for s in d["te"]["sweep"]:
        uni = s["detectable"]["univariate 3-sigma (the wall of charts)"]["median_delay"]
        for n in RP.DETECTORS:
            assert s["detectable"][n]["median_delay"] >= uni - 1e-9, (n, s)


@pytest.mark.skipif(not RESULT.exists(), reason="run run_real_process.py first")
def test_skab_is_reported_as_a_null_rather_than_dropped():
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    if not d["skab"].get("available") or not d["skab"]["rows"]:
        pytest.skip("SKAB not fetched")
    s = d["skab_summary"]
    assert all(s[n]["detected"] == s[n]["of"] for n in RP.DETECTORS)
    doc = (ROOT / "docs" / "REAL_PROCESS.md").read_text(encoding="utf-8")
    assert "null result" in doc and "separates nothing" in doc
