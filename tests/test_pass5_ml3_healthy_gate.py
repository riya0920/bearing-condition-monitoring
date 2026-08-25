"""Pass 5: the healthy gate, which had never been calibrated.

The gate is now a value chosen from a measured plateau rather than a constant.
These tests pin the plateau's existence and the leave-one-out procedure, because
those are the two things that make changing a shipped default a calibration
rather than a tune.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as FE                    # noqa: E402

_spec = importlib.util.spec_from_file_location("hg", ROOT / "run_healthy_gate.py")
HG = importlib.util.module_from_spec(_spec)
sys.modules["hg"] = HG
_spec.loader.exec_module(HG)

CWRU = ROOT / "out" / "cwru.json"
RESULT = ROOT / "out" / "healthy_gate.json"


def _rows():
    return json.loads(CWRU.read_text(encoding="utf-8"))["rows"]


def test_the_shipped_gate_is_the_calibrated_one():
    import inspect
    sig = inspect.signature(FE.diagnose)
    assert sig.parameters["min_ratio"].default == HG.NEW_GATE == 6.0


def test_the_gate_still_refuses_when_nothing_is_there():
    """Raising it must not break the first gate's purpose."""
    quiet = {"env_BPFO_ratio": 2.0, "env_BPFI_ratio": 2.1,
             "env_BSF_ratio": 1.9, "env_BSF2_ratio": 1.5, "sbp_BPFI": 0.2}
    assert FE.diagnose(quiet)[0] == "healthy"


def test_a_loud_fault_is_still_named():
    loud = {"env_BPFO_ratio": 90.0, "env_BPFI_ratio": 12.0,
            "env_BSF_ratio": 11.0, "env_BSF2_ratio": 8.0, "sbp_BPFI": 3.0}
    assert FE.diagnose(loud)[0] == "BPFO"


def test_a_spectrum_between_the_old_and_new_gate_flips():
    """The whole point: 5.0 was a fault under the old gate and is healthy now."""
    mid = {"env_BPFO_ratio": 5.0, "env_BPFI_ratio": 3.0, "env_BSF_ratio": 3.0,
           "env_BSF2_ratio": 2.0, "sbp_BPFI": 0.5}
    assert FE.diagnose(mid, min_ratio=4.0)[0] != "healthy"
    assert FE.diagnose(mid, min_ratio=6.0)[0] == "healthy"


@pytest.mark.skipif(not CWRU.exists(), reason="run validate_cwru.py first")
def test_raising_the_gate_is_monotone_on_the_healthy_side():
    rows = _rows()
    prev = -1.0
    for g in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        h = HG._score(rows, g)["healthy_called_healthy"]
        assert h >= prev - 1e-9, f"healthy accuracy fell when raising to {g}"
        prev = h


@pytest.mark.skipif(not CWRU.exists(), reason="run validate_cwru.py first")
def test_the_old_gate_sat_inside_the_healthy_distribution():
    """The cause, not a symptom. If this stops holding, the finding is stale."""
    d = HG.band_ratio_distributions(_rows())
    assert d["healthy"]["p75"] > d["old_gate"]
    assert d["healthy_above_old_gate"] > 0.5


@pytest.mark.skipif(not CWRU.exists(), reason="run validate_cwru.py first")
def test_the_plateau_exists_and_contains_the_new_gate():
    """A single best value on four recordings would be an overfit. A wide flat
    band that the choice sits inside is not."""
    rows = _rows()
    grid = [round(3.0 + 0.25 * i, 2) for i in range(45)]
    pl = HG._plateau(HG.sweep(rows, grid))
    assert pl["exists"] is True
    assert pl["width"] >= 2.0, pl
    assert pl["lo"] < HG.NEW_GATE < pl["hi"], pl


@pytest.mark.skipif(not CWRU.exists(), reason="run validate_cwru.py first")
def test_the_new_gate_costs_nothing_on_the_faulty_side():
    rows = _rows()
    old = HG._score(rows, HG.OLD_GATE)
    new = HG._score(rows, HG.NEW_GATE)
    assert new["faulty_correct_race"] >= old["faulty_correct_race"] - 1e-9
    assert new["assets_faulty_called_healthy"] == old["assets_faulty_called_healthy"]
    assert new["assets_healthy_called_healthy"] > old["assets_healthy_called_healthy"]


@pytest.mark.skipif(not CWRU.exists(), reason="run validate_cwru.py first")
def test_leave_one_out_holds_the_scored_file_out_of_the_choice():
    """Otherwise the threshold is scored on its own training set."""
    rows = _rows()
    grid = [round(3.0 + 0.25 * i, 2) for i in range(45)]
    loo = HG.leave_one_healthy_file_out(rows, grid)
    assert loo["n_healthy_files"] == 4
    held = [f["held_out_file"] for f in loo["folds"]]
    assert len(set(held)) == 4
    assert loo["gate_spread"] <= 1.0, loo


@pytest.mark.skipif(not CWRU.exists(), reason="run validate_cwru.py first")
def test_every_fold_lands_inside_the_plateau():
    """The combination that makes this a calibration: stable across folds AND
    insensitive to getting it exactly right."""
    rows = _rows()
    grid = [round(3.0 + 0.25 * i, 2) for i in range(45)]
    pl = HG._plateau(HG.sweep(rows, grid))
    loo = HG.leave_one_healthy_file_out(rows, grid)
    for f in loo["folds"]:
        assert pl["lo"] <= f["chosen_min_ratio"] <= pl["hi"], f


@pytest.mark.skipif(not RESULT.exists(), reason="run run_healthy_gate.py first")
def test_the_document_records_the_conclusion_it_first_got_wrong():
    doc = (ROOT / "docs" / "HEALTHY_GATE.md").read_text(encoding="utf-8")
    assert "concluded the opposite" in doc
    assert "plateau" in doc
