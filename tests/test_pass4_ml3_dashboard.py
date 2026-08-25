"""Pass 4: the fleet dashboard.

A dashboard is easy to test badly -- assert it produced some bytes and move on.
The tests here are about the two things that make it a monitoring screen rather
than a table dump: the ordering, and what it refuses to hide.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dashboard as D           # noqa: E402

OUT = ROOT / "out"
HAVE_DATA = (OUT / "cwru.json").exists() and (OUT / "process.json").exists()


def _rows(n_per_asset=4):
    """Three assets: a clear outer-race fault, a split call, and a healthy one."""
    def snap(fid, call, fault, expected, score, bands, sbp):
        return {"fid": fid, "call": call, "fault": fault, "expected": expected,
                "score": score, "size_in": 0.021, "load_hp": 1,
                "shaft_hz": 29.9, "sbp_BPFI": sbp,
                **{f"r_{k}": v for k, v in bands.items()}}
    rows = []
    for i in range(n_per_asset):
        rows.append(snap("A", "BPFO", "outer_race", "BPFO", 9.0,
                         {"BPFO": 90.0, "BPFI": 12.0, "BSF": 11.0, "FTF": 0.0}, 5.0))
        rows.append(snap("B", "BPFI" if i < 3 else "BSF", "ball", "BSF", 4.0,
                         {"BPFO": 12.0, "BPFI": 20.0, "BSF": 18.0, "FTF": 0.0}, 9.0))
        rows.append(snap("C", "healthy", "normal", None, 1.0,
                         {"BPFO": 3.0, "BPFI": 3.1, "BSF": 3.0, "FTF": 0.0}, 1.0))
    return rows


def test_snapshots_collapse_to_one_row_per_asset():
    fleet = D.fleet_from_rows(_rows())
    assert [r["asset"] for r in sorted(fleet, key=lambda r: r["asset"])] == \
        ["A", "B", "C"]
    assert all(r["n_snapshots"] == 4 for r in fleet)


def test_a_split_call_carries_its_agreement():
    """A bearing called BPFI on 15 of 16 snapshots and one called BPFI on 9 of
    16 are different situations, and a single label hides which."""
    fleet = {r["asset"]: r for r in D.fleet_from_rows(_rows())}
    assert fleet["A"]["agreement"] == pytest.approx(1.0)
    assert fleet["B"]["call"] == "BPFI"
    assert fleet["B"]["agreement"] == pytest.approx(0.75)


def test_faults_sort_above_abstentions_above_healthy():
    """The ordering IS the product. An operator has ten minutes."""
    rows = _rows()
    for r in rows:
        if r["fid"] == "C":
            r["call"] = "indeterminate"
    order = [r["state"] for r in D.fleet_from_rows(rows + _rows()[2::3])]
    assert order.index("faulty") < order.index("indeterminate")


def test_within_a_severity_the_strongest_evidence_comes_first():
    fleet = D.fleet_from_rows(_rows())
    faulty = [r for r in fleet if r["state"] == "faulty"]
    assert [r["asset"] for r in faulty] == ["A", "B"]
    assert faulty[0]["score"] >= faulty[1]["score"]


def test_the_call_is_graded_only_where_a_race_is_expected():
    fleet = {r["asset"]: r for r in D.fleet_from_rows(_rows())}
    assert fleet["A"]["correct"] is True
    assert fleet["B"]["correct"] is False      # called BPFI, expected BSF
    assert fleet["C"]["correct"] is None       # nothing to be right about


def test_the_page_shows_the_evidence_and_not_just_the_call(tmp_path):
    """"BPFI on MC-14" is an assertion. The band ratios are the reason, and they
    are what lets a millwright disagree."""
    out = D.render(tmp_path / "d.html", {"rows": _rows(), "scores": {}},
                   {"runs": []})
    html = (tmp_path / "d.html").read_text(encoding="utf-8")
    assert out["assets"] == 3
    for band in ("BPFO", "BPFI", "BSF", "FTF"):
        assert band in html
    assert "sbp" in html
    assert "90.0" in html                      # the winning band's ratio


def test_an_abstention_is_rendered_rather_than_dropped(tmp_path):
    rows = _rows()
    for r in rows:
        if r["fid"] == "C":
            r["call"] = "indeterminate"
    out = D.render(tmp_path / "d.html", {"rows": rows, "scores": {}},
                   {"runs": []})
    html = (tmp_path / "d.html").read_text(encoding="utf-8")
    assert out["indeterminate"] == 1
    assert out["assets"] == 3
    assert "indeterminate" in html


def test_the_page_makes_no_network_requests(tmp_path):
    D.render(tmp_path / "d.html", {"rows": _rows(), "scores": {}}, {"runs": []})
    html = (tmp_path / "d.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="(?!#)([^"]+)"', html)
    assert refs == [], refs
    assert "//" not in re.sub(r"https?://[^\"']*", "", html).split("<footer")[0] \
        or "http" not in html.split("<footer")[0]


def test_the_confusion_diagonal_maps_fault_names_to_defect_frequencies(tmp_path):
    """The truth labels are fault NAMES and the calls are defect FREQUENCIES.
    A naive truth == call diagonal would mark every correct diagnosis wrong."""
    scores = {"confusion": {"inner_race": {"BPFI": 10, "BPFO": 2},
                            "normal": {"healthy": 5, "BSF": 1}}}
    D.render(tmp_path / "d.html", {"rows": _rows(), "scores": scores},
             {"runs": []})
    html = (tmp_path / "d.html").read_text(encoding="utf-8")
    body = html.split('class="conf"')[1]
    # the inner_race -> BPFI cell must carry the diagonal class
    assert 'class="n diag">10<' in body
    assert 'class="n miss">2<' in body


def test_a_process_run_names_the_tag_and_not_only_the_statistic(tmp_path):
    run = {"fault": "cooling fouling", "description": "d", "fault_start": 100,
           "n_components": 3, "variance_explained": 0.9,
           "detectors": {"PCA T2 only": {"delay": 523, "false_per_1000": 0.0,
                                         "alarm_frac_after": 0.05},
                         "PCA SPE only": {"delay": None, "false_per_1000": 0.0,
                                          "alarm_frac_after": 0.01}},
           "first_diagnosis": {"sample": 18, "statistic": "SPE", "t2": 3.2,
                               "spe": 5.1,
                               "top_tags": [
                                   {"tag": "cool_water_outlet_temp",
                                    "contribution": 2.7},
                                   {"tag": "reactor_level",
                                    "contribution": 1.8}]}}
    D.render(tmp_path / "d.html", {"rows": _rows(), "scores": {}},
             {"runs": [run]})
    html = (tmp_path / "d.html").read_text(encoding="utf-8")
    assert "cool_water_outlet_temp" in html
    assert "reactor_level" in html
    assert "never" in html            # the SPE detector that did not fire


def test_a_detector_that_never_fires_says_so_rather_than_showing_zero(tmp_path):
    run = {"fault": "f", "detectors": {"a": {"delay": None}}, "first_diagnosis": {}}
    D.render(tmp_path / "d.html", {"rows": _rows(), "scores": {}},
             {"runs": [run]})
    html = (tmp_path / "d.html").read_text(encoding="utf-8")
    assert ">never<" in html


# ---------------------------------------------------------------------------
# against the real measured results
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_DATA, reason="run the CWRU and process stages first")
def test_it_renders_from_the_real_outputs():
    out = D.render_from_out(OUT)
    assert out["assets"] == 40
    assert out["process_runs"] >= 1
    assert out["bytes"] > 20000


@pytest.mark.skipif(not HAVE_DATA, reason="run the CWRU and process stages first")
def test_the_fleet_view_is_now_green_where_it_should_be():
    """This test used to assert the opposite, and that was the point of it.

    The fleet view's finding was that NOTHING was green: four healthy bearings
    and none called healthy. Pass 5 traced that to an uncalibrated gate inside
    the healthy distribution and fixed it (see run_healthy_gate.py), so the
    assertion is inverted -- and the callout must now be absent, because a page
    that warns about a problem it no longer has is as misleading as one that
    hides a problem it does.
    """
    out = D.render_from_out(OUT)
    assert out["truly_healthy_assets"] > 0
    assert out["healthy_assets_called_healthy"] == out["truly_healthy_assets"]
    html = (OUT / "fleet_dashboard.html").read_text(encoding="utf-8")
    assert "Nothing on this screen is green" not in html


def test_the_all_red_callout_still_fires_when_it_should(tmp_path):
    """The mechanism, kept alive on synthetic data now that the real fleet no
    longer triggers it."""
    rows = []
    for i in range(4):
        rows.append({"fid": "H", "call": "BPFO", "fault": "normal",
                     "expected": None, "score": 9.0, "size_in": 0.0,
                     "load_hp": 0, "shaft_hz": 29.9, "sbp_BPFI": 1.0,
                     "r_BPFO": 90.0, "r_BPFI": 10.0, "r_BSF": 9.0, "r_FTF": 0.0})
    out = D.render(tmp_path / "d.html", {"rows": rows, "scores": {}},
                   {"runs": []})
    assert out["truly_healthy_assets"] == 1
    assert out["healthy_assets_called_healthy"] == 0
    assert "Nothing on this screen is green" in (
        tmp_path / "d.html").read_text(encoding="utf-8")


@pytest.mark.skipif(not HAVE_DATA, reason="run the CWRU and process stages first")
def test_every_asset_in_the_data_appears_exactly_once():
    cwru = json.loads((OUT / "cwru.json").read_text(encoding="utf-8"))
    fleet = D.fleet_from_rows(cwru["rows"])
    ids = [r["asset"] for r in fleet]
    assert len(ids) == len(set(ids)) == cwru["n_files"]
