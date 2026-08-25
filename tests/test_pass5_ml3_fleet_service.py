"""Pass 5: the fleet dashboard as a service.

The tests that matter are about re-arming. An acknowledgement that survives the
evidence getting worse is how a monitoring screen becomes a screen nobody reads,
and it is the easy thing to get wrong because the happy path looks identical.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fleet_service as FS      # noqa: E402


def _fleet(n=4, call="BPFI", score=10.0):
    return [{"asset": f"A{i}", "call": call, "state": "faulty",
             "score": score + i, "agreement": 1.0, "truth": "inner_race"}
            for i in range(n)]


@pytest.fixture
def store(tmp_path):
    return FS.FleetStore(tmp_path / "f.db")


# --- history -----------------------------------------------------------------

def test_a_run_is_appended_not_replaced(store):
    store.record_run(_fleet(), run_id="a")
    store.record_run(_fleet(score=20.0), run_id="b")
    h = store.history("A0")
    assert len(h) == 2
    assert h[0]["score"] > h[1]["score"], "newest first"


def test_changed_since_names_what_moved(store):
    store.record_run(_fleet(), run_id="mon")
    tue = _fleet()
    tue[1]["call"] = "BPFO"
    store.record_run(tue, run_id="tue")
    ch = store.changed_since("mon")
    assert ch["available"] and ch["n_changed"] == 1
    c = ch["changed"][0]
    assert c["asset"] == "A1"
    assert c["from_call"] == "BPFI" and c["to_call"] == "BPFO"


def test_changed_since_refuses_an_unknown_or_newest_run(store):
    store.record_run(_fleet(), run_id="only")
    assert store.changed_since("only")["available"] is False
    store.record_run(_fleet(), run_id="second")
    assert store.changed_since("nonsense")["available"] is False


def test_a_new_asset_is_reported_separately(store):
    store.record_run(_fleet(n=3), run_id="a")
    store.record_run(_fleet(n=4), run_id="b")
    ch = store.changed_since("a")
    assert ch["appeared"] == ["A3"]
    assert ch["disappeared"] == []


# --- acknowledgement ---------------------------------------------------------

def test_an_acknowledgement_needs_a_reason(store):
    store.record_run(_fleet(), run_id="a")
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="reason"):
            store.acknowledge("A0", "someone", bad)


def test_acknowledging_an_unknown_asset_is_refused(store):
    store.record_run(_fleet(), run_id="a")
    with pytest.raises(KeyError):
        store.acknowledge("NOPE", "someone", "because")


def test_an_acknowledgement_records_what_was_acknowledged(store):
    """Not just the asset. Acknowledging an ASSET is the design that turns a
    monitoring screen into a screen nobody reads."""
    store.record_run(_fleet(), run_id="a")
    out = store.acknowledge("A0", "millwright-7", "inspected, replace at stop")
    assert out["acknowledged_call"] == "BPFI"
    assert out["acknowledged_score"] == pytest.approx(10.0)
    assert store.ack_status("A0")["acknowledged"] is True


def test_it_rearms_when_the_call_changes(store):
    store.record_run(_fleet(), run_id="a")
    store.acknowledge("A0", "w", "accepted this fault")
    store.record_run(_fleet(call="BPFO"), run_id="b")
    st = store.ack_status("A0")
    assert st["acknowledged"] is False
    assert "call changed" in st["why"]


def test_it_rearms_when_the_evidence_worsens(store):
    store.record_run(_fleet(), run_id="a")
    store.acknowledge("A0", "w", "accepted at this severity")
    store.record_run(_fleet(score=30.0), run_id="b")
    st = store.ack_status("A0")
    assert st["acknowledged"] is False
    assert "worsened" in st["why"]


def test_it_does_not_rearm_on_a_small_change(store):
    """Re-arming on any movement at all is the same as not being able to
    acknowledge anything."""
    store.record_run(_fleet(score=10.0), run_id="a")
    store.acknowledge("A0", "w", "accepted", worsen_frac=0.25)
    store.record_run(_fleet(score=11.0), run_id="b")      # +10%
    assert store.ack_status("A0")["acknowledged"] is True


def test_it_does_not_rearm_when_the_evidence_improves(store):
    store.record_run(_fleet(score=20.0), run_id="a")
    store.acknowledge("A0", "w", "accepted")
    store.record_run(_fleet(score=5.0), run_id="b")
    assert store.ack_status("A0")["acknowledged"] is True


def test_it_expires(store):
    store.record_run(_fleet(), run_id="a")
    store.acknowledge("A0", "w", "accepted", ttl_hours=72)
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=100)
    st = store.ack_status("A0", now=later)
    assert st["acknowledged"] is False
    assert "older than its" in st["why"]


def test_rearming_is_recorded_not_just_computed(store):
    """So 'why did this come back' has an answer later."""
    store.record_run(_fleet(), run_id="a")
    store.acknowledge("A0", "w", "accepted")
    store.record_run(_fleet(score=30.0), run_id="b")
    store.ack_status("A0")
    row = store.conn.execute(
        "SELECT rearmed_ts, rearmed_why FROM acknowledgement "
        "WHERE asset='A0'").fetchone()
    assert row["rearmed_ts"] and "worsened" in row["rearmed_why"]


def test_an_acknowledged_asset_is_marked_not_hidden(store):
    """A screen that hides what somebody accepted cannot be audited."""
    store.record_run(_fleet(), run_id="a")
    store.acknowledge("A0", "millwright-7", "accepted")
    now = {r["asset"]: r for r in store.fleet_now()}
    assert "A0" in now
    assert now["A0"]["acknowledged"] is True
    assert now["A0"]["ack_by"] == "millwright-7"
    assert now["A0"]["ack_reason"] == "accepted"


# --- over HTTP ---------------------------------------------------------------

def _call(url, path, body=None):
    hdr = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, headers=hdr)
    try:
        with urllib.request.urlopen(req) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


@pytest.fixture
def live(store):
    store.record_run(_fleet(), run_id="a")
    h = FS.serve(store, "<h1>fleet</h1>")
    yield h
    h["server"].shutdown()


def test_the_fleet_endpoint_serves_current_state(live):
    c, b = _call(live["url"], "/api/fleet")
    assert c == 200 and len(b) == 4
    assert all("acknowledged" in r for r in b)


def test_acknowledging_over_http_requires_every_field(live):
    c, b = _call(live["url"], "/api/acknowledge", {"asset": "A0"})
    assert c == 400 and "missing" in b["error"]


def test_acknowledging_over_http_works_and_shows_up(live):
    c, _ = _call(live["url"], "/api/acknowledge",
                 {"asset": "A0", "who": "w", "reason": "inspected"})
    assert c == 200
    _, fleet = _call(live["url"], "/api/fleet")
    a0 = next(r for r in fleet if r["asset"] == "A0")
    assert a0["acknowledged"] is True


def test_an_empty_reason_is_a_400_not_a_500(live):
    c, b = _call(live["url"], "/api/acknowledge",
                 {"asset": "A0", "who": "w", "reason": "  "})
    assert c == 400 and "reason" in b["error"]


def test_an_unknown_route_is_a_404(live):
    assert _call(live["url"], "/api/nope")[0] == 404


def test_it_does_not_import_se2():
    """The pattern transfers; the code does not."""
    src = (ROOT / "src" / "fleet_service.py").read_text(encoding="utf-8")
    assert "import server" not in src
    assert "se2" not in src.lower().replace("se-2", "")


def test_the_limits_are_stated():
    joined = " ".join(FS.LIMITS)
    assert "No authentication" in joined
    assert "Polling, not push" in joined
