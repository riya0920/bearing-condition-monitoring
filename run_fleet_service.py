"""Serve the fleet dashboard, with history and acknowledgement.

    python run_fleet_service.py            # records a run, serves, prints the URL
    python run_fleet_service.py --demo     # records three runs and shows the API

The static renderer stays: `run_dashboard.py` still writes a self-contained page,
because a page that needs a service running is a page that cannot be emailed.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import dashboard as D          # noqa: E402
import fleet_service as FS     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"


def _fleet():
    rows = json.loads((OUT / "cwru.json").read_text(encoding="utf-8"))["rows"]
    return D.fleet_from_rows(rows)


def _page() -> str:
    proc = json.loads((OUT / "process.json").read_text(encoding="utf-8"))
    cwru = json.loads((OUT / "cwru.json").read_text(encoding="utf-8"))
    tmp = OUT / "_served.html"
    D.render(tmp, cwru, proc)
    html = tmp.read_text(encoding="utf-8")
    tmp.unlink(missing_ok=True)
    return html


def demo() -> dict:
    """Three runs, an acknowledgement, and the acknowledgement re-arming."""
    import copy
    store = FS.FleetStore(OUT / "fleet_demo.db")
    fleet = _fleet()
    out = {"runs": [], "steps": []}

    out["runs"].append(store.record_run(fleet, run_id="monday"))

    tue = copy.deepcopy(fleet)
    flip = next(a for a in tue if a["state"] == "faulty")
    flip["call"] = "BSF" if flip["call"] != "BSF" else "BPFO"
    worse = next(a for a in tue if a["asset"] != flip["asset"]
                 and a["state"] == "faulty")
    worse["score"] *= 2.0
    out["runs"].append(store.record_run(tue, run_id="tuesday"))

    ch = store.changed_since("monday")
    out["steps"].append({"step": "what changed since monday",
                         "n_changed": ch["n_changed"],
                         "changed": ch["changed"][:3]})

    try:
        store.acknowledge(worse["asset"], "millwright-7", "")
    except ValueError as e:
        out["steps"].append({"step": "acknowledge with no reason",
                             "refused": str(e)})

    ack = store.acknowledge(worse["asset"], "millwright-7",
                            "inspected, spall confirmed, replace at next stop")
    out["steps"].append({"step": "acknowledge", **ack})
    out["steps"].append({"step": "status after acknowledging",
                         "acknowledged":
                             store.ack_status(worse["asset"])["acknowledged"]})

    wed = copy.deepcopy(tue)
    nxt = next(a for a in wed if a["asset"] == worse["asset"])
    nxt["score"] *= 2.0
    out["runs"].append(store.record_run(wed, run_id="wednesday"))
    st = store.ack_status(worse["asset"])
    out["steps"].append({"step": "status after the evidence worsened",
                         "acknowledged": st["acknowledged"],
                         "why": st.get("why")})

    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=100)
    store.acknowledge(worse["asset"], "millwright-7", "re-accepted after review")
    st2 = store.ack_status(worse["asset"], now=later)
    out["steps"].append({"step": "status 100 hours later",
                         "acknowledged": st2["acknowledged"],
                         "why": st2.get("why")})

    out["fleet_now_sample"] = store.fleet_now()[:3]
    out["limits"] = FS.LIMITS
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if "--demo" in sys.argv:
        (OUT / "fleet_demo.db").unlink(missing_ok=True)
        d = demo()
        (OUT / "fleet_service_demo.json").write_text(
            json.dumps(d, indent=2, default=str), encoding="utf-8")
        for s in d["steps"]:
            print(f"  {s['step']}: " + json.dumps(
                {k: v for k, v in s.items() if k != "step"}, default=str)[:150])
        print(f"wrote out/fleet_service_demo.json")
        return

    store = FS.FleetStore(OUT / "fleet.db")
    print(store.record_run(_fleet()))
    h = FS.serve(store, _page())
    print(f"serving on {h['url']}  (ctrl-c to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        h["server"].shutdown()


if __name__ == "__main__":
    main()
