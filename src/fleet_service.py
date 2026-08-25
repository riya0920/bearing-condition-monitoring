"""The fleet dashboard as a service: history, acknowledgement, and re-arming.

The README's item: *the dashboard is a static page, not a service. It is rendered
from the result files by a script; nothing polls, nothing pushes, there is no
acknowledge-and-clear, and an asset's history is not kept -- so it shows what the
fleet looks like now and cannot show what changed since yesterday, which is the
question a monitoring screen is usually asked.*

Three things, and the middle one is where the design is.

HISTORY. Every scoring run is appended, so an asset has a trajectory rather than
a current value. That is what makes "what changed since yesterday" answerable and
it is also what makes an acknowledgement meaningful -- see below.

ACKNOWLEDGEMENT IS A CLAIM ABOUT EVIDENCE, NOT ABOUT AN ASSET. "MC-14
acknowledged" is the design that turns a monitoring screen into a screen nobody
reads: the bearing gets worse, the alarm stays acknowledged, and the first anyone
hears is the failure. So an acknowledgement here records WHAT WAS ACKNOWLEDGED --
the call and the evidence strength at that moment -- and **re-arms itself when
the evidence changes materially**:

    the call changes            BPFO -> BPFI is a different fault
    the evidence worsens        beyond a stated fraction, default 25%
    the acknowledgement expires a stale acknowledgement is a forgotten one

Any of those and the asset comes back. That is the difference between silencing
an alarm and accepting a risk for a stated period.

WHY NOT AN IMPORT FROM SE-2. SE-2 has a working write-path server, and this
project's neighbours have argued repeatedly that a cross-project import is what
makes two systems impossible to deploy separately. The pattern transfers; the
code does not.
"""
from __future__ import annotations

import datetime as dt
import http.server
import json
import pathlib
import sqlite3
import threading
import time
import urllib.parse

DEFAULT_TTL_HOURS = 72.0
DEFAULT_WORSEN_FRAC = 0.25

SCHEMA = """
CREATE TABLE IF NOT EXISTS observation (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    run_id    TEXT NOT NULL,
    asset     TEXT NOT NULL,
    call      TEXT NOT NULL,
    state     TEXT NOT NULL,
    score     REAL NOT NULL,
    agreement REAL,
    truth     TEXT
);
CREATE INDEX IF NOT EXISTS ix_obs_asset ON observation (asset, id);

CREATE TABLE IF NOT EXISTS acknowledgement (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    asset         TEXT NOT NULL,
    who           TEXT NOT NULL,
    reason        TEXT NOT NULL,
    ack_call      TEXT NOT NULL,
    ack_score     REAL NOT NULL,
    ttl_hours     REAL NOT NULL,
    worsen_frac   REAL NOT NULL,
    rearmed_ts    TEXT,
    rearmed_why   TEXT
);
CREATE INDEX IF NOT EXISTS ix_ack_asset ON acknowledgement (asset, id);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts)


class FleetStore:
    """History and acknowledgements. One connection per thread."""

    def __init__(self, path):
        self.path = str(path)
        self._tl = threading.local()
        c = self.conn
        c.executescript(SCHEMA)
        c.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._tl, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode = WAL")
            self._tl.conn = c
        return c

    # -- history ----------------------------------------------------------
    def record_run(self, fleet: list, run_id: str | None = None,
                   ts: str | None = None) -> dict:
        """Append one scoring run. `fleet` is `dashboard.fleet_from_rows` output."""
        run_id = run_id or f"run-{int(time.time() * 1000)}"
        ts = ts or _now()
        rows = [(ts, run_id, r["asset"], r["call"], r["state"],
                 float(r["score"]), float(r.get("agreement") or 0.0),
                 r.get("truth")) for r in fleet]
        self.conn.executemany(
            "INSERT INTO observation (ts, run_id, asset, call, state, score, "
            "agreement, truth) VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return {"run_id": run_id, "ts": ts, "assets": len(rows)}

    def latest(self, asset: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM observation WHERE asset=? ORDER BY id DESC LIMIT 1",
            (asset,)).fetchone()
        return dict(r) if r else None

    def history(self, asset: str, limit: int = 50) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM observation WHERE asset=? ORDER BY id DESC LIMIT ?",
            (asset, limit))]

    def runs(self, limit: int = 20) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT run_id, ts, COUNT(*) n FROM observation "
            "GROUP BY run_id, ts ORDER BY MIN(id) DESC LIMIT ?", (limit,))]

    def changed_since(self, run_id: str) -> dict:
        """What is different between the newest run and a named earlier one.

        The question a monitoring screen is actually asked, and the one a static
        page cannot answer at all.
        """
        runs = self.runs(limit=200)
        if not runs:
            return {"available": False, "why": "no runs recorded"}
        newest = runs[0]["run_id"]
        if newest == run_id:
            return {"available": False, "why": "that is the newest run"}
        prev = {r["asset"]: dict(r) for r in self.conn.execute(
            "SELECT * FROM observation WHERE run_id=?", (run_id,))}
        cur = {r["asset"]: dict(r) for r in self.conn.execute(
            "SELECT * FROM observation WHERE run_id=?", (newest,))}
        if not prev:
            return {"available": False, "why": f"no run {run_id!r}"}

        changed, appeared, gone = [], [], []
        for a, c in cur.items():
            p = prev.get(a)
            if p is None:
                appeared.append(a)
                continue
            if p["call"] != c["call"] or p["state"] != c["state"]:
                changed.append({"asset": a, "from_call": p["call"],
                                "to_call": c["call"],
                                "from_state": p["state"], "to_state": c["state"],
                                "score_delta": c["score"] - p["score"]})
        gone = [a for a in prev if a not in cur]
        return {"available": True, "from_run": run_id, "to_run": newest,
                "changed": changed, "appeared": appeared, "disappeared": gone,
                "n_changed": len(changed)}

    # -- acknowledgement --------------------------------------------------
    def acknowledge(self, asset: str, who: str, reason: str,
                    ttl_hours: float = DEFAULT_TTL_HOURS,
                    worsen_frac: float = DEFAULT_WORSEN_FRAC) -> dict:
        """Accept a risk, for a stated period, against stated evidence.

        A reason is required. "Acknowledged" with no reason is indistinguishable
        from "dismissed", and six months later nobody can tell which it was.
        """
        if not reason or not reason.strip():
            raise ValueError(
                "an acknowledgement needs a reason: without one it is "
                "indistinguishable from dismissing the alarm")
        cur = self.latest(asset)
        if cur is None:
            raise KeyError(f"no observations for asset {asset!r}")
        self.conn.execute(
            "INSERT INTO acknowledgement (ts, asset, who, reason, ack_call, "
            "ack_score, ttl_hours, worsen_frac) VALUES (?,?,?,?,?,?,?,?)",
            (_now(), asset, who, reason.strip(), cur["call"],
             float(cur["score"]), float(ttl_hours), float(worsen_frac)))
        self.conn.commit()
        return {"asset": asset, "acknowledged_call": cur["call"],
                "acknowledged_score": cur["score"], "ttl_hours": ttl_hours}

    def _open_ack(self, asset: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM acknowledgement WHERE asset=? AND rearmed_ts IS NULL "
            "ORDER BY id DESC LIMIT 1", (asset,)).fetchone()
        return dict(r) if r else None

    def ack_status(self, asset: str, now: dt.datetime | None = None) -> dict:
        """Is this asset acknowledged, and if not any more, why not.

        Re-arming is evaluated on READ rather than on a timer, so a service that
        is not running does not silently keep an asset quiet.
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        ack = self._open_ack(asset)
        if ack is None:
            return {"acknowledged": False, "reason": None}
        cur = self.latest(asset)
        if cur is None:
            return {"acknowledged": True, "ack": ack}

        why = None
        if cur["call"] != ack["ack_call"]:
            why = (f"the call changed from {ack['ack_call']} to {cur['call']} "
                   "-- a different fault was not what was accepted")
        elif cur["score"] > ack["ack_score"] * (1.0 + ack["worsen_frac"]):
            why = (f"the evidence worsened by more than "
                   f"{ack['worsen_frac'] * 100:.0f}% "
                   f"({ack['ack_score']:.1f} -> {cur['score']:.1f})")
        elif (now - _parse(ack["ts"])).total_seconds() > ack["ttl_hours"] * 3600:
            why = (f"the acknowledgement is older than its "
                   f"{ack['ttl_hours']:.0f}-hour term")

        if why:
            self.conn.execute(
                "UPDATE acknowledgement SET rearmed_ts=?, rearmed_why=? "
                "WHERE id=?", (_now(), why, ack["id"]))
            self.conn.commit()
            return {"acknowledged": False, "rearmed": True, "why": why,
                    "was": ack}
        return {"acknowledged": True, "ack": ack}

    def fleet_now(self, now: dt.datetime | None = None) -> list:
        """Current state per asset, with acknowledgement applied.

        An acknowledged asset is not removed -- it is marked. A screen that
        hides what somebody accepted cannot be audited, and "why was nobody
        looking at MC-14" needs an answer.
        """
        runs = self.runs(limit=1)
        if not runs:
            return []
        newest = runs[0]["run_id"]
        out = []
        for r in self.conn.execute(
                "SELECT * FROM observation WHERE run_id=? ORDER BY score DESC",
                (newest,)):
            row = dict(r)
            st = self.ack_status(row["asset"], now=now)
            row["acknowledged"] = st["acknowledged"]
            row["ack_reason"] = (st.get("ack") or {}).get("reason")
            row["ack_by"] = (st.get("ack") or {}).get("who")
            row["rearmed_why"] = st.get("why")
            hist = self.history(row["asset"], limit=2)
            row["previous_call"] = hist[1]["call"] if len(hist) > 1 else None
            row["changed"] = (row["previous_call"] is not None
                              and row["previous_call"] != row["call"])
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _handler(store: FleetStore, page: str):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, payload, ctype="application/json"):
            body = (json.dumps(payload, default=str).encode()
                    if ctype == "application/json" else payload.encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _guard(self, fn):
            try:
                return fn()
            except KeyError as e:
                return self._send(404, {"ok": False, "error": str(e.args[0])})
            except ValueError as e:
                return self._send(400, {"ok": False, "error": str(e)})
            except Exception as e:                       # noqa: BLE001
                return self._send(500, {"ok": False, "kind": type(e).__name__,
                                        "error": str(e)})

        def do_GET(self):
            return self._guard(self._get)

        def do_POST(self):
            return self._guard(self._post)

        def _get(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path == "/api/fleet":
                return self._send(200, store.fleet_now())
            if u.path == "/api/runs":
                return self._send(200, store.runs())
            if u.path == "/api/history":
                return self._send(200, store.history(q["asset"][0]))
            if u.path == "/api/changed":
                return self._send(200, store.changed_since(q["since"][0]))
            return self._send(404, {"ok": False, "error": "no such route"})

        def _post(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/acknowledge":
                for f in ("asset", "who", "reason"):
                    if f not in body:
                        return self._send(400, {"ok": False,
                                                "error": f"missing {f}"})
                out = store.acknowledge(
                    body["asset"], body["who"], body["reason"],
                    ttl_hours=float(body.get("ttl_hours", DEFAULT_TTL_HOURS)),
                    worsen_frac=float(body.get("worsen_frac",
                                               DEFAULT_WORSEN_FRAC)))
                return self._send(200, {"ok": True, **out})
            return self._send(404, {"ok": False, "error": "no such route"})

    return H


def serve(store: FleetStore, page: str = "<h1>fleet</h1>", port: int = 0):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                          _handler(store, page))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return {"store": store, "server": srv, "thread": t,
            "port": srv.server_address[1],
            "url": f"http://127.0.0.1:{srv.server_address[1]}"}


LIMITS = [
    "No authentication. `who` is whatever the caller says it is, so the "
    "acknowledgement trail records a claim rather than an identity.",
    "Polling, not push. The page re-fetches; nothing is streamed, and an asset "
    "that degrades between polls is seen at the next one.",
    "Re-arming is evaluated on READ. A service nobody opens does not re-arm "
    "anything -- which is the safe direction (nothing is silenced by a service "
    "that is not running) and does mean an expired acknowledgement is not "
    "noticed until somebody looks.",
    "No notification. Re-arming changes what the screen shows and tells nobody.",
    "One process, one SQLite file. It is a shop-floor screen, not a fleet "
    "platform.",
]
