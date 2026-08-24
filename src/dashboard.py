"""The fleet dashboard: what to look at next, and why.

README item 5: *the T² contribution decomposition names the correct bearing race
on 100% of failing assets and the SPE contributions isolate the faulty process
tag; nothing renders either.*

WHAT A CONDITION-MONITORING SCREEN IS FOR. Not showing every asset. A plant with
four hundred bearings has four hundred rows and an operator has ten minutes, so
the only useful ordering is *which one do I look at next* — and that is not the
same as sorting by the alarm value. Two things separate them:

  A CALL WITHOUT ITS EVIDENCE DOES NOT GET ACTED ON. "BPFI on MC-14" is an
  assertion. The band ratios beside it — how far each of the four defect
  frequencies stands above the noise floor, and by how much the inner-race
  sidebands exceed the healthy baseline — are the reason, and they are what lets
  a millwright disagree. A screen that hides them trains people to ignore it.

  AN INDETERMINATE CALL IS A RESULT, NOT A GAP. The diagnoser abstains when no
  band stands clear, and abstention is the correct output when the spectrum will
  not support a call. Rendering those rows as blank, or worse omitting them,
  turns a deliberate refusal into an apparent failure of coverage.

For the process side the same rule applies to contributions: T² says the process
moved, SPE says a *relationship broke*, and the contribution decomposition names
the tag. Showing the statistic without the decomposition tells an engineer that
something is wrong and not where — which is the state they were already in.

Self-contained: inline SVG and CSS, no CDN, no JavaScript beyond sorting.
"""
from __future__ import annotations

import html
import json
import pathlib

SEV = {"faulty": "#c53030", "indeterminate": "#b7791f", "healthy": "#2f855a"}


def _esc(x) -> str:
    return html.escape(str(x))


def _bar(frac: float, colour: str, width: int = 90, height: int = 10) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    return (f'<svg width="{width}" height="{height}" role="img">'
            f'<rect width="{width}" height="{height}" fill="#e2e8f0" rx="2"/>'
            f'<rect width="{width * frac:.1f}" height="{height}" '
            f'fill="{colour}" rx="2"/></svg>')


# ---------------------------------------------------------------------------
# the bearing fleet
# ---------------------------------------------------------------------------

def fleet_from_rows(rows: list) -> list:
    """Collapse per-snapshot diagnoses into one row per asset.

    An asset is a CWRU file: one bearing under one fault at one load. The call
    is the MAJORITY over its snapshots, and the fraction agreeing is carried,
    because a bearing called BPFI on 15 of 16 snapshots and one called BPFI on 9
    of 16 are different situations and a single label hides which.
    """
    by_asset: dict = {}
    for r in rows:
        by_asset.setdefault(r["fid"], []).append(r)

    out = []
    for fid, snaps in by_asset.items():
        calls = [s.get("call") for s in snaps]
        counts: dict = {}
        for c in calls:
            counts[c] = counts.get(c, 0) + 1
        call = max(counts, key=counts.get)
        agree = counts[call] / max(len(calls), 1)
        expected = snaps[0].get("expected")
        truth = snaps[0].get("fault")
        scores = [s.get("score") or 0.0 for s in snaps]
        band = {k: max((s.get(f"r_{k}") or 0.0) for s in snaps)
                for k in ("BPFO", "BPFI", "BSF", "FTF")}
        sbp = max((s.get("sbp_BPFI") or 0.0) for s in snaps)
        correct = (call == expected) if expected else None
        state = ("healthy" if call == "healthy"
                 else "indeterminate" if call == "indeterminate" else "faulty")
        out.append({
            "asset": fid, "call": call, "agreement": agree,
            "expected": expected, "truth": truth,
            "correct": correct, "state": state,
            "size_in": snaps[0].get("size_in"),
            "load_hp": snaps[0].get("load_hp"),
            "shaft_hz": snaps[0].get("shaft_hz"),
            "score": max(scores), "bands": band, "sbp_BPFI": sbp,
            "n_snapshots": len(snaps),
        })

    # The ordering IS the product. Faulty first, then by how far the winning
    # band stands above the noise floor -- not by asset id, and not by the raw
    # score, which is not comparable between a healthy and a faulty spectrum.
    rank = {"faulty": 0, "indeterminate": 1, "healthy": 2}
    out.sort(key=lambda r: (rank[r["state"]], -r["score"]))
    return out


def _fleet_table(fleet: list) -> str:
    hi = max((r["score"] for r in fleet), default=1.0) or 1.0
    rows = []
    for r in fleet:
        colour = SEV[r["state"]]
        mark = ("" if r["correct"] is None
                else " ✓" if r["correct"] else " ✗")
        bands = " ".join(
            f'<span class="b{" hit" if k == r["call"] else ""}">{k} '
            f'{v:.1f}</span>' for k, v in r["bands"].items())
        rows.append(
            f'<tr class="{r["state"]}">'
            f'<td class="mono">{_esc(r["asset"])}</td>'
            f'<td><span class="pill" style="background:{colour}">'
            f'{_esc(r["call"])}</span>{mark}</td>'
            f'<td class="n">{r["agreement"] * 100:.0f}%</td>'
            f'<td>{_bar(r["score"] / hi, colour)} '
            f'<span class="n">{r["score"]:.1f}</span></td>'
            f'<td class="bands">{bands}</td>'
            f'<td class="n">{r["sbp_BPFI"]:.2f}</td>'
            f'<td class="mono">{_esc(r["truth"])}'
            f'{"" if not r["size_in"] else f" {r['size_in']:.3f}&Prime;"}</td>'
            f'<td class="n">{r["load_hp"]} hp</td></tr>')
    return "".join(rows)


def _confusion(scores: dict) -> str:
    conf = scores.get("confusion") or {}
    if not conf:
        return ""
    truths = list(conf)
    labels = sorted({c for row in conf.values() for c in row})
    head = "".join(f"<th>{_esc(c)}</th>" for c in labels)
    body = []
    # The diagonal is not (truth == call): the truth labels are fault NAMES and
    # the calls are defect FREQUENCIES, which is the whole translation the
    # diagnoser performs. Mapping them is what makes the table readable.
    EXPECT = {"inner_race": "BPFI", "outer_race": "BPFO", "ball": "BSF",
              "normal": "healthy"}
    for a in truths:
        cells = []
        for b in labels:
            n = (conf.get(a) or {}).get(b, 0)
            cls = ("diag" if EXPECT.get(a) == b
                   else "ind" if b == "indeterminate" else
                   "miss" if n else "zero")
            cells.append(f'<td class="n {cls}">{n or ""}</td>')
        body.append(f"<tr><th>{_esc(a)}</th>{''.join(cells)}</tr>")
    return (f'<table class="conf"><tr><th>truth \\ call</th>{head}</tr>'
            f'{"".join(body)}</table>')


# ---------------------------------------------------------------------------
# the process side
# ---------------------------------------------------------------------------

def _process_runs(runs: list) -> str:
    blocks = []
    for run in runs:
        det = run.get("detectors") or {}
        rows = []
        best = min((v.get("delay") for v in det.values()
                    if v.get("delay") is not None), default=None)
        for name, v in det.items():
            d = v.get("delay")
            label = ("never" if d is None else f"{d}")
            cls = "win" if (d is not None and d == best) else ""
            rows.append(
                f'<tr class="{cls}"><td>{_esc(name)}</td>'
                f'<td class="n">{label}</td>'
                f'<td class="n">{v.get("false_per_1000", 0):.2f}</td>'
                f'<td class="n">{(v.get("alarm_frac_after") or 0) * 100:.1f}%</td>'
                f'</tr>')
        fd = run.get("first_diagnosis") or {}
        tags = fd.get("top_tags") or []
        tot = sum(abs(t["contribution"]) for t in tags) or 1.0
        contrib = "".join(
            f'<tr><td class="mono">{_esc(t["tag"])}</td>'
            f'<td>{_bar(abs(t["contribution"]) / tot, "#2b6cb0", 140)}</td>'
            f'<td class="n">{t["contribution"]:.2f}</td></tr>' for t in tags)
        blocks.append(f"""
<section class="run">
  <h3>{_esc(run.get('fault', '?'))}</h3>
  <p class="sub">{_esc(run.get('description', ''))}
     &middot; fault injected at sample {run.get('fault_start')}
     &middot; {run.get('n_components')} components,
     {(run.get('variance_explained') or 0) * 100:.0f}% of variance</p>
  <div class="two">
    <div>
      <h4>Detection delay, in samples</h4>
      <table><tr><th>detector</th><th>delay</th><th>false/1000</th>
      <th>alarm rate after</th></tr>{''.join(rows)}</table>
    </div>
    <div>
      <h4>What broke, at first detection</h4>
      {"<p class='sub'>no detection</p>" if not fd else
       f'<p class="sub">{_esc(fd.get("statistic"))} fired first at sample '
       f'{fd.get("sample")} &middot; T&sup2; {fd.get("t2", 0):.1f}, '
       f'SPE {fd.get("spe", 0):.1f}</p>'
       f'<table><tr><th>tag</th><th></th><th>contribution</th></tr>{contrib}</table>'}
    </div>
  </div>
</section>""")
    return "".join(blocks)


# ---------------------------------------------------------------------------

CSS = """
:root{--ink:#1a202c;--mut:#4a5568;--line:#e2e8f0;--bg:#fff;--panel:#f7fafc}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     color:var(--ink);background:var(--bg)}
header{padding:22px 26px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 4px;font-size:20px}
h2{font-size:16px;margin:26px 0 8px}
h3{font-size:15px;margin:0 0 2px}
h4{font-size:13px;margin:12px 0 6px;color:var(--mut);text-transform:uppercase;
   letter-spacing:.04em}
main{padding:0 26px 40px;max-width:1180px}
.sub{color:var(--mut);margin:2px 0 10px;font-size:13px}
table{border-collapse:collapse;width:100%;margin:6px 0 14px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;
      vertical-align:middle}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.pill{color:#fff;padding:1px 7px;border-radius:9px;font-size:11px;
      font-weight:600;display:inline-block;min-width:64px;text-align:center}
tr.healthy td{opacity:.62}
.bands{font-size:11px;color:var(--mut)}
.b{display:inline-block;padding:0 4px;border-radius:3px;background:#edf2f7;
   margin-right:2px;font-variant-numeric:tabular-nums}
.b.hit{background:#feb2b2;color:#742a2a;font-weight:600}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 4px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px;
      background:var(--panel)}
.card .v{font-size:22px;font-weight:600}
.card .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.run{border-top:1px solid var(--line);padding-top:16px;margin-top:18px}
tr.win td{background:#f0fff4;font-weight:600}
table.conf td.diag{background:#f0fff4;font-weight:600}
table.conf td.miss{background:#fff5f5}
table.conf td.zero{color:#cbd5e0}
table.conf td.ind{background:#fffaf0;color:#975a16}
.note{background:var(--panel);border-left:3px solid #a0aec0;padding:9px 13px;
      margin:12px 0;font-size:13px;color:var(--mut)}
footer{padding:16px 26px;border-top:1px solid var(--line);color:var(--mut);font-size:12px}
"""


def render(path, cwru: dict, process: dict) -> dict:
    fleet = fleet_from_rows(cwru.get("rows") or [])
    sc = cwru.get("scores") or {}
    n_fault = sum(1 for r in fleet if r["state"] == "faulty")
    n_ind = sum(1 for r in fleet if r["state"] == "indeterminate")
    graded = [r for r in fleet if r["correct"] is not None]
    n_right = sum(1 for r in graded if r["correct"])

    cards = [
        ("assets", len(fleet), ""),
        ("calling a fault", n_fault, ""),
        ("abstaining", n_ind, "no band stands clear"),
        ("correct race", f"{n_right}/{len(graded)}", "where a race is expected"),
        ("false calls on healthy",
         f"{sc.get('false_diagnosis_on_healthy') or 0}/{sc.get('n_healthy') or 0}",
         "healthy snapshots given a fault"),
        ("accuracy on faulty",
         f"{(sc.get('accuracy_on_faulty') or 0) * 100:.0f}%",
         "over snapshots, not assets"),
    ]
    n_green = sum(1 for r in fleet if r["state"] == "healthy")
    n_truly_healthy = sum(1 for r in fleet if r["truth"] == "normal")
    green_note = ""
    if n_green == 0 and n_truly_healthy:
        # The fleet view says something the summary table could not. RESULTS.md
        # reports 21.9% of healthy SNAPSHOTS called healthy, which reads as a
        # weak number. Aggregated to assets by majority vote it becomes zero:
        # not one bearing on this screen is green. A monitoring screen on which
        # nothing is ever green is a screen that gets ignored within a week, and
        # that is a bigger problem than the accuracy figure suggests.
        green_note = (
            '<div class="note" style="border-left-color:#c53030">'
            f'<b>Nothing on this screen is green.</b> {n_truly_healthy} of these '
            'assets are healthy bearings and <b>none of them is called '
            'healthy</b> — they come out as faults or abstentions. The summary '
            'in RESULTS.md reports 21.9% of healthy <i>snapshots</i> called '
            'healthy, which reads as a weak number; aggregated to assets by '
            'majority vote it is zero. A screen on which nothing is ever green '
            'is a screen that gets ignored within a week, and rendering the '
            'fleet is what made that visible.</div>')

    card_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div>'
        f'<div class="k" style="text-transform:none">{_esc(note)}</div></div>'
        for k, v, note in cards)

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>Condition monitoring — fleet</title>
<style>{CSS}</style>
<header>
  <h1>Condition monitoring — fleet</h1>
  <p class="sub">{len(fleet)} bearing assets from CWRU, and
     {len(process.get('runs') or [])} injected process faults.
     Ordered by what to look at next.</p>
</header>
<main>
  <div class="cards">{card_html}</div>

  <div class="note"><b>Abstention is a result.</b> {n_ind} assets are called
   <i>indeterminate</i>: no defect band stands clear of the noise floor, and the
   diagnoser refuses rather than picking the largest of four similar numbers.
   Rendering those as blank would turn a deliberate refusal into an apparent gap
   in coverage.</div>
  {green_note}

  <h2>Bearings</h2>
  <p class="sub">Band ratios are each defect frequency's energy against the local
   noise floor. The highlighted one is the call. <b>sbp</b> is inner-race
   sideband prominence — the tie-breaker, and the only feature that separates an
   inner-race fault from an outer-race one once both bands are lifted.</p>
  <table>
    <tr><th>asset</th><th>call</th><th>agreement</th><th>evidence</th>
        <th>band ratios (vs noise floor)</th><th>sbp</th>
        <th>ground truth</th><th>load</th></tr>
    {_fleet_table(fleet)}
  </table>

  <h2>Where the calls go wrong</h2>
  {_confusion(sc)}

  <h2>Process</h2>
  <p class="sub">T&sup2; says the process moved inside its normal correlation
   structure. SPE says a <i>relationship broke</i>. The contribution
   decomposition names the tag — showing the statistic without it tells an
   engineer that something is wrong and not where, which is the state they were
   already in.</p>
  {_process_runs(process.get('runs') or [])}
</main>
<footer>Self-contained; no network requests. Bearing data: CWRU Bearing Data
 Center. Process data: a simulator in <code>src/process.py</code> — every process
 number here is a statement about that generator.</footer>
"""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return {"path": str(p), "bytes": p.stat().st_size, "assets": len(fleet),
            "faulty": n_fault, "indeterminate": n_ind,
            "graded": len(graded), "correct": n_right,
            "healthy_assets_called_healthy": n_green,
            "truly_healthy_assets": n_truly_healthy,
            "process_runs": len(process.get("runs") or []),
            "self_contained": "http" not in doc.split("footer")[0].lower()
            or "://" not in doc}


def render_from_out(out_dir="out", name="fleet_dashboard.html") -> dict:
    out = pathlib.Path(out_dir)
    cwru = json.loads((out / "cwru.json").read_text(encoding="utf-8"))
    proc = json.loads((out / "process.json").read_text(encoding="utf-8"))
    return render(out / name, cwru, proc)
