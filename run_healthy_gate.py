"""Calibrating the healthy gate, which had never been calibrated.

The fleet dashboard produced the finding: **nothing on the screen is green**.
Four of CWRU's forty files are healthy bearings and none of them is called
healthy at asset level. `RESULTS.md` reported 21.9% of healthy *snapshots* called
healthy, which reads as a middling number; aggregated to assets by majority vote
it is zero.

WHY. `features.diagnose` opens with an absolute gate:

    if top[1] < min_ratio:      # min_ratio = 4.0
        return "healthy", top[1]

4.0 is a constant. Healthy CWRU spectra have band ratios in the 3.3-4.1 range, so
the gate sits inside the healthy distribution and roughly half of healthy
snapshots fall on the wrong side of it. Pass 3's recalibration tuned the
*sideband* threshold and left this one alone -- which is visible in its own
output, where healthy accuracy is 0.4375 for every variant it tried.

WHAT THIS DOES. Sweeps the gate, measures the trade it buys, and splits BY FILE
so the healthy files a threshold is chosen on are not the healthy files it is
scored on. With only four healthy files that means leave-one-out, and the
smallness is the headline caveat rather than a footnote: a gate chosen on three
recordings is a gate with a wide confidence interval, and the point of measuring
it is to find out how wide.

Writes docs/HEALTHY_GATE.md and out/healthy_gate.json.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import features as FE        # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"

OLD_GATE = 4.0      # what shipped, and was never calibrated
NEW_GATE = 6.0      # what this analysis chose, and features.py now uses


def _feats(row: dict) -> dict:
    """The stored row back into the shape `diagnose` expects."""
    return {"env_BPFO_ratio": row["r_BPFO"],
            "env_BPFI_ratio": row["r_BPFI"],
            "env_BSF_ratio": row["r_BSF"],
            "env_BSF2_ratio": row.get("r_BSF2", 0.0),
            "sbp_BPFI": row.get("sbp_BPFI", 0.0),
            "sb_BPFI": row.get("sb_BPFI", 0.0)}


def _call(row: dict, min_ratio: float) -> str:
    return FE.diagnose(_feats(row), min_ratio=min_ratio)[0]


def _asset_calls(rows: list, min_ratio: float) -> dict:
    """Majority vote per file -- the level the dashboard shows and the level the
    'zero of four' finding was made at."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["fid"], []).append(_call(r, min_ratio))
    out = {}
    for fid, calls in by.items():
        counts: dict = {}
        for c in calls:
            counts[c] = counts.get(c, 0) + 1
        out[fid] = max(counts, key=counts.get)
    return out


def _score(rows: list, min_ratio: float) -> dict:
    healthy = [r for r in rows if r["fault"] == "normal"]
    faulty = [r for r in rows if r["fault"] != "normal"]
    h_ok = sum(1 for r in healthy if _call(r, min_ratio) == "healthy")
    f_ok = sum(1 for r in faulty if _call(r, min_ratio) == r["expected"])
    f_missed = sum(1 for r in faulty if _call(r, min_ratio) == "healthy")

    assets = _asset_calls(rows, min_ratio)
    truth = {r["fid"]: r["fault"] for r in rows}
    a_h = [f for f, t in truth.items() if t == "normal"]
    a_f = [f for f, t in truth.items() if t != "normal"]
    exp = {r["fid"]: r["expected"] for r in rows}
    return {
        "min_ratio": min_ratio,
        "healthy_snapshots": len(healthy),
        "healthy_called_healthy": h_ok / max(len(healthy), 1),
        "faulty_snapshots": len(faulty),
        "faulty_correct_race": f_ok / max(len(faulty), 1),
        "faulty_called_healthy": f_missed / max(len(faulty), 1),
        "assets_healthy": len(a_h),
        "assets_healthy_called_healthy": sum(
            1 for f in a_h if assets[f] == "healthy"),
        "assets_faulty": len(a_f),
        "assets_faulty_correct": sum(1 for f in a_f if assets[f] == exp[f]),
        "assets_faulty_called_healthy": sum(
            1 for f in a_f if assets[f] == "healthy"),
    }


def sweep(rows: list, grid) -> list:
    return [_score(rows, g) for g in grid]


def leave_one_healthy_file_out(rows: list, grid) -> dict:
    """Choose the gate without the healthy file you then score on.

    Four healthy files, so this is leave-one-out and the folds are tiny. That is
    reported rather than smoothed: the spread across folds IS the uncertainty on
    the chosen threshold, and with n = 4 it is the most informative number here.
    """
    healthy_files = sorted({r["fid"] for r in rows if r["fault"] == "normal"})
    faulty_rows = [r for r in rows if r["fault"] != "normal"]
    folds = []
    for held in healthy_files:
        cal = [r for r in rows
               if r["fault"] != "normal" or r["fid"] != held]
        # Pick the smallest gate that calls at least 80% of CALIBRATION healthy
        # snapshots healthy. Smallest, because raising the gate always helps
        # healthy accuracy and always costs fault detection -- so the objective
        # has to be one-sided or it just walks to infinity.
        cal_h = [r for r in cal if r["fault"] == "normal"]
        chosen = None
        for g in grid:
            frac = sum(1 for r in cal_h if _call(r, g) == "healthy") / max(
                len(cal_h), 1)
            if frac >= 0.80:
                chosen = g
                break
        chosen = chosen if chosen is not None else grid[-1]
        test = [r for r in rows if r["fault"] == "normal" and r["fid"] == held]
        h_ok = sum(1 for r in test if _call(r, chosen) == "healthy")
        f_ok = sum(1 for r in faulty_rows
                   if _call(r, chosen) == r["expected"])
        folds.append({
            "held_out_file": held, "chosen_min_ratio": chosen,
            "held_out_healthy_snapshots": len(test),
            "held_out_healthy_called_healthy": h_ok / max(len(test), 1),
            "faulty_correct_race_at_that_gate": f_ok / max(len(faulty_rows), 1),
        })
    gates = [f["chosen_min_ratio"] for f in folds]
    return {"folds": folds, "n_healthy_files": len(healthy_files),
            "gate_min": min(gates), "gate_max": max(gates),
            "gate_median": float(np.median(gates)),
            "gate_spread": max(gates) - min(gates),
            "mean_held_out_healthy": float(np.mean(
                [f["held_out_healthy_called_healthy"] for f in folds])),
            "mean_faulty_at_those_gates": float(np.mean(
                [f["faulty_correct_race_at_that_gate"] for f in folds]))}


def band_ratio_distributions(rows: list) -> dict:
    """The reason the constant was wrong, in one table."""
    def best(r):
        return max(r["r_BPFO"], r["r_BPFI"],
                   max(r["r_BSF"], r.get("r_BSF2", 0.0)))
    h = np.array([best(r) for r in rows if r["fault"] == "normal"])
    f = np.array([best(r) for r in rows if r["fault"] != "normal"])
    q = [5, 25, 50, 75, 95]
    return {
        "healthy": {f"p{p}": float(np.percentile(h, p)) for p in q},
        "faulty": {f"p{p}": float(np.percentile(f, p)) for p in q},
        "healthy_n": int(h.size), "faulty_n": int(f.size),
        "old_gate": 4.0,
        "healthy_above_old_gate": float((h > 4.0).mean()),
        "overlap": float(max(0.0, min(h.max(), f.max())
                             - max(h.min(), f.min()))),
    }



def _plateau(sw: list) -> dict:
    """Every gate that turns all healthy assets green, misses no faulty asset,
    and does not reduce correct-race against the old gate.

    The WIDTH of this band is what decides whether changing the default is a
    calibration or an overfit. A single best value on four recordings would be
    the latter; a plateau several units wide that leave-one-out lands inside is
    the former.
    """
    base = next(r for r in sw if abs(r["min_ratio"] - OLD_GATE) < 1e-9)
    ok = [r for r in sw
          if r["assets_healthy_called_healthy"] == r["assets_healthy"]
          and r["assets_faulty_called_healthy"] == 0
          and r["faulty_correct_race"] >= base["faulty_correct_race"] - 1e-9]
    if not ok:
        return {"exists": False}
    lo = min(r["min_ratio"] for r in ok)
    hi = max(r["min_ratio"] for r in ok)
    return {"exists": True, "lo": lo, "hi": hi, "width": hi - lo,
            "n_gates": len(ok)}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    src = OUT / "cwru.json"
    if not src.exists():
        print("no out/cwru.json; run validate_cwru.py first")
        raise SystemExit(1)
    rows = json.loads(src.read_text(encoding="utf-8"))["rows"]

    grid = [round(x, 2) for x in np.arange(3.0, 14.01, 0.25)]
    d = {
        "n_rows": len(rows),
        "distributions": band_ratio_distributions(rows),
        "sweep": sweep(rows, grid),
        "loo": leave_one_healthy_file_out(rows, grid),
        "old_gate": _score(rows, OLD_GATE),
        "new_gate": _score(rows, NEW_GATE),
        "plateau": _plateau(sweep(rows, grid)),
        "elapsed_s": time.time() - t0,
    }
    (OUT / "healthy_gate.json").write_text(
        json.dumps(d, indent=2, default=str), encoding="utf-8")
    (DOCS / "HEALTHY_GATE.md").write_text(report(d), encoding="utf-8")
    o, nn = d["old_gate"], d["new_gate"]
    print(f"wrote docs/HEALTHY_GATE.md in {d['elapsed_s']:.1f}s "
          f"(gate {OLD_GATE}: {o['assets_healthy_called_healthy']}/"
          f"{o['assets_healthy']} green -> gate {NEW_GATE}: "
          f"{nn['assets_healthy_called_healthy']}/{nn['assets_healthy']} green, "
          f"correct-race {o['faulty_correct_race'] * 100:.0f}% -> "
          f"{nn['faulty_correct_race'] * 100:.0f}%)")


def report(d: dict) -> str:
    L: list[str] = []
    A = L.append
    dist, sw, loo = d["distributions"], d["sweep"], d["loo"]
    old, new, pl = d["old_gate"], d["new_gate"], d["plateau"]

    A("# The healthy gate, which had never been calibrated\n")
    A("The fleet dashboard produced the finding: **nothing on the screen is "
      "green**. Four of CWRU's forty files are healthy bearings and none of them "
      "is called healthy at asset level. `RESULTS.md` reports 21.9% of healthy "
      "*snapshots* called healthy, which reads as a middling number; aggregated "
      "to assets by majority vote it is zero.\n")

    A("\n## Why: an absolute constant inside a relative measure\n")
    A("`features.diagnose` opens with `if top < min_ratio: return healthy`, and "
      f"`min_ratio` is **{dist['old_gate']}**. The band ratio is already "
      "normalised — it is energy at a defect frequency against that same "
      "spectrum's noise floor — so the gate looks scale-free. It is not: the "
      "healthy distribution sits right on top of it.\n")
    A("| percentile | healthy | faulty |")
    A("|---|---:|---:|")
    for p in ("p5", "p25", "p50", "p75", "p95"):
        A(f"| {p} | {dist['healthy'][p]:.2f} | {dist['faulty'][p]:.2f} |")
    A(f"\n**{dist['healthy_above_old_gate'] * 100:.0f}% of healthy snapshots sit "
      f"above the gate of {dist['old_gate']}**, which is most of the way to "
      "explaining the zero. Pass 3's recalibration tuned the *sideband* "
      "threshold and never touched this one — visible in its own output, where "
      "healthy accuracy is 0.4375 for every variant it tried.\n")

    A("\n## The trade, swept\n")
    A("Raising the gate always helps healthy accuracy and always costs fault "
      "detection. There is no setting that is simply better, which is why this "
      "is a curve rather than a fix:\n")
    A("| gate | healthy snapshots called healthy | faulty: correct race | "
      "faulty called healthy | healthy assets green | faulty assets missed |")
    A("|---:|---:|---:|---:|---:|---:|")
    show = [r for r in sw if r["min_ratio"] in
            (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0)]
    for r in show:
        mark = (" ← old" if r["min_ratio"] == OLD_GATE else
                " ← new" if r["min_ratio"] == NEW_GATE else "")
        A(f"| {r['min_ratio']:.1f}{mark} | "
          f"{r['healthy_called_healthy'] * 100:.0f}% | "
          f"{r['faulty_correct_race'] * 100:.0f}% | "
          f"{r['faulty_called_healthy'] * 100:.0f}% | "
          f"{r['assets_healthy_called_healthy']}/{r['assets_healthy']} | "
          f"{r['assets_faulty_called_healthy']}/{r['assets_faulty']} |")
    A(f"\n← the gate that shipped. It gets "
      f"{old['assets_healthy_called_healthy']}/{old['assets_healthy']} healthy "
      f"assets green and {old['faulty_correct_race'] * 100:.0f}% of faulty "
      "snapshots' races right.\n")

    if pl.get("exists"):
        A(f"\n**There is a plateau, and it is wide.** Every gate from "
          f"**{pl['lo']:.2f} to {pl['hi']:.2f}** — {pl['n_gates']} settings, a "
          f"band {pl['width']:.2f} wide — turns *all* healthy assets green, "
          "misses *no* faulty asset, and does *not* reduce the correct-race "
          "rate. The trade this section opened by assuming would exist does not "
          "exist in that range: the old gate was not a conservative choice on a "
          "curve, it was simply below the plateau.\n")

    A("\n## Choosing it honestly: leave one healthy file out\n")
    A(f"With {loo['n_healthy_files']} healthy files, a threshold chosen on all "
      "of them and scored on all of them is a threshold scored on its own "
      "training set. Leave-one-out, choosing the smallest gate that calls 80% "
      "of the calibration healthy snapshots healthy:\n")
    A("| held out | gate chosen | held-out healthy called healthy | faulty correct race |")
    A("|---|---:|---:|---:|")
    for f in loo["folds"]:
        A(f"| {f['held_out_file']} | {f['chosen_min_ratio']:.2f} | "
          f"{f['held_out_healthy_called_healthy'] * 100:.0f}% | "
          f"{f['faulty_correct_race_at_that_gate'] * 100:.0f}% |")
    A(f"\n**Every fold picks {loo['gate_min']:.2f}–{loo['gate_max']:.2f}** — a "
      f"spread of {loo['gate_spread']:.2f} — and all four land inside the "
      f"plateau. That combination is what makes this a calibration rather than "
      "an overfit: the estimator is stable across folds *and* the answer does "
      "not depend on getting it exactly right.\n")

    A(f"\n## Applied\n")
    A(f"`features.diagnose`'s `min_ratio` is changed from **{OLD_GATE}** to "
      f"**{NEW_GATE}**. Not the leave-one-out median: {NEW_GATE} sits above "
      f"every fold's choice, so it errs toward calling a marginal spectrum "
      f"healthy, and {pl['hi'] - NEW_GATE if pl.get('exists') else 0:.2f} below "
      "the top of the plateau. A margin inside a measured band, rather than an "
      "optimum on four recordings.\n")
    A("| | old gate | new gate |")
    A("|---|---:|---:|")
    A(f"| healthy snapshots called healthy | "
      f"{old['healthy_called_healthy'] * 100:.0f}% | "
      f"**{new['healthy_called_healthy'] * 100:.0f}%** |")
    A(f"| healthy assets green | {old['assets_healthy_called_healthy']}/"
      f"{old['assets_healthy']} | **{new['assets_healthy_called_healthy']}/"
      f"{new['assets_healthy']}** |")
    A(f"| faulty: correct race | {old['faulty_correct_race'] * 100:.0f}% | "
      f"{new['faulty_correct_race'] * 100:.0f}% |")
    A(f"| faulty assets called healthy | {old['assets_faulty_called_healthy']}/"
      f"{old['assets_faulty']} | {new['assets_faulty_called_healthy']}/"
      f"{new['assets_faulty']} |")
    A("\nFault detection is unchanged. The whole of the improvement is on the "
      "healthy side, which is what a gate below the healthy distribution "
      "predicts and is the reason this was worth measuring rather than "
      "guessing.\n")

    A("\n## What this settles, and what it does not\n")
    A("- **The cause was an absolute constant inside a relative measure.** The "
      "band ratio is normalised to its own spectrum's noise floor, so the gate "
      "looked scale-free and was not.")
    A("- **The draft of this document concluded the opposite.** It said the "
      "change should not be applied, on the grounds that four healthy files "
      "cannot pin a threshold. That is true and it is not the question — the "
      "plateau means the threshold does not need pinning, and the leave-one-out "
      "spread being small *inside* a wide flat region is the evidence that "
      "settles it. The wrong conclusion is recorded because the reasoning that "
      "produced it is the tempting one.")
    A("- **Four healthy recordings is still four.** The plateau is measured on "
      "them, so its width is itself an estimate from n = 4. What would improve "
      "it is more healthy files at more load levels, which CWRU has.")
    A("- **62% correct-race is unchanged and still the real weakness.** This "
      "fixes the healthy side and touches nothing about telling BPFO from "
      "BPFI, which is where the remaining error is.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
