"""ML-3, the next 30%: alarm explanations, cold start, P-F interval, more assets.

    python extend.py
    python extend.py --quick
    python extend.py --report-only

Four gaps the first build named:
  1. the per-alarm "why" panel -- promised as the reason to ship T-squared, then
     not built
  2. cold-start validation -- LOW_CONFIDENCE existed but was never exercised
  3. P-F interval quantification -- the vocabulary was used, the number was not
  4. more assets, so the medians stop being medians over nine numbers
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import bearing  # noqa: E402
import detectors  # noqa: E402
import explain as X  # noqa: E402
import features  # noqa: E402
import health  # noqa: E402

OUT = ROOT / "out"
GEOM = bearing.BearingGeometry()
BASELINE_CYCLES = 60
BAND = (3000.0, 4000.0)


def build_fleet(quick: bool) -> list[dict]:
    n_cycles = 160 if quick else 240
    onset = 80 if quick else 120
    seeds = (1, 2) if quick else (1, 2, 3, 4, 5, 6)
    fleet = []
    for fault in ("BPFO", "BPFI", "BSF"):
        for s in seeds:
            snaps, sev, _, truth = bearing.simulate_run_to_failure(
                GEOM, fault, n_cycles=n_cycles, onset=onset,
                seed=s * 17 + zlib.crc32(fault.encode()) % 100)
            fleet.append({"asset": f"{fault}-{s}", "fault": fault, "snaps": snaps,
                          "severity": sev, "truth": truth, "failing": True})
    for s in seeds:
        snaps, _ = bearing.simulate_healthy(GEOM, n_cycles=n_cycles, seed=900 + s)
        fleet.append({"asset": f"healthy-{s}", "fault": None, "snaps": snaps,
                      "severity": np.zeros(n_cycles),
                      "truth": {"fault": None, "onset_cycle": None,
                                "failure_cycle": None}, "failing": False})
    return fleet


def featurise(fleet):
    for a in fleet:
        a["feats"] = [features.snapshot_features(s, GEOM, band=BAND) for s in a["snaps"]]
        a["baseline"] = health.Baseline.fit(a["feats"][:BASELINE_CYCLES])
        hi = np.array([health.health_index(f, a["baseline"]) for f in a["feats"]])
        a["hi"] = health.smooth(hi, 5)


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "extensions.json").read_text())
        (ROOT / "docs" / "EXTENSIONS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/EXTENSIONS.md")
        return

    t0 = time.perf_counter()
    print("1/4 building a larger fleet ...", flush=True)
    fleet = build_fleet(quick)
    featurise(fleet)
    failing = [a for a in fleet if a["failing"]]
    healthy = [a for a in fleet if not a["failing"]]
    res: dict = {"fleet": {"failing": len(failing), "healthy": len(healthy),
                           "cycles": len(fleet[0]["snaps"])}}
    print(f"    {len(failing)} failing + {len(healthy)} healthy assets", flush=True)

    print("2/4 per-alarm explanations ...", flush=True)
    res["explanations"] = explanation_stage(fleet)

    print("3/4 cold-start policy ...", flush=True)
    res["cold_start"] = cold_start_stage(fleet)

    print("4/4 P-F interval and the inspection interval it implies ...", flush=True)
    res["pf"] = pf_stage(failing)
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "extensions.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "EXTENSIONS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/EXTENSIONS.md ({res['wall_seconds']:.0f}s)")


def explanation_stage(fleet) -> dict:
    """Decompose the T-squared score at each asset's alarm point."""
    keys = list(detectors.FEATURE_KEYS)
    out = []
    correct = 0
    total = 0
    for a in fleet:
        x = np.array([[f[k] for k in keys] for f in a["feats"]], dtype=float)
        train = x[:BASELINE_CYCLES]
        mu = train.mean(axis=0)
        cov = np.cov(train, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0]) * 0.1
        inv = np.linalg.pinv(cov)

        idx = len(x) - 1 if a["failing"] else len(x) // 2
        ex = X.explain_alarm(x[idx], mu, inv, keys)
        row = {"asset": a["asset"], "true_fault": a["fault"] or "healthy",
               "cycle": idx, "t2": ex["t2"],
               "top": ex["top_contributors"][:3],
               "top_n_pct": ex["top_n_pct_of_score"],
               "sentence": ex["sentence"]}
        if a["failing"]:
            total += 1
            lead = ex["top_contributors"][0]["feature"]
            row["explanation_names_true_fault"] = a["fault"] in lead
            correct += int(a["fault"] in lead)
        out.append(row)
    return {
        "per_asset": out,
        "top_contributor_names_true_fault_pct":
            100.0 * correct / max(1, total),
        "n_failing_checked": total,
    }


def cold_start_stage(fleet) -> dict:
    """Exercise the policy that was implemented and never tested."""
    baselines = [a["baseline"] for a in fleet if not a["failing"]]
    prior = X.fleet_prior(baselines)

    rows = []
    victim = [a for a in fleet if a["failing"]][0]
    for n in (5, 15, 30, 60, 90, 180):
        pol = X.cold_start_policy(n)
        # What would the health index say with only n cycles of baseline?
        if n >= 5:
            b = health.Baseline.fit(victim["feats"][:n])
            hi_last = health.health_index(victim["feats"][-1], b)
        else:
            hi_last = float("nan")
        rows.append({**pol, "health_at_failure_with_this_baseline": hi_last})
    return {"fleet_prior_n_assets": prior["n_assets"], "policy_by_history": rows}


def pf_stage(failing) -> dict:
    rows = []
    for a in failing:
        r = X.pf_interval(a["hi"], failure_index=len(a["hi"]) - 1)
        r["asset"] = a["asset"]
        r["fault"] = a["fault"]
        rows.append(r)
    intervals = [r["pf_interval_cycles"] for r in rows if r["detected"]]
    return {"per_asset": rows, "summary": X.inspection_interval(intervals)}


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    f = res["fleet"]
    A("# ML-3 extensions — generated by `extend.py`, not hand-edited\n")
    A(f"Fleet: **{f['failing']} failing + {f['healthy']} healthy** assets "
      f"({f['cycles']} cycles each) — up from 9+3, so the medians below rest on "
      "more than nine numbers.\n")

    e = res["explanations"]
    A("## 1. The per-alarm \"why\" panel\n")
    A("The first build recommended shipping Hotelling T² **partly because its "
      "score decomposes into per-feature contributions** — and then did not "
      "decompose it. This closes that gap, and it matters more than the detector "
      "choice: an alarm reading \"health 62, investigate\" gets acknowledged and "
      "ignored, because a scalar is not actionable.\n")
    A("The decomposition is exact rather than an attribution heuristic. With "
      "`d = x − μ`, feature *j* contributes `d_j · (S⁻¹d)_j`, and those "
      "contributions **sum to T² identically** — unlike SHAP or occlusion, which "
      "approximate. That distinction is worth having in a domain where \"the model "
      "said so\" is not an acceptable answer to a maintenance planner.\n")
    A("| asset | true fault | T² | leading contributor | % of score | top-3 % | ")
    A("|---|---|---|---|---|---|")
    for r in e["per_asset"][:12]:
        t = r["top"][0]
        A(f"| {r['asset']} | {r['true_fault']} | {r['t2']:.0f} | {t['feature']} | "
          f"{t['pct_of_total']:.0f}% | {r['top_n_pct']:.0f}% |")
    A(f"\n**The leading contributor names the true fault frequency on "
      f"{e['top_contributor_names_true_fault_pct']:.0f}% of the "
      f"{e['n_failing_checked']} failing assets.** That is the property that makes "
      "the panel worth putting on screen: the top line of the explanation points at "
      "the right bearing race, so the alarm card carries a diagnosis and not just a "
      "score.\n")
    sample = next((r for r in e["per_asset"] if r["true_fault"] != "healthy"), None)
    if sample:
        A("A sample alarm card:\n")
        A("```")
        A(f"ASSET  {sample['asset']}      HEALTH  alarm at cycle {sample['cycle']}")
        A(f"WHY    {sample['sentence']}")
        A("```")
    A("\nContributions can be **negative** when features are correlated — a feature "
      "moving *with* its correlated partners reduces the distance. That is real "
      "information (the deviation is in the expected direction) and it is reported "
      "rather than clipped to zero.")

    c = res["cold_start"]
    A("\n## 2. Cold start — implemented before, never exercised\n")
    A(f"Fleet prior pooled from {c['fleet_prior_n_assets']} healthy assets.\n")
    A("| baseline cycles | state | confidence | fleet prior? | threshold widening | health index at failure |")
    A("|---|---|---|---|---|---|")
    for r in c["policy_by_history"]:
        hv = (f"{r['health_at_failure_with_this_baseline']:.1f}"
              if np.isfinite(r["health_at_failure_with_this_baseline"]) else "—")
        A(f"| {r['n_cycles']} | **{r['state']}** | {r['confidence']} | "
          f"{'yes' if r['uses_fleet_prior'] else 'no'} | "
          f"×{r['threshold_widening']:.3f} | {hv} |")
    A("\n**The middle state is the one most systems omit.** `NO_BASELINE` is not "
      "silence — silence is indistinguishable from healthy, and a new machine is "
      "exactly when infant-mortality failures happen. `PROVISIONAL` widens "
      "thresholds by **1 + 1/√(2(n−1))**, the approximate relative standard error "
      "of a standard-deviation estimate from n samples. That is not a tuning knob; "
      "it is the sampling error of the quantity being estimated, so the detector "
      "automatically becomes less trigger-happy exactly when its baseline is least "
      "trustworthy.\n")
    A("The last column is the practical consequence: the same failing asset scored "
      "against baselines of different lengths. A short baseline is not merely "
      "noisier — it can be *contaminated*, because an asset with an incipient fault "
      "at install bakes that fault into its own definition of normal.")

    p = res["pf"]
    s = p["summary"]
    A("\n## 3. The P-F interval, and the number a planner actually asks for\n")
    A("**P** = potential failure, the first point the condition is detectable. "
      "**F** = functional failure. The P-F interval is what sets the inspection "
      "interval: to catch a fault you must inspect at least twice within it.\n")
    A("| statistic | cycles |")
    A("|---|---|")
    A(f"| assets with a detected P point | {s['n']} |")
    A(f"| mean P-F interval | {s['pf_mean']:.0f} |")
    A(f"| median | {s['pf_median']:.0f} |")
    A(f"| **P10** | **{s['pf_p10']:.0f}** |")
    A(f"| minimum observed | {s['pf_min']:.0f} |")
    A(f"| **recommended inspection interval** (P10 ÷ 2) | **{s['recommended_interval_cycles']:.0f}** |")
    A(f"| interval if the MEAN were used instead | {s['interval_if_mean_used']:.0f} |")
    ratio = s["interval_if_mean_used"] / max(s["recommended_interval_cycles"], 1e-9)
    A(f"\n**Setting the interval from the mean would make it {ratio:.1f}× too long.** "
      "Half the *mean* P-F interval misses the fast half of the failures by "
      "construction — the distribution is the deliverable, not its centre. This is "
      "the same argument as quoting lead time at a false-alarm budget rather than "
      "on its own, and it is the form a planner can act on: *inspect every "
      f"{s['recommended_interval_cycles']:.0f} cycles* is a schedule; *the mean P-F "
      "interval is " + f"{s['pf_mean']:.0f}* is trivia.")

    A("\n---\n*Regenerate with `python extend.py`.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
