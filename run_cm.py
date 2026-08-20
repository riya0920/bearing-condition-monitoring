"""ML-3 end-to-end: physics features, three detector families, lead-time economics.

    python run_cm.py            # full fleet (~10-15 min)
    python run_cm.py --quick    # smaller fleet

Writes docs/RESULTS.md and out/results.json. Ground truth (fault type, onset cycle,
failure cycle) comes from the simulator, which is what makes "detected 61 cycles
early" a scoreable claim rather than a screenshot of red dots.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import bearing  # noqa: E402
import detectors  # noqa: E402
import features  # noqa: E402
import health  # noqa: E402

OUT = ROOT / "out"
GEOM = bearing.BearingGeometry()
BASELINE_CYCLES = 60


def build_fleet(quick: bool) -> list[dict]:
    """A fleet with ground truth: failing assets of each fault type, plus healthy
    assets that never fail.

    The healthy assets are not decoration. False-alarm rate measured only on the
    healthy PREFIX of failing assets is measured on assets that are about to fail,
    which is not the fleet a plant runs. Without never-failing assets the
    false-alarm number is optimistic and no one can tell.
    """
    n_cycles = 160 if quick else 260
    onset = 80 if quick else 130
    seeds = (1, 2) if quick else (1, 2, 3)
    fleet = []
    for fault in ("BPFO", "BPFI", "BSF"):
        for s in seeds:
            snaps, sev, speeds, truth = bearing.simulate_run_to_failure(
                GEOM, fault, n_cycles=n_cycles, onset=onset, seed=s * 17 + hash(fault) % 100
            )
            fleet.append({"asset": f"{fault}-{s}", "fault": fault, "snaps": snaps,
                          "severity": sev, "truth": truth, "failing": True})
    for s in seeds:
        snaps, speeds = bearing.simulate_healthy(GEOM, n_cycles=n_cycles, seed=900 + s)
        fleet.append({"asset": f"healthy-{s}", "fault": None, "snaps": snaps,
                      "severity": np.zeros(n_cycles),
                      "truth": {"fault": None, "onset_cycle": None,
                                "failure_cycle": None},
                      "failing": False})
    return fleet


# The demodulation band, as a COMMISSIONING PARAMETER rather than something learned.
#
# This started as an adaptive kurtogram over the baseline period and that was wrong,
# in an instructive way: the baseline period is healthy, a healthy bearing produces
# no impulses, and a kurtogram with nothing to find returns whichever band the noise
# favoured. Every asset came back with the same meaningless 500-1062 Hz band.
#
# The physical fact underneath: the demodulation band is set by the STRUCTURE (the
# housing/bearing resonance rung by defect impulses), not by the fault. It is
# established at commissioning with an impact/bump test, and it does not change
# unless the machine is rebuilt. So it is configuration, and `kurtogram_agreement`
# below reports whether a kurtogram run on degraded data recovers it -- which is the
# check that the commissioned value is still right.
COMMISSIONED_BAND = (3000.0, 4000.0)


def featurise(fleet: list[dict]) -> None:
    for a in fleet:
        t0 = time.perf_counter()
        a["band"] = COMMISSIONED_BAND
        a["feats"] = [features.snapshot_features(s, GEOM, band=COMMISSIONED_BAND)
                      for s in a["snaps"]]
        a["baseline"] = health.Baseline.fit(a["feats"][:BASELINE_CYCLES])
        # Does a kurtogram recover the commissioned band? Run it on the healthiest
        # and the most degraded snapshot of this asset.
        lo_h, hi_h, sk_h = features.select_band(a["snaps"][BASELINE_CYCLES // 2])
        lo_d, hi_d, sk_d = features.select_band(a["snaps"][-1])
        a["kurtogram"] = {
            "healthy_band": (lo_h, hi_h), "healthy_sk": sk_h,
            "degraded_band": (lo_d, hi_d), "degraded_sk": sk_d,
            "healthy_agrees": bool(lo_h <= COMMISSIONED_BAND[1] and hi_h >= COMMISSIONED_BAND[0]),
            "degraded_agrees": bool(lo_d <= COMMISSIONED_BAND[1] and hi_d >= COMMISSIONED_BAND[0]),
        }
        print(f"    {a['asset']:<12} kurtogram healthy {lo_h:.0f}-{hi_h:.0f} "
              f"(SK {sk_h:.1f}) / degraded {lo_d:.0f}-{hi_d:.0f} (SK {sk_d:.1f})  "
              f"{time.perf_counter()-t0:.1f}s", flush=True)


def kurtosis_leads_rms(fleet: list[dict]) -> list[dict]:
    """Measure the claim rather than asserting it.

    For each failing asset, find the first cycle at which the smoothed kurtosis
    exceeds its baseline p95, and the same for RMS. The gap between them is the
    lead time that impulsiveness buys over energy.
    """
    rows = []
    for a in fleet:
        if not a["failing"]:
            continue
        out = {"asset": a["asset"], "fault": a["fault"],
               "onset_cycle": a["truth"]["onset_cycle"]}
        for key in ("kurtosis", "rms", "env_kurtosis", "crest_factor"):
            v = np.array([f[key] for f in a["feats"]], dtype=float)
            base = v[:BASELINE_CYCLES]
            thr = float(np.percentile(base, 95))
            vs = health.smooth(v, 5)
            post = np.flatnonzero((vs > thr) & (np.arange(len(vs)) >= BASELINE_CYCLES))
            # first index after which it STAYS above, to avoid crediting a blip
            first = None
            for i in post:
                if (vs[i:] > thr).mean() > 0.9:
                    first = int(i)
                    break
            out[f"first_{key}"] = first
        rows.append(out)
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        # Re-render the document from the last run's measurements, so a wording
        # fix costs seconds instead of a full re-run, and so the prose can never
        # drift away from out/results.json.
        prev = json.loads((OUT / "results.json").read_text())
        (ROOT / "docs" / "RESULTS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/RESULTS.md from out/results.json")
        return
    t0 = time.perf_counter()
    res: dict = {"geometry": {
        "n_elements": GEOM.n_elements,
        "ball_diameter_mm": GEOM.ball_diameter_mm,
        "pitch_diameter_mm": GEOM.pitch_diameter_mm,
        "orders": {k: float(v) for k, v in GEOM.orders().items()},
        "harmonic_collision": features.HARMONIC_COLLISION,
    }}

    print("1/5 simulating fleet ...", flush=True)
    fleet = build_fleet(quick)
    res["fleet"] = [{"asset": a["asset"], "fault": a["fault"], "failing": a["failing"],
                     "n_cycles": len(a["snaps"]), "onset": a["truth"]["onset_cycle"]}
                    for a in fleet]

    print("2/5 extracting physics features ...", flush=True)
    featurise(fleet)
    res["commissioned_band"] = list(COMMISSIONED_BAND)
    res["kurtogram"] = [{"asset": a["asset"], "failing": a["failing"], **a["kurtogram"]}
                        for a in fleet]

    print("3/5 kurtosis-vs-RMS lead ...", flush=True)
    res["kurtosis_vs_rms"] = kurtosis_leads_rms(fleet)

    print("4/5 detector bake-off ...", flush=True)
    res["detectors"] = detectors.bakeoff(fleet, BASELINE_CYCLES)

    print("5/5 alarm policy, lead time vs false alarms ...", flush=True)
    res["operating_curve"] = detectors.lead_time_vs_false_alarms(fleet, BASELINE_CYCLES)
    res["alarm_state_machine"] = detectors.state_machine_report(fleet, BASELINE_CYCLES)
    res["diagnosis"] = detectors.diagnosis_report(fleet, BASELINE_CYCLES)
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/RESULTS.md and out/results.json ({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    g = res["geometry"]
    A("# ML-3 results — generated by `run_cm.py`, not hand-edited\n")
    A("> **Data provenance.** The vibration is *synthesised* by `src/bearing.py`: "
      "impulse trains at the kinematic fault frequencies, exciting a decaying "
      "structural resonance, with slip, speed jitter, shaft-rate content and noise. "
      "It is not CWRU and not IMS. Nothing here is comparable to a paper using those "
      "datasets. What the simulation buys is ground truth — fault type, onset cycle, "
      "failure cycle — which is what makes every claim below scoreable.\n")

    A("## 1. Fault frequencies from geometry\n")
    A(f"{g['n_elements']} rolling elements, ball ⌀{g['ball_diameter_mm']} mm, pitch "
      f"⌀{g['pitch_diameter_mm']} mm, 0° contact angle (SKF 6205-like).\n")
    A("| frequency | orders of shaft speed | at 29.95 Hz shaft |")
    A("|---|---|---|")
    for k in ("BPFO", "BPFI", "BSF", "FTF"):
        A(f"| {k} | {g['orders'][k]:.4f}× | {g['orders'][k]*29.95:.1f} Hz |")
    A(f"\n**A collision worth knowing about:** {g['harmonic_collision']}. Those two "
      "lines are closer together than the slip tolerance any real detector has to "
      "allow, so harmonic energy alone *cannot* separate an outer-race fault from an "
      "inner-race one on this bearing. The separator is sidebands — an inner-race "
      "defect rotates through the load zone, so its impulse train is amplitude-"
      "modulated at shaft rate and its lines carry ±1× shaft sidebands. An outer-race "
      "defect sits still and its lines are clean.")

    A("\n## 2. Where the demodulation band comes from (and a mistake worth keeping)\n")
    cb = res["commissioned_band"]
    A(f"The band is commissioned at **{cb[0]:.0f}–{cb[1]:.0f} Hz**, not learned. The first "
      "version of this code chose it adaptively by kurtogram over each asset's baseline "
      "period, and every asset came back with the same meaningless 500–1062 Hz band. The "
      "reason is physical, not numerical: *the baseline period is healthy*, a healthy "
      "bearing produces no impulses, and a kurtogram with nothing impulsive to find "
      "returns whichever band the noise happened to favour.\n")
    A("The demodulation band is a property of the STRUCTURE — the housing resonance that "
      "defect impulses ring — established by an impact test at commissioning. It is "
      "configuration. What a kurtogram is good for is *checking* it on degraded data:\n")
    A("| asset | kurtogram on healthy data | agrees | kurtogram on degraded data | agrees |")
    A("|---|---|---|---|---|")
    for k in res["kurtogram"]:
        hb, db = k["healthy_band"], k["degraded_band"]
        A(f"| {k['asset']} | {hb[0]:.0f}–{hb[1]:.0f} Hz (SK {k['healthy_sk']:.1f}) | "
          f"{'yes' if k['healthy_agrees'] else 'no'} | {db[0]:.0f}–{db[1]:.0f} Hz "
          f"(SK {k['degraded_sk']:.1f}) | {'yes' if k['degraded_agrees'] else 'no'} |")
    fail_k = [k for k in res["kurtogram"] if k["failing"]]
    if fail_k:
        agree_d = sum(k["degraded_agrees"] for k in fail_k)
        agree_h = sum(k["healthy_agrees"] for k in fail_k)
        A(f"\nOn failing assets the kurtogram recovers the commissioned band from degraded "
          f"data in **{agree_d}/{len(fail_k)}** cases, and from healthy data in "
          f"**{agree_h}/{len(fail_k)}**. That asymmetry is the whole argument for treating "
          "the band as commissioned configuration with a kurtogram *audit*, rather than as "
          "something to re-derive every acquisition.")

    A("\n## 3. Kurtosis leads RMS — measured, not asserted\n")
    A("First cycle at which each feature rises above its own baseline 95th percentile "
      "*and stays there*. Degradation onset is at the cycle in column 2.\n")
    A("| asset | onset | kurtosis | env. kurtosis | crest factor | RMS | lead of kurtosis over RMS |")
    A("|---|---|---|---|---|---|---|")
    leads = []
    for r in res["kurtosis_vs_rms"]:
        fk, fr = r.get("first_env_kurtosis"), r.get("first_rms")
        lead = (fr - fk) if (fk is not None and fr is not None) else None
        if lead is not None:
            leads.append(lead)
        f = lambda k: str(r[k]) if r.get(k) is not None else "never"
        A(f"| {r['asset']} | {r['onset_cycle']} | {f('first_kurtosis')} | "
          f"{f('first_env_kurtosis')} | {f('first_crest_factor')} | {f('first_rms')} | "
          f"{lead if lead is not None else '—'} |")
    if leads:
        med = float(np.median(leads))
        n_pos = sum(1 for x in leads if x > 0)
        A(f"\nMedian lead of envelope kurtosis over RMS: **{med:.0f} cycles**, positive on "
          f"{n_pos} of {len(leads)} assets.\n")
        A("The textbook claim is that kurtosis moves first: a single spall changes the "
          "*shape* of the signal (one impulse per element pass) long before it changes its "
          "*energy*, and RMS integrates over the whole record so a brief impulse barely "
          "moves it.")
        if n_pos < len(leads) or med < 15:
            A("\n**That claim is only weakly supported by this data, and the reason is a "
              "limitation of my simulator rather than a refutation of the physics.** The "
              "degradation model raises a single amplitude parameter smoothly, so the "
              "impulse train gains energy and impulsiveness *together*; a real bearing goes "
              "through a distinct phase where one small spall produces sharp impulses at "
              "almost constant total energy, and that phase is where the kurtosis lead is "
              f"won. On {len(leads) - n_pos} of {len(leads)} assets the lead is negative or "
              "zero, and the inner-race cases are the worst of them, because their "
              "amplitude modulation lifts RMS early as well.\n")
            A("The honest reading: on this data the two indicators are near-simultaneous, "
              "the measured advantage is small, and I would not sell a lead-time claim on "
              "kurtosis alone. What actually delivers the lead here is the envelope band "
              "energy at the fault frequency — section 5 — which is physics-located rather "
              "than a generic shape statistic.")
        else:
            A("\nThat gap is the P-F interval, and it is the reason to instrument a bearing "
              "rather than wait for it to get loud.")

    A("\n## 4. Detector bake-off — statistical vs ML vs deep, on equal footing\n")
    A("All three consume the SAME physics features and are tuned to the SAME false-alarm "
      "budget on healthy assets, because that is the only way the comparison means "
      "anything.\n")
    A("| detector | median lead time (cycles) | worst-case lead | false alarms per asset-life | assets detected |")
    A("|---|---|---|---|---|")
    for d in res["detectors"]:
        A(f"| {d['name']} | {d['median_lead']:.0f} | {d['min_lead']:.0f} | "
          f"{d['false_alarms_per_asset']:.2f} | {d['n_detected']}/{d['n_failing']} |")
    leads_d = [d["median_lead"] for d in res["detectors"]]
    spread = max(leads_d) - min(leads_d)
    best = max(res["detectors"], key=lambda d: d["median_lead"])
    n_assets = res["detectors"][0]["n_failing"]
    if spread <= 5:
        cyc = "cycle" if spread == 1 else "cycles"
        A(f"\n**There is no winner, and that is the result.** The three families land within "
          f"{spread:.0f} {cyc} of each other ({min(leads_d):.0f}–{max(leads_d):.0f}) at an "
          f"identical false-alarm budget, on {n_assets} failing assets. A "
          f"{spread:.0f}-{cyc} spread across {n_assets} trajectories is not a difference; "
          f"it is the sampling noise of a median over {n_assets} numbers, and calling "
          f"{best['name']} the winner on that basis would be exactly the mistake this "
          "table exists to prevent.\n")
        A("What I would ship is **Hotelling T²**. It is thirty lines, it has no training "
          "step and therefore no retraining pipeline, its score decomposes into per-feature "
          "contributions so the alarm can be explained to the operator who receives it, and "
          "its failure modes are a hundred years old and documented. The autoencoder buys "
          "nothing here and costs a model registry, a GPU-free inference path, and an "
          "answer to \"why did it alarm\" that I do not have.\n")
        A("The reason the deep model does not win is not that deep models are bad. It is "
          "that **the features already contain the physics**. Once the envelope energy at "
          "BPFO is a column, the remaining problem is 'is this column unusually large', "
          "which is a job for a covariance and not for representation learning. A deep "
          "model earns its place when the features are *not* known — and on a bearing with "
          "published geometry, they are.")
    else:
        A(f"\nWinner on median lead at the matched budget: **{best['name']}**, by "
          f"{spread:.0f} cycles over the worst family.")

    A("\n## 5. The operating curve — lead time against false alarms\n")
    A("The deliverable chart of this whole project, as a table. Sweeping the alarm "
      "threshold trades warning against nuisance.\n")
    A("| health-index alarm threshold | median lead (cycles) | P05 lead | false alarms per healthy asset-life | assets missed |")
    A("|---|---|---|---|---|")
    for r in res["operating_curve"]:
        A(f"| {r['threshold']:.0f} | {r['median_lead']:.0f} | {r['p05_lead']:.0f} | "
          f"{r['false_alarms_per_asset']:.2f} | {r['n_missed']} |")
    budget = [r for r in res["operating_curve"] if r["false_alarms_per_asset"] <= 1.0]
    if budget:
        pick = max(budget, key=lambda r: r["median_lead"])
        A(f"\nAt a budget of **≤1 false alarm per asset-lifetime**, the best available "
          f"threshold is {pick['threshold']:.0f}, delivering a median of "
          f"{pick['median_lead']:.0f} cycles of warning with {pick['n_missed']} assets "
          "missed. That sentence — a lead time quoted *at* a false-alarm budget — is the "
          "operational contract. A lead time quoted without one is a number chosen after "
          "seeing the answer.")

    A("\n## 6. Alarm state machine: does it flap?\n")
    A("| asset | first sustained ALERT | cycles before failure | flaps (in-and-out of alarm) | final state |")
    A("|---|---|---|---|---|")
    for r in res["alarm_state_machine"]:
        A(f"| {r['asset']} | {r['first_alert'] if r['first_alert'] is not None else '—'} | "
          f"{r['lead'] if r['lead'] is not None else '—'} | {r['flaps']} | {r['final_state']} |")
    total_flaps = sum(r["flaps"] for r in res["alarm_state_machine"])
    A(f"\nTotal flaps across the fleet: **{total_flaps}**. Hysteresis (separate enter and "
      "exit thresholds) plus 3-of-5 persistence is what produces that number. Without the "
      "exit/enter gap, a score sitting on the threshold toggles every acquisition, and an "
      "operator who is interrupted six times by the same bearing stops reading the system "
      "in week three. This is the whole answer to \"the last vendor got switched off after "
      "six weeks\": the vendor optimised detection and never measured flapping.")

    A("\n## 7. Diagnosis — naming the fault, and refusing to\n")
    A("| true fault | healthy | BPFO | BPFI | BSF | indeterminate |")
    A("|---|---|---|---|---|---|")
    for row in res["diagnosis"]["confusion"]:
        A(f"| {row['true']} | {row.get('healthy',0)} | {row.get('BPFO',0)} | "
          f"{row.get('BPFI',0)} | {row.get('BSF',0)} | {row.get('indeterminate',0)} |")
    d = res["diagnosis"]
    A(f"\nOn cycles where a fault is genuinely developed (severity > "
      f"{d['severity_threshold']}), the diagnosis names the correct race "
      f"{d['accuracy_when_developed']*100:.0f}% of the time, and says "
      f"\"indeterminate\" {d['indeterminate_when_developed']*100:.0f}% of the time. "
      "Indeterminate is a supported output, not a failure: \"the outer race is spalled, "
      "order part X\" and \"something is wrong with this bearing\" are different work "
      "orders, and issuing the first when you only know the second is how a monitoring "
      "team loses its credibility with maintenance.")

    A("\n---\n*Generated from `out/results.json`. Every lead time is measured against a "
      "known failure cycle; every false alarm is counted on assets that never fail.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
