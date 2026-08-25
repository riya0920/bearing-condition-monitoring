"""ML-3 pass 5: the speed-varying case.

    python run_speed_varying.py
    python run_speed_varying.py --quick

Item 9 of the not-built list: no run-up, no coast-down, no order tracking --
"which is where fixed-frequency band energy stops working entirely". That claim
had never been measured, so this run measures it, and then measures the fix.

Three things, in the order that makes them believable:

  1  a sweep sweep. Speed variation from 0% (constant) to +/-50%, and what each
     detector calls at each width. The first column is the control: at 0% the
     two methods are analysing the same thing and must agree.

  2  the cost of having no tacho. The same comparison with the shaft phase
     recovered from the vibration signal instead of handed over by a keyphasor.
     Most installations have no tacho, so this is the number that decides
     whether the method is usable in the situation this project is written for.

  3  REAL CWRU RECORDS, at constant speed. Order tracking has to agree with the
     frequency method there. An implementation that disagrees on constant speed
     is not a better method, it is a broken one, and this is the only check
     available that can tell those apart.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import bearing as B              # noqa: E402
import cwru                      # noqa: E402
import features as F             # noqa: E402
import order_tracking as OT      # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
QUICK = "--quick" in sys.argv

GEOM = B.BearingGeometry()
FAULTS = ("BPFO", "BPFI")
CENTRE_HZ = 29.95
SPREADS = (0.0, 0.05, 0.10, 0.25, 0.50)
# Severities at which the FIXED method starts near the healthy gate rather than
# a hundred times above it -- i.e. what an early fault looks like.
SEVERITIES = (0.05, 0.10, 0.20)
GATE = 6.0                       # features.diagnose's min_ratio, calibrated in pass 5
N_TRIALS = 4 if QUICK else 12
DURATION_S = 2.0
BAND = (2500.0, 4500.0)          # the housing resonance the generator excites


# ---------------------------------------------------------------------------
# one trial
# ---------------------------------------------------------------------------

def one_trial(fault: str, spread: float, seed: int, kind: str = "linear",
              severity: float = 1.0) -> dict:
    """Generate a sweep and score it three ways.

    Same signal, three analyses:
      fixed       envelope spectrum in TIME, fault frequencies at the MEAN speed
      order_tacho envelope order spectrum, phase from the true speed profile
      order_est   envelope order spectrum, phase tracked off the signal
    """
    rng = np.random.default_rng(seed)
    f0 = CENTRE_HZ * (1 - spread)
    f1 = CENTRE_HZ * (1 + spread)
    speed = OT.sweep_profile(f0, f1, DURATION_S, kind=kind)
    x = OT.simulate_sweep(GEOM, fault, speed, severity, rng)
    rev_true = OT.phase_from_speed(speed)

    # -- fixed frequency, the method every other detector in this project uses.
    # Given the MEAN speed, which is the most favourable single number
    # available: a real system would use nameplate speed or a stale tacho
    # reading and do worse.
    freqs, spec, _ = F.envelope_spectrum(x, BAND, OT.FS)
    fixed = OT.frequency_ratios(freqs, spec, GEOM, float(speed.mean()))

    # -- order tracked, exact phase
    o, s = OT.envelope_order_spectrum(x, rev_true, BAND, OT.FS)
    tacho = OT.order_ratios(o, s, GEOM)

    # -- order tracked, phase estimated from the signal
    try:
        est_speed = OT.track_speed(x, OT.FS, search=(CENTRE_HZ * 0.4,
                                                     CENTRE_HZ * 1.8))
        rev_est = OT.phase_from_speed(est_speed)
        o2, s2 = OT.envelope_order_spectrum(x, rev_est, BAND, OT.FS)
        est = OT.order_ratios(o2, s2, GEOM)
        speed_err = float(np.mean(np.abs(est_speed - speed) / speed))
    except Exception as exc:                                  # noqa: BLE001
        est, speed_err = {k: 0.0 for k in tacho}, float("nan")
        est["error"] = str(exc)

    return {
        "fault": fault, "spread": spread, "kind": kind, "seed": seed,
        "severity": severity,
        "fixed": fixed, "order_tacho": tacho, "order_est": est,
        "speed_error": speed_err,
        "call_fixed": OT.call(fixed)[0],
        "call_tacho": OT.call(tacho)[0],
        "call_est": OT.call(est)[0],
        "ratio_fixed": fixed[fault], "ratio_tacho": tacho[fault],
        "ratio_est": est.get(fault, 0.0),
    }


def sweep_study() -> dict:
    rows = []
    for spread in SPREADS:
        for fault in FAULTS:
            for k in range(N_TRIALS):
                rows.append(one_trial(fault, spread, seed=1000 + k))
        done = [r for r in rows if r["spread"] == spread]
        print(f"  +/-{spread:.0%}: fixed {_hit(done, 'call_fixed'):.0%} correct, "
              f"tacho {_hit(done, 'call_tacho'):.0%}, "
              f"estimated {_hit(done, 'call_est'):.0%}", flush=True)
    return {"rows": rows, "spreads": list(SPREADS), "n_trials": N_TRIALS}


def _hit(rows, key) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r[key] == r["fault"] for r in rows]))


def severity_study() -> dict:
    """The consequence the correct-call rate hides.

    The headline table keeps calling the fault correctly out to +/-25% even
    though the fixed-frequency ratio has fallen by most of an order of
    magnitude, because every COMPETING candidate is smeared too and the winner
    only has to win. That is real, and it is also why the ratio table matters
    more than the call table: the margin is what a marginal fault spends.

    So: repeat at severities where the fixed method starts near the healthy
    gate rather than a hundred times above it, which is what an EARLY fault
    looks like -- the ones the whole project exists to catch.
    """
    rows = []
    for sev in SEVERITIES:
        for spread in (0.0, 0.25, 0.50):
            for fault in FAULTS:
                for k in range(N_TRIALS):
                    rows.append(one_trial(fault, spread, seed=3000 + k,
                                          severity=sev))
        print(f"  severity {sev:.2f} done", flush=True)
    return {"rows": rows, "severities": list(SEVERITIES),
            "spreads": [0.0, 0.25, 0.50], "gate": GATE}


def coast_study() -> dict:
    """A coast-down is not a reversed run-up: the speed decays exponentially, so
    most of the record sits near the FINAL speed and only the beginning moves
    fast. If the two shapes scored the same, one of them would not be
    implemented."""
    rows = []
    for fault in FAULTS:
        for k in range(N_TRIALS):
            rows.append(one_trial(fault, 0.50, seed=2000 + k, kind="coast"))
    return {"rows": rows,
            "fixed": _hit(rows, "call_fixed"),
            "tacho": _hit(rows, "call_tacho"),
            "est": _hit(rows, "call_est")}


# ---------------------------------------------------------------------------
# the check that decides whether any of this is real
# ---------------------------------------------------------------------------

def constant_speed_check(n_files: int = 8, n_snaps: int = 6) -> dict:
    """Real CWRU records. Constant speed, so the two methods MUST agree.

    Not a formality. Angular resampling has several ways to be silently wrong --
    an off-by-one in the phase integration, an order axis scaled by the wrong
    factor, a samples-per-rev that aliases the fault order onto another one --
    and every one of them still produces a plausible-looking spectrum. On
    constant speed the angle axis is a linear function of the time axis, so the
    two spectra are the same measurement in different units, and any
    disagreement is a bug rather than a finding.
    """
    if not cwru.available():
        return {"available": False}
    man = cwru.manifest().get("files", {})
    fs = cwru.FS_CWRU
    band = cwru.band_for_fs(fs)
    out = []
    for fid, meta in list(man.items())[:n_files]:
        try:
            rec = cwru.load_file(fid)
            hz = cwru.shaft_hz(rec)
        except Exception as exc:                              # noqa: BLE001
            out.append({"fid": fid, "error": str(exc)})
            continue
        truth = meta.get("fault") if isinstance(meta, dict) else None
        for j, snap in enumerate(cwru.snapshots(rec["de"], n=n_snaps,
                                                length=fs, seed=3)):
            freqs, spec, _ = F.envelope_spectrum(snap, band, fs)
            fixed = OT.frequency_ratios(freqs, spec, GEOM, hz)
            # Constant speed: revolutions elapsed is just time x speed.
            rev = np.arange(len(snap)) / fs * hz
            o, s = OT.envelope_order_spectrum(snap, rev, band, fs)
            order = OT.order_ratios(o, s, GEOM)
            out.append({"fid": fid, "snap": j, "truth": truth, "shaft_hz": hz,
                        "call_fixed": OT.call(fixed)[0],
                        "call_order": OT.call(order)[0],
                        "ratio_fixed": {k: round(v, 3) for k, v in fixed.items()},
                        "ratio_order": {k: round(v, 3) for k, v in order.items()}})
    scored = [r for r in out if "call_fixed" in r]
    agree = [r for r in scored if r["call_fixed"] == r["call_order"]]
    # Correlation of the two ratio vectors across every fault type and snapshot:
    # agreeing on the CALL is a coarse check, agreeing on the numbers is not.
    a = np.array([[r["ratio_fixed"][k] for k in ("BPFO", "BPFI", "BSF2", "FTF")]
                  for r in scored]).ravel()
    b = np.array([[r["ratio_order"][k] for k in ("BPFO", "BPFI", "BSF2", "FTF")]
                  for r in scored]).ravel()
    rho = float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]) if len(a) > 3 else None
    return {"available": True, "rows": out, "n": len(scored),
            "agreement": len(agree) / len(scored) if scored else None,
            "log_ratio_correlation": rho,
            "n_files": len({r["fid"] for r in scored})}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(sw: dict, sev: dict, coast: dict, real: dict) -> str:
    L = []
    A = L.append
    A("# ML-3 pass 5 — the speed-varying case\n")
    A("Item 9 of the not-built list said no run-up, no coast-down and no order "
      "tracking, *\"which is where fixed-frequency band energy stops working "
      "entirely\"*. That was an assertion. Here it is as a measurement, the fix, "
      "and — first — the correction, because **the assertion was too strong.**\n")

    A("## The claim was overstated\n")
    A("Fixed-frequency detection does not stop working at the first sign of "
      "speed variation. It keeps naming the right fault well past the point "
      "where its evidence has mostly evaporated, because every competing "
      "candidate smears too and the winner only has to win. What collapses is "
      "the MARGIN, and the margin is what an early fault has to spend. Both "
      "tables below are needed: the call rate says when the method fails "
      "outright, the ratio says when it stopped being worth trusting.\n")

    A("## Why it degrades, before the numbers\n")
    A("Every detector in this project locates energy at a FREQUENCY. BPFO is "
      f"{GEOM.orders()['BPFO']:.3f}× shaft, so at {CENTRE_HZ:.2f} Hz shaft it "
      f"is {GEOM.fault_frequencies(CENTRE_HZ)['BPFO']:.1f} Hz and the search "
      "window goes there with a 2% tolerance. During a run-up the line sweeps "
      "clean across that window and out the other side, so the energy is "
      "spread over a band far wider than the tolerance and **the peak the "
      "detector is looking for does not exist at any single frequency.**\n")
    A("Order tracking resamples onto uniform shaft ANGLE. A defect strikes once "
      "every fixed number of revolutions, not once every fixed number of "
      "seconds, so in the angle domain the line is stationary again — at an "
      "*order* rather than a frequency. `BearingGeometry.orders()` has been in "
      "this codebase since the first pass and nothing had ever used it.\n")

    A("## 1. The sweep sweep\n")
    A(f"{sw['n_trials']} trials per fault per width, two faults, "
      f"{DURATION_S:.0f}-second records, at full fault severity. Correct-call "
      f"rate:\n")
    A("| speed variation | fixed frequency | order, tacho phase | "
      "order, phase estimated from the signal |")
    A("|---|---|---|---|")
    for sp in sw["spreads"]:
        rows = [r for r in sw["rows"] if r["spread"] == sp]
        A(f"| ±{sp:.0%} | {_hit(rows, 'call_fixed'):.0%} | "
          f"{_hit(rows, 'call_tacho'):.0%} | {_hit(rows, 'call_est'):.0%} |")
    A("")
    zero = [r for r in sw["rows"] if r["spread"] == 0.0]
    A(f"**The first row is the control.** At constant speed the two domains are "
      f"analysing the same thing and score {_hit(zero, 'call_fixed'):.0%} "
      f"against {_hit(zero, 'call_tacho'):.0%}. A table whose first row "
      "disagreed would be measuring an implementation bug and nothing else.\n")

    A("Now the ratio at the true fault order, which is where the mechanism "
      "shows:\n")
    A("| speed variation | fixed frequency | order, tacho | order, estimated |")
    A("|---|---|---|---|")
    for sp in sw["spreads"]:
        rows = [r for r in sw["rows"] if r["spread"] == sp]
        A(f"| ±{sp:.0%} | {np.median([r['ratio_fixed'] for r in rows]):.1f} | "
          f"{np.median([r['ratio_tacho'] for r in rows]):.1f} | "
          f"{np.median([r['ratio_est'] for r in rows]):.1f} |")
    A("")
    wide = [r for r in sw["rows"] if r["spread"] == max(sw["spreads"])]
    r0 = np.median([r["ratio_fixed"] for r in zero])
    r1 = np.median([r["ratio_fixed"] for r in wide])
    t0 = np.median([r["ratio_tacho"] for r in zero])
    t1 = np.median([r["ratio_tacho"] for r in wide])
    A(f"The fixed-frequency ratio falls **{r0:.0f} → {r1:.0f}**, a factor of "
      f"{r0 / max(r1, 1e-9):.0f}, while the order-tracked ratio goes "
      f"{t0:.0f} → {t1:.0f} — flat, or slightly up. **The energy did not go "
      "anywhere. It is exactly where it always was, in angle.** The call "
      f"survives to ±{max(sw['spreads']):.0%} only because a strong fault can "
      f"afford to lose {r0 / max(r1, 1e-9):.0f}× of its evidence and still "
      "outrank the alternatives.\n")

    A("## 2. Which is why the next table matters more\n")
    A(f"The same comparison at severities where the fixed method starts near "
      f"the healthy gate of {sev['gate']:.0f} rather than a hundred times above "
      f"it. That is what an EARLY fault looks like, and early faults are what "
      f"the whole project exists to catch.\n")
    A("| severity | speed variation | fixed: correct | fixed: median ratio | "
      "order: correct | order: median ratio |")
    A("|---|---|---|---|---|---|")
    for s_ in sev["severities"]:
        for sp in sev["spreads"]:
            rows = [r for r in sev["rows"]
                    if r["spread"] == sp and r["severity"] == s_]
            if not rows:
                continue
            A(f"| {s_:.2f} | ±{sp:.0%} | {_hit(rows, 'call_fixed'):.0%} | "
              f"{np.median([r['ratio_fixed'] for r in rows]):.1f} | "
              f"{_hit(rows, 'call_tacho'):.0%} | "
              f"{np.median([r['ratio_tacho'] for r in rows]):.1f} |")
    A("")

    # Paired, because "how many fall below the gate" counts faults that were
    # never detectable at any speed. The question is narrower and harder: of
    # the faults this detector DOES find at constant speed, how many does it
    # lose when the speed moves? Trials share seeds across widths, so the same
    # fault can be followed from one row to the next.
    def _key(r):
        return (r["severity"], r["fault"], r["seed"])

    at_rest = {_key(r): r for r in sev["rows"] if r["spread"] == 0.0}
    lost_fixed = lost_order = base = 0
    for r in sev["rows"]:
        if r["spread"] == 0.0:
            continue
        b = at_rest.get(_key(r))
        if b is None or b["ratio_fixed"] < sev["gate"]:
            continue          # never detectable; not this study's business
        base += 1
        lost_fixed += r["ratio_fixed"] < sev["gate"]
        lost_order += r["ratio_tacho"] < sev["gate"]
    A(f"Counting only the weak faults this detector **does** find at constant "
      f"speed ({base} paired trials - same fault, same seed, speed varied): "
      f"fixed-frequency analysis loses **{lost_fixed} of {base}** below the "
      f"healthy gate, and order tracking loses **{lost_order} of {base}**. "
      f"A fault that falls under the gate is not misnamed. It is called "
      f"healthy, and nobody looks at it again.\n")
    _s20 = [r["ratio_fixed"] for r in sev["rows"]
            if r["severity"] == 0.2 and r["spread"] == 0.0]
    A("Read the severity-0.20 rows on their own. At constant speed the fixed "
      f"method is right every time with a ratio near {np.median(_s20):.0f}; at "
      "±25% the same faults sit just above the gate and it names none of "
      "them, while order tracking has not moved. **That is the row the "
      "strong-fault table in section 1 was hiding.**\n")

    A("## 3. The cost of having no tacho\n")
    err = [r["speed_error"] for r in sw["rows"] if np.isfinite(r["speed_error"])]
    A(f"Speed tracked off the vibration signal instead of a keyphasor. Median "
      f"absolute speed error across every trial: **{np.median(err) * 100:.2f}%**.\n")
    gaps = []
    for sp in sw["spreads"]:
        rows = [r for r in sw["rows"] if r["spread"] == sp]
        gaps.append((sp, _hit(rows, "call_tacho") - _hit(rows, "call_est"),
                     np.median([r["ratio_tacho"] for r in rows]),
                     np.median([r["ratio_est"] for r in rows])))
    A("| speed variation | correct-call cost of estimating | "
      "ratio, tacho | ratio, estimated |")
    A("|---|---|---|---|")
    for sp, g, rt, re_ in gaps:
        A(f"| ±{sp:.0%} | {g:+.0%} | {rt:.1f} | {re_:.1f} |")
    A("")
    A("The call rate barely moves and **the ratio loses about half its "
      "margin at every width, including at constant speed**. A constant speed "
      "estimated one bin off is a constant speed ERROR, and a constant speed "
      "error integrates into a phase that drifts linearly across the record — "
      "so the impulses at the end land at a different angle from the ones at "
      "the start and the line smears. Estimating the phase does not cost "
      "accuracy here; it costs the margin that keeps a weak fault above the "
      "gate, which is the same currency the previous section was spending.\n")

    A("## 4. Coast-down\n")
    A(f"A coast-down is not a reversed run-up — a machine losing energy to "
      f"friction decays roughly exponentially, so the fast part is at the "
      f"beginning and most of the record sits near the final speed. Over the "
      f"same ±50% range: fixed {coast['fixed']:.0%}, order/tacho "
      f"{coast['tacho']:.0%}, order/estimated {coast['est']:.0%}.\n")

    A("## 5. The check that decides whether any of this is real\n")
    if not real.get("available"):
        A("CWRU data is not present in this checkout, so the agreement check "
          "did not run. **Everything above is therefore simulation only**, and "
          "the number that would tell you the order-tracking implementation is "
          "not silently wrong is missing. Run `fetch_cwru.py`.\n")
    else:
        A(f"{real['n']} snapshots from {real['n_files']} real CWRU records, all "
          f"at constant speed. The two methods agree on the call "
          f"**{real['agreement']:.0%}** of the time, and their ratio vectors "
          f"correlate at **r = {real['log_ratio_correlation']:.3f}** in log "
          f"space.\n")
        A("Angular resampling has several ways to be silently wrong — an "
          "off-by-one in the phase integration, an order axis scaled by the "
          "wrong factor, a samples-per-rev low enough to alias one fault order "
          "onto another — and every one of them still produces a "
          "plausible-looking spectrum with a peak in it. On constant speed the "
          "angle axis is a linear function of the time axis, so the two "
          "spectra are the same measurement in different units. This is the "
          "only available check that can tell a better method from a broken "
          "one, and it is the reason the simulated tables above are worth "
          "reading at all.\n")

    A("## Honest limits\n")
    for lim in OT.LIMITS:
        A(f"- {lim}")
    A("- The comparison gives the fixed-frequency method the MEAN speed, which "
      "is the most favourable single number available to it. A real system "
      "would use nameplate speed or a stale tacho reading and do worse, so "
      "every gap above is a lower bound on the gap in a plant.")
    A("- Two fault types. Ball faults are excluded because this project "
      "already knows its line-energy detector is close to the wrong instrument "
      "for them (19% on real data), and adding a case that fails for an "
      "unrelated reason would muddy what these tables measure.")
    A("- The severity table's 'missed fault' count is against this project's "
      "own gate of 6.0, which was calibrated on constant-speed data. A plant "
      "running speed-varying machines would calibrate its own, and the right "
      "gate for order-tracked ratios is not the right gate for fixed-frequency "
      "ones.")
    return "\n".join(L)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    print("1/4 speed-variation sweep ...", flush=True)
    sw = sweep_study()
    print("2/4 weak faults ...", flush=True)
    sev = severity_study()
    print("3/4 coast-down ...", flush=True)
    coast = coast_study()
    print("4/4 real CWRU constant-speed agreement ...", flush=True)
    real = constant_speed_check()
    if real.get("available"):
        print(f"  agreement {real['agreement']:.0%} over {real['n']} snapshots, "
              f"log-ratio r = {real['log_ratio_correlation']:.3f}", flush=True)
    res = {"sweep": sw, "severity": sev, "coast": coast, "real": real}
    (OUT / "speed_varying.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    (DOCS / "SPEED_VARYING.md").write_text(report(sw, sev, coast, real),
                                           encoding="utf-8")
    print(f"\nwrote docs/SPEED_VARYING.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
