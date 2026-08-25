"""The process comparison, re-run on data I did not generate.

The README said it plainly and left it there:

    The process case is a simulator I wrote, which is the same circularity the
    bearing side had before CWRU -- and the bearing side is exactly where real
    data overturned a claim I was confident in. The residual model winning is
    the result most likely to be an artefact of a generator built from linear
    relationships, and I would not defend it until it has run on real TE or
    SKAB data.

So it runs on both. The detectors are the ones already in `src/pca_monitor.py`
and `src/process.py`, unchanged -- the only new code is the loading, and that is
the point: if the ranking flips, it flips because of the data.

WHAT COUNTS AS A FAIR TEST HERE. Three things, and the third is the one that is
easy to skip:

  * The limits are fitted on NORMAL data only and applied to the fault runs.
    Fitting on the run you are scoring is how a monitor gets a detection delay
    of zero and a false-alarm rate to match.
  * The false-alarm rate is measured on a normal run held out from fitting
    (`d00_te`), not on the pre-fault segment of the fault runs. Those segments
    are short and the pre-fault period of a run that later goes wrong is not a
    clean normal sample.
  * The three faults the literature agrees are near-undetectable (3, 9, 15) are
    included. A comparison that quietly drops them reports a better number for
    every method, and the interesting question -- whether a method claims to
    detect what nobody detects -- is exactly the one dropping them hides.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pca_monitor as PM      # noqa: E402
import process as PR          # noqa: E402

DATA = ROOT / "data" / "PROCESS"
OUT = ROOT / "out"
DOCS = ROOT / "docs"

M_OF_N = 3          # persistence: m consecutive samples above the limit
ALPHA = 0.99        # limit quantile, matching the synthetic study
# Every detector is held to the SAME false-alarm budget: alarm episodes per
# 1000 normal samples. One per 1000 is roughly one nuisance interruption per
# two days at 3-minute sampling, which is about what a control room tolerates.
FA_BUDGET_PER_1000 = 1.0


# ---------------------------------------------------------------------------
# detectors -- the same four the synthetic study compared
# ---------------------------------------------------------------------------

class _Bank:
    """The five detectors, each exposing a CONTINUOUS statistic.

    Continuous rather than a boolean, because the thresholds have to be set
    afterwards and jointly -- see `calibrate`.
    """

    def __init__(self, train: np.ndarray):
        self.mu = train.mean(0)
        sd = train.std(0, ddof=1)
        self.sd = np.where(sd < 1e-12, 1.0, sd)
        self.pca = PM.PCAMonitor().fit(train, alpha=ALPHA)
        self.res = PR.ResidualModel().fit(train)
        r = self.res.residuals(train)
        rs = r.std(0, ddof=1)
        self.rs = np.where(rs < 1e-12, 1.0, rs)

    def stats(self, x: np.ndarray) -> dict:
        s = self.pca.score(x)
        # Each statistic is reduced to ONE number per sample. The univariate
        # "wall of charts" is the worst z-score across tags, which is what an
        # operator watching 52 charts effectively responds to -- and it is also
        # what stops it getting 52 chances to alarm while T2 gets one.
        uni = np.abs((x - self.mu) / self.sd).max(axis=1)
        resid = np.abs(self.res.residuals(x) / self.rs).max(axis=1)
        t2, spe = np.asarray(s["t2"], float), np.asarray(s["spe"], float)
        return {
            "univariate 3-sigma (the wall of charts)": uni,
            "PCA T2 only": t2,
            "PCA SPE only": spe,
            # A union of two statistics needs a single scale to threshold, so
            # each is divided by its own 99% training limit first. Thresholding
            # the raw max of a T2 and an SPE would just be thresholding whichever
            # happens to be numerically larger.
            "T2 or SPE": np.maximum(t2 / max(float(s["t2_limit"]), 1e-12),
                                    spe / max(float(s["spe_limit"]), 1e-12)),
            "residual (model-based)": resid,
        }


def calibrate(bank: "_Bank", normal: np.ndarray, target_per_1000: float,
              m: int = M_OF_N) -> dict:
    """Thresholds chosen so every detector runs at the SAME false-alarm rate.

    This is the whole methodology, and the first version of this script did not
    have it. Setting each detector at its own 99% quantile and then comparing
    detection delays compares THRESHOLDS, not methods: the loosest one wins
    every race and pays for it in false alarms that the delay column never
    shows. It produced a table where all five detectors found all ten TE faults
    including the three the literature calls undetectable, at false-alarm rates
    of four to six percent. That table was wrong in the most flattering
    possible direction.

    The budget is expressed as alarm EPISODES per 1000 samples on a held-out
    normal run, m-of-n persistence included, because that is what a control
    room absorbs -- a single excursion that trips five samples in a row is one
    interruption, not five.
    """
    out = {}
    st = bank.stats(normal)
    n = len(normal)
    for name, s in st.items():
        # Search the threshold over the observed statistic's own quantiles.
        lo, hi = float(np.min(s)), float(np.max(s)) * 1.5 + 1e-9
        best = hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rate = PM.false_alarm_rate(s > mid, n + 1, m=m)
            if rate > target_per_1000:
                lo = mid
            else:
                best = mid
                hi = mid
        out[name] = {"threshold": float(best),
                     "achieved_per_1000": float(
                         PM.false_alarm_rate(s > best, n + 1, m=m))}
    return out


def _alarms(bank: "_Bank", test: np.ndarray, thresholds: dict) -> dict:
    st = bank.stats(test)
    return {k: st[k] > thresholds[k]["threshold"] for k in st}


def _score(alarm: np.ndarray, onset: int, normal_alarm: np.ndarray) -> dict:
    d = PM.detection_delay(alarm, onset, m=M_OF_N)
    fa = PM.false_alarm_rate(normal_alarm, len(normal_alarm) + 1, m=M_OF_N)
    post = alarm[onset:]
    return {"delay": None if d is None else int(d),
            "detected": d is not None,
            "alarm_frac_after": float(post.mean()) if len(post) else float("nan"),
            "false_per_1000": float(fa)}


DETECTORS = ["univariate 3-sigma (the wall of charts)", "PCA T2 only",
             "PCA SPE only", "T2 or SPE", "residual (model-based)"]


def summarise(rows: list, exclude=()) -> dict:
    out = {}
    for name in DETECTORS:
        det = [r["detectors"][name] for r in rows
               if r.get("fault") not in exclude]
        found = [d for d in det if d["delay"] is not None]
        out[name] = {
            "detected": len(found), "of": len(det),
            "median_delay": float(np.median([d["delay"] for d in found]))
            if found else None,
            "mean_false_per_1000": float(np.mean([d["false_per_1000"]
                                                  for d in det])),
        }
    return out

# ---------------------------------------------------------------------------
# Tennessee Eastman
# ---------------------------------------------------------------------------

def run_te() -> dict:
    p = DATA / "te.npz"
    if not p.exists():
        return {"available": False, "why": f"no {p}; run fetch_process.py"}
    z = np.load(p, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    onset = int(z["test_onset"])
    train = z["train_normal"]
    normal_test = z["test_normal"]
    ids = [int(i) for i in z["fault_ids"]]
    desc = {i: str(d) for i, d in zip(ids, z["fault_desc"])}

    bank = _Bank(train)
    # Thresholds set on the held-out normal run, to a common budget. `d00_te`
    # is normal throughout and was not used to fit anything.
    thr = calibrate(bank, normal_test, FA_BUDGET_PER_1000)
    normal_alarms = _alarms(bank, normal_test, thr)

    rows = []
    for fid in ids:
        key = f"test_fault_{fid:02d}"
        if key not in z.files:
            continue
        test = z[key]
        alarms = _alarms(bank, test, thr)
        det = {name: _score(a, onset, normal_alarms[name])
               for name, a in alarms.items()}
        # what the contribution plot names at first detection
        diag = None
        best = min((v["delay"] for v in det.values()
                    if v["delay"] is not None), default=None)
        if best is not None:
            at = min(onset + best, len(test) - 1)
            dg = bank.pca.diagnose(test[at:at + 1], names, top=3)
            diag = dg[0] if dg else None
        rows.append({"fault": fid, "description": desc.get(fid, ""),
                     "n": int(len(test)), "onset": onset,
                     "detectors": det, "first_diagnosis": diag})

    # A ranking that only holds at one operating point is not a ranking. Every
    # detector achieved 0.00 alarms on the held-out normal run at the 1-per-1000
    # budget, which means the budget was not binding and the thresholds sat at
    # the extreme of the normal statistic. Sweeping it says whether the order
    # survives a control room that will tolerate more nuisance.
    sweep = []
    for budget in (1.0, 5.0, 20.0, 50.0):
        t = calibrate(bank, normal_test, budget)
        na = _alarms(bank, normal_test, t)
        srows = []
        for fid in ids:
            key = f"test_fault_{fid:02d}"
            if key not in z.files:
                continue
            a = _alarms(bank, z[key], t)
            srows.append({"fault": fid,
                          "detectors": {n: _score(v, onset, na[n])
                                        for n, v in a.items()}})
        sweep.append({
            "budget_per_1000": budget,
            "achieved": {n: t[n]["achieved_per_1000"] for n in t},
            "all": summarise(srows),
            "detectable": summarise(srows, exclude=(3, 9, 15)),
            # The hard three on their own. On the detectable faults every
            # detector sits at the m-of-n floor, so that column has no
            # resolution -- the only place these methods can actually differ is
            # where the fault is nearly invisible.
            "hard": summarise(srows, exclude=tuple(
                f for f in ids if f not in (3, 9, 15)))})

    return {"available": True, "rows": rows, "thresholds": thr,
            "fa_budget_per_1000": FA_BUDGET_PER_1000, "sweep": sweep,
            "n_train": int(len(train)), "n_vars": int(train.shape[1]),
            "n_normal_test": int(len(normal_test)),
            "sample_minutes": float(z["sample_minutes"]),
            "hard_faults": [3, 9, 15]}


# ---------------------------------------------------------------------------
# SKAB
# ---------------------------------------------------------------------------

def run_skab(max_runs: int = 12) -> dict:
    p = DATA / "skab.npz"
    if not p.exists():
        return {"available": False, "why": f"no {p}; run fetch_process.py"}
    z = np.load(p, allow_pickle=True)
    normal = z["normal"]
    if len(normal) < 200:
        return {"available": False, "why": "no anomaly-free reference run"}
    names = [str(s) for s in z["names"]]
    srcs = [str(s) for s in z["sources"]]

    rows = []
    for i in range(min(max_runs, len(srcs))):
        x, y = z[f"x_{i}"], z[f"y_{i}"]
        if len(x) < 60 or y.sum() == 0 or y.sum() == len(y):
            continue
        onset = int(np.argmax(y > 0))
        if onset < 10 or onset > len(x) - 10:
            continue
        # Fit on the first half of the anomaly-free run, calibrate on the
        # second. Calibrating on the half the model was fitted to would set
        # every threshold optimistically low.
        half = len(normal) // 2
        bank = _Bank(normal[:half])
        thr = calibrate(bank, normal[half:], FA_BUDGET_PER_1000)
        normal_alarms = _alarms(bank, normal[half:], thr)
        alarms = _alarms(bank, x, thr)
        det = {name: _score(a, onset, normal_alarms[name])
               for name, a in alarms.items()}
        rows.append({"source": srcs[i], "n": int(len(x)), "onset": onset,
                     "anomaly_frac": float(y.mean()), "detectors": det})
    return {"available": True, "rows": rows, "tags": names,
            "n_normal": int(len(normal))}


# ---------------------------------------------------------------------------
# summarising
# ---------------------------------------------------------------------------





def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    print("Tennessee Eastman ...", flush=True)
    te = run_te()
    print("SKAB ...", flush=True)
    sk = run_skab()

    d = {"te": te, "skab": sk, "elapsed_s": time.time() - t0}
    if te.get("available"):
        ids = [r["fault"] for r in te["rows"]]
        d["te_summary_all"] = summarise(te["rows"])
        d["te_summary_detectable"] = summarise(te["rows"], exclude=(3, 9, 15))
        d["te_summary_hard"] = summarise(te["rows"], exclude=tuple(
            f for f in ids if f not in (3, 9, 15)))
    if sk.get("available") and sk["rows"]:
        d["skab_summary"] = summarise(sk["rows"])

    # what the synthetic study concluded, for the comparison
    syn = OUT / "process.json"
    if syn.exists():
        s = json.loads(syn.read_text(encoding="utf-8"))
        d["synthetic"] = {r["fault"]: r["detectors"] for r in s.get("runs", [])}

    (OUT / "real_process.json").write_text(json.dumps(d, indent=2, default=str),
                                           encoding="utf-8")
    (DOCS / "REAL_PROCESS.md").write_text(report(d), encoding="utf-8")
    print(f"wrote docs/REAL_PROCESS.md in {d['elapsed_s']:.0f}s")


SHORT = {
    "univariate 3-sigma (the wall of charts)": "univariate",
    "PCA T2 only": "T²",
    "PCA SPE only": "SPE",
    "T2 or SPE": "T² or SPE",
    "residual (model-based)": "residual",
}


def _delay_cell(v: dict) -> str:
    if v["median_delay"] is None:
        return "never"
    return f"{v['median_delay']:.0f}"


def _sweep_table(sweep: list, kind: str, A) -> None:
    A("| false-alarm budget | " + " | ".join(SHORT[n] for n in DETECTORS) + " |")
    A("|---|" + "---:|" * len(DETECTORS))
    for s in sweep:
        cells = []
        best = min((s[kind][n]["median_delay"] for n in DETECTORS
                    if s[kind][n]["median_delay"] is not None), default=None)
        for n in DETECTORS:
            v = s[kind][n]
            c = _delay_cell(v)
            if v["detected"] < v["of"]:
                c += f" ({v['detected']}/{v['of']})"
            if v["median_delay"] is not None and v["median_delay"] == best:
                c = f"**{c}**"
            cells.append(c)
        A(f"| {s['budget_per_1000']:.0f} per 1000 | " + " | ".join(cells) + " |")


def report(d: dict) -> str:
    L: list[str] = []
    A = L.append
    te, sk = d["te"], d["skab"]

    A("# The process comparison, on data I did not generate\n")
    A("The README named this as the project's remaining circularity and said the "
      "residual model's win was *the result most likely to be an artefact of a "
      "generator built from linear relationships*. It runs here on the Tennessee "
      "Eastman benchmark and on SKAB, with the detectors unchanged — the only new "
      "code is the loading and the calibration.\n")
    A("**Tennessee Eastman is still a simulation**, and calling it real data "
      "would be the overclaim this project keeps catching. What it is: a "
      "simulation *somebody else built*, of a process I did not design, with "
      "faults I did not choose, that the literature has used as its reference for "
      "thirty years. That breaks the circularity — the monitor cannot have been "
      "tuned to a generator I never saw. **SKAB is a real rig**: a water "
      "circulation loop with faults induced by hand.\n")

    if not te.get("available"):
        A(f"\n_TE unavailable: {te.get('why')}_\n")
        return "\n".join(L) + "\n"

    A("\n## The first version of this was wrong, and wrong flatteringly\n")
    A("Setting each detector at its own 99% limit and comparing detection delays "
      "produced this: **all five detectors found all ten faults**, including the "
      "three the literature agrees are close to undetectable, at false-alarm "
      "rates of four to six percent. That is not a comparison of methods, it is "
      "a comparison of thresholds — the loosest detector wins every race and pays "
      "for it in a column the delay table does not show.\n")
    A(f"Everything below sets thresholds so that every detector runs at the "
      f"**same false-alarm budget**, measured as alarm episodes per 1000 samples "
      f"on `d00_te`, a normal run held out from fitting. And because a ranking "
      "that holds at one operating point is not a ranking, the budget is swept.\n")

    A(f"\n## Tennessee Eastman\n")
    A(f"{te['n_vars']} variables, {te['n_train']} normal training samples, "
      f"{te['n_normal_test']} held-out normal samples for calibration, "
      f"{len(te['rows'])} fault runs, fault injected at sample "
      f"{te['rows'][0]['onset']} of {te['rows'][0]['n']} "
      f"({te['sample_minutes']:.0f}-minute sampling).\n")

    A("### Per fault, at 1 false alarm per 1000\n")
    A("| fault | " + " | ".join(SHORT[n] for n in DETECTORS) + " | |")
    A("|---|" + "---:|" * len(DETECTORS) + "---|")
    for r in te["rows"]:
        best = min((v["delay"] for v in r["detectors"].values()
                    if v["delay"] is not None), default=None)
        cells = []
        for n in DETECTORS:
            v = r["detectors"][n]
            if v["delay"] is None:
                cells.append("never")
            elif v["delay"] == best:
                cells.append(f"**{v['delay']}**")
            else:
                cells.append(str(v["delay"]))
        hard = " ⚠" if r["fault"] in te["hard_faults"] else ""
        A(f"| {r['fault']:02d}{hard} | " + " | ".join(cells) + f" | {r['description'][:44]} |")
    A("\n⚠ marks the three faults the literature agrees are close to "
      "undetectable. They are reported rather than dropped: a comparison that "
      "quietly excludes them flatters every method, and *whether a detector "
      "claims to find what nobody finds* is exactly the question dropping them "
      "hides.\n")

    A("\n### The seven detectable faults — no resolution at all\n")
    _sweep_table(te["sweep"], "detectable", A)
    A("\nEvery detector sits at the m-of-n floor. With 3-sample persistence the "
      "fastest possible delay is 2, and at every budget the **univariate "
      "\"wall of charts\" is never worse than anything else**. On this fault "
      "set the multivariate machinery buys nothing: TE's detectable faults are "
      "steps and drifts that push individual measurements clean outside their "
      "normal range, which is precisely the case a per-tag limit was already "
      "good at. The multivariate argument is about faults that break "
      "*correlations* without moving any single tag much, and these are not "
      "those.\n")

    A("\n### The three hard faults — where the methods differ, and the ranking will not sit still\n")
    _sweep_table(te["sweep"], "hard", A)
    A("\n**The ranking flips three times across four budgets.** At 1 per 1000 "
      "the univariate detector is fastest; at 5 it is the slowest of the five "
      "and T² is fastest; at 20 and 50 the residual model is. Nothing about the "
      "data changed — only how much nuisance the operating point tolerates.\n")
    A("That is the finding, and it is a criticism of the synthetic study rather "
      "than a result from it: **the synthetic comparison reported a single "
      "operating point**, and on this evidence a single operating point cannot "
      "support a ranking. The residual model's win there is not refuted so much "
      "as shown to have been unfalsifiable as stated.\n")
    A("The zeros at the loose budgets should be read with suspicion rather than "
      "satisfaction. A delay of 0 on a fault the literature calls undetectable, "
      "bought at 20–50 nuisance alarms per 1000 samples, is a detector alarming "
      "most of the time and being right by coincidence.\n")

    if sk.get("available") and sk.get("rows"):
        s = d.get("skab_summary", {})
        A("\n## SKAB — a real rig, and a null result\n")
        A(f"{len(sk['rows'])} runs, {len(sk['tags'])} tags "
          f"({', '.join(sk['tags'][:4])}…), fitted on the first half of "
          f"{sk['n_normal']} anomaly-free samples and calibrated on the second.\n")
        A("| detector | detected | median delay |")
        A("|---|---:|---:|")
        for n in DETECTORS:
            v = s.get(n)
            if v:
                A(f"| {n} | {v['detected']}/{v['of']} | {_delay_cell(v)} |")
        allzero = all((s.get(n, {}).get("median_delay") or 0) == 0
                      for n in DETECTORS if s.get(n))
        if allzero:
            A("\n**Every detector fires on the first labelled sample, so SKAB "
              "separates nothing.** That is a property of the dataset as used "
              "here, not a compliment to the detectors: the anomalies are "
              "physical interventions on a small test loop — a valve closed by "
              "hand, a rotor unbalanced — and they are large, abrupt, and "
              "already under way at the first sample the label marks. A "
              "benchmark on which everything scores identically is reported as "
              "such rather than as five methods agreeing.\n")

    A("\n## What this does and does not settle\n")
    A("- **It settles the circularity.** The detectors were written against a "
      "generator I built; they now have a score on a process I did not, and the "
      "score does not support what the synthetic study concluded.")
    A("- **It does not make TE real.** It is a simulation with a thirty-year "
      "literature, which is a different and better thing than a simulation with "
      "a README.")
    A("- **The synthetic study is not re-run or retracted here.** Its numbers "
      "stand as measurements of its own generator; what changes is the claim "
      "built on top of them, and `docs/RESULTS.md` now points here.")
    A("- **SKAB was a null.** It is kept because a null that is reported is "
      "worth more than a null that is dropped, and dropping it would leave the "
      "impression that both datasets agreed with TE.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
