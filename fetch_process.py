"""Fetch the real process datasets this project has been measured against the absence of.

WHY. `src/process.py` is an 18-tag process simulator I wrote, and the README has
said so from pass 3 in the strongest terms available:

    The residual model winning is the result most likely to be an artefact of a
    generator built from linear relationships, and I would not defend it until it
    has run on real TE or SKAB data.

That was the right thing to write and the wrong thing to leave. The bearing side
of this project is exactly where real data (CWRU) overturned a claim I was
confident in, and the process side has had no equivalent test.

TENNESSEE EASTMAN. The standard benchmark for multivariate process monitoring —
a simulated chemical plant, but simulated by Downs and Vogel from a real Eastman
process and used as the reference for T2/SPE work for thirty years. 52 variables
(41 measurements XMEAS, 11 manipulated XMV), 21 fault modes, 3-minute sampling.
Fetched from the Braatz group's distribution.

    IT IS STILL A SIMULATION, and calling it "real data" would be the same
    overclaim this project keeps catching. What it is: a simulation somebody
    else built, for a process I did not design, with faults I did not choose,
    that the literature has agreed on for decades. That is enough to break the
    circularity -- my monitor cannot have been tuned to a generator I never saw.

SKAB. The Skoltech Anomaly Benchmark: a REAL water-circulation testbed with
accelerometers, thermocouples, pressure and flow sensors, and hand-labelled
anomalies induced by physically interfering with the rig. Small, and genuinely
physical.

Neither is redistributed here: data/PROCESS/ is gitignored.

    python fetch_process.py
    python fetch_process.py --check
"""
from __future__ import annotations

import io
import pathlib
import sys
import time
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DEST = ROOT / "data" / "PROCESS"

TE_BASE = ("https://raw.githubusercontent.com/camaramm/"
           "tennessee-eastman-profBraatz/master")
SKAB_BASE = "https://raw.githubusercontent.com/waico/SKAB/master/data"

# The faults worth carrying. Not all 21: these span the failure MODES, which is
# what a monitoring comparison turns on, and they include the three the
# literature agrees are close to undetectable.
TE_FAULTS = {
    1: "A/C feed ratio step, B composition constant",
    4: "reactor cooling water inlet temperature step",
    5: "condenser cooling water inlet temperature step",
    6: "A feed loss (step)",
    11: "reactor cooling water inlet temperature, random variation",
    13: "reaction kinetics, slow drift",
    14: "reactor cooling water valve sticking",
    # The hard three. Every published TE comparison scores near zero on these,
    # and a method that claims otherwise is usually scoring its own threshold.
    3: "D feed temperature step (widely reported as near-undetectable)",
    9: "D feed temperature random variation (near-undetectable)",
    15: "condenser cooling water valve sticking (near-undetectable)",
}

# XMEAS 1-41 then XMV 1-11. Names abbreviated from Downs & Vogel table 4/5.
TE_NAMES = [
    "A_feed", "D_feed", "E_feed", "A_C_feed", "recycle_flow", "reactor_feed",
    "reactor_press", "reactor_level", "reactor_temp", "purge_rate",
    "sep_temp", "sep_level", "sep_press", "sep_underflow", "stripper_level",
    "stripper_press", "stripper_underflow", "stripper_temp", "stripper_steam",
    "compressor_work", "reactor_cw_outlet", "cond_cw_outlet",
    # XMEAS 23-28: reactor feed analysis, components A-F only (six, not eight --
    # G and H are not measured in the feed stream). Getting this wrong gives 54
    # names for 52 columns and silently mislabels every contribution plot from
    # the composition block onwards.
] + [f"comp_{k}" for k in "ABCDEF"] + [f"purge_{k}" for k in "ABCDEFGH"] + [
    f"prod_{k}" for k in "DEFGH"
] + [
    "D_feed_valve", "E_feed_valve", "A_feed_valve", "A_C_feed_valve",
    "compressor_valve", "purge_valve", "sep_underflow_valve",
    "stripper_underflow_valve", "stripper_steam_valve", "reactor_cw_valve",
    "cond_cw_valve",
]

# Fault onset, in samples. The Braatz distribution introduces the fault 8 hours
# into the 48-hour test runs at 3-minute sampling; the training runs introduce
# it at sample 20. Getting this wrong makes every detection delay meaningless,
# so it is a named constant rather than a number inside a loop.
TE_TEST_ONSET = 160
TE_TRAIN_ONSET = 20
TE_SAMPLE_MINUTES = 3.0


def _get(url: str, tries: int = 4, timeout: int = 40) -> bytes:
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(0.6 * (a + 1))
    raise RuntimeError(f"{url}: {type(last).__name__}: {last}")


def _parse_dat(blob: bytes, name: str) -> np.ndarray:
    """Whitespace-delimited floats -> (n_samples, 52).

    THE TRANSPOSE TRAP. `d00.dat` ships as 52 rows x 500 columns while every
    other file is n_samples x 52. It is the single most common way to get this
    dataset wrong, it produces an array that is the right dtype and the wrong
    shape, and downstream it looks like a model that cannot fit rather than a
    file that was read sideways. Detected on the shape rather than on the
    filename, so a differently-named copy is handled too.
    """
    a = np.loadtxt(io.BytesIO(blob))
    if a.ndim != 2:
        raise ValueError(f"{name}: expected 2-D, got {a.shape}")
    if a.shape[0] == 52 and a.shape[1] != 52:
        a = a.T
    if a.shape[1] != 52:
        raise ValueError(f"{name}: expected 52 variables, got {a.shape[1]}")
    return np.ascontiguousarray(a, dtype=np.float64)


def fetch_te() -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "te.npz"
    arrays: dict = {}
    wanted = [("d00", "train_normal"), ("d00_te", "test_normal")]
    for f in sorted(TE_FAULTS):
        wanted.append((f"d{f:02d}_te", f"test_fault_{f:02d}"))

    for stem, key in wanted:
        cache = DEST / f"{stem}.dat"
        if cache.exists():
            blob = cache.read_bytes()
        else:
            print(f"  {stem}.dat ...", flush=True)
            blob = _get(f"{TE_BASE}/{stem}.dat")
            cache.write_bytes(blob)
        arrays[key] = _parse_dat(blob, stem)

    np.savez_compressed(
        out, names=np.array(TE_NAMES), test_onset=TE_TEST_ONSET,
        train_onset=TE_TRAIN_ONSET, sample_minutes=TE_SAMPLE_MINUTES,
        fault_ids=np.array(sorted(TE_FAULTS)),
        fault_desc=np.array([TE_FAULTS[f] for f in sorted(TE_FAULTS)]),
        **arrays)
    shapes = {k: v.shape for k, v in arrays.items()}
    print(f"wrote {out} -- {len(arrays)} runs, {shapes['train_normal']} train")
    return {"path": str(out), "runs": len(arrays), "shapes": shapes}


def fetch_skab(n_files: int = 8) -> dict:
    """SKAB: a real rig. Kept small on purpose -- it is a supporting witness
    here, not the main experiment, because its anomalies are physical
    interventions on a test loop rather than process faults."""
    DEST.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    groups = ["valve1", "valve2", "other"]
    frames, labels, srcs = [], [], []
    cols_ref = None
    got = 0
    for g in groups:
        for i in range(n_files):
            rel = f"{g}/{i}.csv"
            cache = DEST / f"skab_{g}_{i}.csv"
            try:
                blob = cache.read_bytes() if cache.exists() else _get(
                    f"{SKAB_BASE}/{rel}")
            except RuntimeError:
                continue
            if not cache.exists():
                cache.write_bytes(blob)
            rows = list(_csv.reader(io.StringIO(blob.decode("utf-8", "replace")),
                                    delimiter=";"))
            if len(rows) < 20:
                continue
            head = rows[0]
            try:
                ai = head.index("anomaly")
            except ValueError:
                continue
            feat = [j for j, h in enumerate(head)
                    if h not in ("datetime", "anomaly", "changepoint")]
            if cols_ref is None:
                cols_ref = [head[j] for j in feat]
            elif [head[j] for j in feat] != cols_ref:
                continue
            X, y = [], []
            for r in rows[1:]:
                if len(r) <= max(feat + [ai]):
                    continue
                try:
                    X.append([float(r[j]) for j in feat])
                    y.append(float(r[ai]))
                except ValueError:
                    continue
            if len(X) < 20:
                continue
            frames.append(np.asarray(X, float))
            labels.append(np.asarray(y, float))
            srcs.append(rel)
            got += 1

    # the anomaly-free reference run, used as the training normal
    free = None
    cache = DEST / "skab_anomaly_free.csv"
    try:
        blob = cache.read_bytes() if cache.exists() else _get(
            f"{SKAB_BASE}/anomaly-free/anomaly-free.csv")
        if not cache.exists():
            cache.write_bytes(blob)
        rows = list(_csv.reader(io.StringIO(blob.decode("utf-8", "replace")),
                                delimiter=";"))
        head = rows[0]
        feat = [j for j, h in enumerate(head)
                if h not in ("datetime", "anomaly", "changepoint")]
        if cols_ref is None:
            cols_ref = [head[j] for j in feat]
        free = np.asarray([[float(r[j]) for j in feat] for r in rows[1:]
                           if len(r) > max(feat)], float)
    except Exception as e:                                    # noqa: BLE001
        print(f"  anomaly-free run unavailable: {e}")

    if not frames:
        print("  SKAB unavailable")
        return {"runs": 0}
    out = DEST / "skab.npz"
    np.savez_compressed(
        out, names=np.array(cols_ref or []),
        normal=free if free is not None else np.zeros((0, len(cols_ref or []))),
        sources=np.array(srcs),
        **{f"x_{i}": f for i, f in enumerate(frames)},
        **{f"y_{i}": l for i, l in enumerate(labels)})
    print(f"wrote {out} -- {got} runs, {len(cols_ref or [])} tags, "
          f"normal {None if free is None else free.shape}")
    return {"path": str(out), "runs": got, "tags": cols_ref}


def main() -> None:
    if "--check" in sys.argv:
        for f in ("te.npz", "skab.npz"):
            p = DEST / f
            if p.exists():
                z = np.load(p, allow_pickle=True)
                print(f"present: {f}, keys {sorted(z.files)[:6]} ...")
            else:
                print(f"not fetched: {f}")
        return
    print("Tennessee Eastman ...", flush=True)
    fetch_te()
    print("SKAB ...", flush=True)
    fetch_skab()


if __name__ == "__main__":
    main()
