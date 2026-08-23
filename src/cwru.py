"""Loader for the CWRU bearing dataset, and the reason it is the harder test.

WHAT CHANGES WHEN THE DATA IS REAL. Every number in RESULTS.md came from
`src/bearing.py`, a simulator I wrote. That makes the whole project circular in
one specific way: the envelope pipeline was designed against a signal model, and
then validated against the same signal model. It could not have failed. A
simulator cannot falsify the physics it was written from.

CWRU breaks the circle. Same bearing geometry (SKF 6205-2RS, which is why
`BearingGeometry`'s defaults were chosen), real accelerometers, real spark-eroded
faults on a known race at a known size, under a known load.

THE FOUR THINGS THAT ARE GENUINELY DIFFERENT, and each is a way the pipeline
could break:

  1. SAMPLING RATE. 12 kHz, not the simulator's 20 kHz. The demodulation band has
     to move: Nyquist is 6 kHz, so the 4-6 kHz resonance band the simulator uses
     is at the very top of the usable range and partially aliased.

  2. SHAFT SPEED IS NOT NOMINAL. The files are labelled 1797/1772/1750/1730 rpm,
     but the actual speed sags with load and varies within a file. Fault
     frequencies scale with shaft speed, so using the nameplate rpm puts the
     search window in the wrong place. Each file carries a measured RPM value and
     it is used.

  3. THE FAULTS ARE SEEDED, NOT GROWN. A spark-eroded pit is a clean geometric
     defect. Natural spalling is rough, spreads, and generates a messier
     signature. CWRU is therefore an EASIER diagnosis problem than a real failing
     bearing, in the one dimension that matters most -- which is worth saying
     before quoting a high accuracy from it.

  4. THERE IS NO DEGRADATION TRAJECTORY. Each file is a bearing at one fault
     size. Three sizes exist (0.007/0.014/0.021 in), and they are three different
     bearings rather than one bearing photographed three times. So CWRU can test
     DIAGNOSIS -- which race is faulty -- and cannot test PROGNOSIS. The lead-time
     and health-index results in RESULTS.md stay simulated, and no CWRU number
     here should be read as validating them.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not retune anything. The band
selection, the exceedance logic, the sideband tie-break and the diagnosis
thresholds are used exactly as `src/features.py` defines them. A pipeline that
has to be retuned per dataset has not been validated by the second dataset.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "CWRU"
FS_CWRU = 12_000


def available() -> bool:
    return DATA.exists() and len(list(DATA.glob("*.mat"))) > 0


def manifest() -> dict:
    p = DATA / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"files": {}}


def load_file(fid: str) -> dict:
    """Load one .mat, returning the drive-end channel and the measured RPM.

    The variable names are positional-by-convention rather than fixed -- a file
    for record 105 holds `X105_DE_time`, `X105_FE_time` and sometimes `X105RPM`.
    Matching by suffix rather than by exact name is what makes the loader survive
    the handful of files that number their variables inconsistently, which several
    of the CWRU files genuinely do.
    """
    import scipy.io

    m = scipy.io.loadmat(str(DATA / f"{fid}.mat"))
    de = fe = rpm = None
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if k.endswith("_DE_time"):
            de = np.asarray(v, dtype=float).ravel()
        elif k.endswith("_FE_time"):
            fe = np.asarray(v, dtype=float).ravel()
        elif k.endswith("RPM"):
            rpm = float(np.asarray(v).ravel()[0])
    if de is None:
        raise ValueError(f"{fid}.mat has no drive-end channel")
    return {"fid": fid, "de": de, "fe": fe, "rpm": rpm, "fs": FS_CWRU}


def snapshots(x: np.ndarray, n: int = 20, length: int = 12_000,
              seed: int = 0) -> list[np.ndarray]:
    """Cut a long record into non-overlapping snapshots.

    Non-overlapping matters. Overlapping windows are the standard way CWRU
    accuracies get inflated past 99% in the literature: neighbouring windows share
    samples, so a random train/test split puts near-duplicate segments on both
    sides and the model is scored on data it has effectively seen. Nothing here is
    trained, so the risk is smaller -- but the same discipline keeps the snapshot
    count honest, since 20 overlapping views of one second of data are not 20
    independent measurements.
    """
    length = int(length)
    total = len(x) // length
    if total == 0:
        return [x]
    take = min(n, total)
    rng = np.random.default_rng(seed)
    starts = rng.choice(total, size=take, replace=False) * length
    return [x[s:s + length] for s in sorted(starts)]


def shaft_hz(rec: dict, fallback_rpm: float | None = None) -> float:
    """Measured shaft speed in Hz, preferring the value recorded in the file.

    Using the nameplate rpm instead is a real error with a visible consequence:
    at 1750 rpm nominal against 1772 actual, BPFI sits 1.3% away from where the
    search window expects it -- inside the slip tolerance, so it does not break
    outright, it just quietly costs peak energy and makes every exceedance
    smaller.
    """
    rpm = rec.get("rpm") or fallback_rpm
    if not rpm or not np.isfinite(rpm) or rpm <= 0:
        raise ValueError(f"{rec['fid']}: no usable RPM")
    return float(rpm) / 60.0


# ---------------------------------------------------------------------------
# label mapping
# ---------------------------------------------------------------------------

# CWRU's fault names map onto the project's frequency names. `ball` maps to BSF:
# a defect on a rolling element strikes both races once per element rotation, and
# BSF is the ball-spin frequency the project already computes.
FAULT_TO_FREQ = {"inner_race": "BPFI", "outer_race": "BPFO",
                 "ball": "BSF", "normal": None}


def expected_frequency(fault: str) -> str | None:
    return FAULT_TO_FREQ.get(fault)


def band_for_fs(fs: float) -> tuple[float, float]:
    """The demodulation band, scaled to the available bandwidth.

    The simulator runs at 20 kHz and the housing resonance it excites sits around
    4-6 kHz. CWRU samples at 12 kHz, so Nyquist is 6 kHz and that band is right at
    the edge. Scaling the band with Nyquist keeps the *relative* position of the
    demodulation window the same rather than silently pushing it into the
    anti-alias roll-off.

    This is the one thing that had to change for real data, and it is a
    consequence of the instrument rather than a tuning knob: the same physical
    resonance is simply not fully observable at 12 kHz.
    """
    nyq = fs / 2.0
    return (0.45 * nyq, 0.90 * nyq)
