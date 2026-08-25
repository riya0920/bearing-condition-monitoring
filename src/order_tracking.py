"""Pass 5: the speed-varying case, which is where band energy stops working.

Item 9 of the not-built list: no run-up, no coast-down, no order tracking. Every
detector in this project locates energy at a FREQUENCY -- BPFO is 3.585 x shaft,
so at 1772 rpm it is 105.9 Hz and the search window goes there. That reasoning
holds exactly as long as the shaft speed holds. During a run-up it does not, and
the failure is not graceful: the fault line sweeps across the whole search window
and out the other side, so its energy is spread over a band far wider than the
tolerance and the peak the detector is looking for never exists at any single
frequency.

ORDER TRACKING is the standard answer and the idea is small. A defect strikes
once every fixed number of shaft REVOLUTIONS, not once every fixed number of
seconds. Resample the signal onto uniform increments of shaft ANGLE instead of
uniform increments of time and the fault becomes a stationary line again --
located at an ORDER (cycles per revolution) rather than at a frequency. The
geometry already publishes orders: `BearingGeometry.orders()` has been there
since the first pass and nothing used it.

What is measured here, in run_speed_varying.py:

  * a fixed-frequency detector on a run-up, which is the failure
  * the same signal order-tracked, which is the fix
  * both on a CONSTANT-speed real CWRU record, where they must agree -- an
    order-tracking implementation that disagrees with the frequency method on
    constant speed is not a better method, it is a broken one, and this is the
    only check available that can tell the two apart

Two phase sources are provided because they fail differently:

  TACHO       exact, and what a real installation with a keyphasor gives you
  ESTIMATED   the 1x line tracked through a spectrogram and integrated, for the
              installations that have no tacho -- which is most of them, and is
              the situation the rest of this project is written for
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

FS = 20_000


# ---------------------------------------------------------------------------
# a machine whose speed is changing
# ---------------------------------------------------------------------------

def sweep_profile(f0: float, f1: float, duration_s: float, fs: float = FS,
                  kind: str = "linear") -> np.ndarray:
    """Instantaneous shaft rate, Hz, sample by sample.

    `coast` is not a reversed run-up. A machine coasting down is losing energy
    to friction, so its speed decays roughly exponentially rather than linearly,
    and the fast part is at the start. It matters here because the two shapes
    put the interesting part of the sweep in different places relative to the
    analysis window.
    """
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    if kind == "linear":
        return f0 + (f1 - f0) * (t / duration_s)
    if kind == "coast":
        tau = duration_s / 3.0
        return f1 + (f0 - f1) * np.exp(-t / tau)
    raise ValueError(f"unknown sweep {kind!r}")


def phase_from_speed(speed_hz: np.ndarray, fs: float = FS) -> np.ndarray:
    """Shaft angle in REVOLUTIONS, by integrating instantaneous speed.

    Cumulative trapezoid rather than cumsum: with speed changing across a
    sample, the rectangle rule accumulates a bias that grows with the record,
    and the whole method depends on the phase being right at the END of the
    record as much as at the start.
    """
    s = np.asarray(speed_hz, dtype=float)
    d = np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) / 2.0) / fs])
    return d


def simulate_sweep(geom, fault: str | None, speed_hz: np.ndarray,
                   severity: float, rng: np.random.Generator,
                   fs: float = FS, noise: float = 0.35,
                   resonance_hz: float = 3500.0, damping: float = 800.0,
                   slip_pct: float = 1.5) -> np.ndarray:
    """A run-up or coast-down with a bearing defect in it.

    The defect impulses are placed by ANGLE, not by time: one strike every
    `1 / order` revolutions. That is the physics, and it is also the whole
    reason order tracking works -- if impulses were placed at a fixed frequency
    while the shaft accelerated, there would be nothing for angular resampling
    to recover and this file would be measuring its own generator.

    The shaft-rate content (1x imbalance, 2x misalignment) sweeps with the
    shaft, so a fixed-frequency detector loses those too. The RESONANCE does
    NOT sweep: it is a property of the structure, not of the rotor, and it stays
    where it is while everything else moves past it. Getting that backwards
    would make the run-up case artificially easy for the envelope band.
    """
    speed = np.asarray(speed_hz, dtype=float)
    n = len(speed)
    rev = phase_from_speed(speed, fs)          # revolutions elapsed

    x = (1.0 * np.sin(2 * np.pi * rev + rng.uniform(0, 6.28))
         + 0.4 * np.sin(2 * np.pi * 2 * rev + rng.uniform(0, 6.28)))
    x += noise * rng.standard_normal(n)

    if fault and severity > 0:
        order = geom.orders()[fault]
        amp = 2.0 * severity
        ring_len = int(fs / damping * 8)
        tt = np.arange(ring_len) / fs
        ring = np.exp(-damping * tt) * np.sin(2 * np.pi * resonance_hz * tt)
        # Strike whenever the accumulated angle passes the next multiple of
        # 1/order revolutions. Slip jitters the ANGULAR interval, which is what
        # slip physically is -- the rolling element creeps, so it arrives a
        # little early or late in ANGLE.
        step = 1.0 / order
        target = step
        env_mod = None
        if fault == "BPFI":
            env_mod = 1 + 0.5 * severity * np.sin(2 * np.pi * rev)
        while target < rev[-1]:
            idx = int(np.searchsorted(rev, target))
            if idx >= n:
                break
            L = min(n - idx, ring_len)
            a = amp * (1 + 0.15 * rng.standard_normal())
            x[idx:idx + L] += a * ring[:L]
            target += step * (1 + slip_pct / 100.0 * rng.standard_normal())
        if env_mod is not None:
            x *= env_mod
    return x


# ---------------------------------------------------------------------------
# recovering the phase when there is no tacho
# ---------------------------------------------------------------------------

def track_speed(x: np.ndarray, fs: float = FS, search=(15.0, 70.0),
                nperseg: int = 4096, overlap: float = 0.75) -> np.ndarray:
    """Follow the 1x line through a spectrogram and interpolate it per sample.

    Deliberately the simple version: strongest bin inside a speed search window,
    per frame. It works here because the 1x imbalance line is the dominant
    low-frequency content, which is true of most rotating machines and is not
    true of all of them. A gearbox with a strong mesh order, or a machine whose
    imbalance is small next to its blade-pass, needs a proper phase-locked
    tracker; that is a limit, and it is stated rather than papered over.

    The frames are smoothed with a short median filter before interpolation. One
    bad frame in a run-up is a step change in estimated speed, and a step in
    speed integrates into a permanent phase OFFSET -- every impulse after it
    lands at the wrong angle, so a single outlier corrupts the whole rest of the
    record rather than one frame of it.
    """
    x = np.asarray(x, dtype=float)
    nperseg = min(nperseg, len(x))
    f, t, S = sig.spectrogram(x, fs=fs, nperseg=nperseg,
                              noverlap=int(nperseg * overlap),
                              scaling="spectrum", mode="magnitude")
    m = (f >= search[0]) & (f <= search[1])
    if not m.any():
        raise ValueError("speed search window falls outside the spectrogram")
    peak = f[m][np.argmax(S[m], axis=0)]
    k = min(5, len(peak) if len(peak) % 2 else len(peak) - 1)
    if k >= 3:
        peak = sig.medfilt(peak, kernel_size=k)
    ts = np.arange(len(x)) / fs
    return np.interp(ts, t, peak)


# ---------------------------------------------------------------------------
# angular resampling
# ---------------------------------------------------------------------------

def angular_resample(x: np.ndarray, rev: np.ndarray, samples_per_rev: int = 256):
    """Resample onto uniform shaft angle.

    Returns (y, samples_per_rev). `y[k]` is the signal at angle
    `k / samples_per_rev` revolutions, so an FFT of `y` has ORDERS on its axis:
    bin j is j cycles per revolution scaled by the record length in revolutions.

    samples_per_rev sets the order Nyquist at samples_per_rev / 2. The default
    256 puts it at order 128, comfortably above BPFI (5.4) and its harmonics and
    above the envelope content this project cares about. Setting it too low is
    the quiet failure: fault orders alias down onto other fault orders, and the
    result is a confident wrong diagnosis rather than a missing one.
    """
    rev = np.asarray(rev, dtype=float)
    total = float(rev[-1] - rev[0])
    if total <= 0:
        raise ValueError("the shaft did not turn: no angle to resample onto")
    n_out = int(total * samples_per_rev)
    if n_out < 8:
        raise ValueError(f"only {total:.2f} revolutions in this record")
    grid = rev[0] + np.arange(n_out) / samples_per_rev
    # np.interp needs an increasing x. Shaft angle is monotone as long as the
    # machine does not reverse, which no bearing rig in this project does; a
    # reversing drive would need the phase unwrapped differently and is a limit.
    y = np.interp(grid, rev, np.asarray(x, dtype=float))
    return y, samples_per_rev


def order_spectrum(y: np.ndarray, samples_per_rev: int):
    """FFT of an angle-domain signal. X axis is ORDERS, not Hz."""
    n = len(y)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * w)) * 2.0 / np.sum(w)
    orders = np.fft.rfftfreq(n, d=1.0 / samples_per_rev)
    return orders, spec


def envelope_order_spectrum(x: np.ndarray, rev: np.ndarray,
                            band: tuple[float, float], fs: float = FS,
                            samples_per_rev: int = 256):
    """Band-pass and envelope in TIME, then resample the envelope in ANGLE.

    The order of operations is the one thing here that is easy to get wrong and
    expensive to get wrong. The resonance band is a property of the STRUCTURE,
    fixed in Hz, so the band-pass has to happen while the signal is still on a
    time axis -- resampling first would smear a fixed-frequency band across
    orders and the filter would then select the wrong content at every speed.
    The ENVELOPE, by contrast, is the impulse train, which is fixed in ANGLE.
    So: filter in time, envelope in time, resample the envelope in angle.
    """
    lo, hi = band
    nyq = fs / 2.0
    lo = max(lo, 1.0)
    hi = min(hi, nyq * 0.99)
    if hi <= lo:
        raise ValueError(f"empty band {band}")
    b, a = sig.butter(4, [lo / nyq, hi / nyq], btype="band")
    env = np.abs(sig.hilbert(sig.filtfilt(b, a, np.asarray(x, dtype=float))))
    env = env - env.mean()
    y, spr = angular_resample(env, rev, samples_per_rev)
    return order_spectrum(y, spr)


# ---------------------------------------------------------------------------
# scoring, in either domain
# ---------------------------------------------------------------------------

def _ratio_at(axis: np.ndarray, spec: np.ndarray, target: float,
              tol_pct: float, floor_lo: float, floor_hi: float,
              harmonics=(1.0, 0.6, 0.3)) -> float:
    """Weighted harmonic energy at `target`, over the local broadband floor.

    Same shape as features.snapshot_features so the numbers are comparable:
    weighted harmonics, MEDIAN floor (a mean would let a strong fault inflate
    its own normaliser).
    """
    floor = float(np.median(spec[(axis > floor_lo) & (axis < floor_hi)])) + 1e-12
    tot = 0.0
    for h, w in enumerate(harmonics, start=1):
        f = target * h
        if f <= 0 or f >= axis[-1]:
            continue
        tol = f * tol_pct / 100.0
        m = (axis >= f - tol) & (axis <= f + tol)
        if m.any():
            tot += w * float(spec[m].max())
    return tot / floor


def order_ratios(orders: np.ndarray, spec: np.ndarray, geom,
                 tol_pct: float = 2.0) -> dict:
    """Fault-order ratios. The speed-independent form of the band ratios."""
    o = geom.orders()
    o["BSF2"] = 2.0 * o["BSF"]
    return {k: _ratio_at(orders, spec, o[k], tol_pct, 0.5, 40.0)
            for k in ("BPFO", "BPFI", "BSF", "BSF2", "FTF", "shaft")}


def frequency_ratios(freqs: np.ndarray, spec: np.ndarray, geom,
                     shaft_hz: float, tol_pct: float = 2.0) -> dict:
    """The same thing the rest of the project does: ratios at fixed frequencies.

    `shaft_hz` is a single number, which is precisely the assumption that fails
    on a run-up. Included here so the comparison uses one code path and the
    difference between the two columns is the DOMAIN, not two different scoring
    functions written months apart.
    """
    ff = geom.fault_frequencies(shaft_hz)
    ff["BSF2"] = 2.0 * ff["BSF"]
    return {k: _ratio_at(freqs, spec, ff[k], tol_pct, 5.0, 1000.0)
            for k in ("BPFO", "BPFI", "BSF", "BSF2", "FTF", "shaft")}


def call(ratios: dict, min_ratio: float = 6.0, margin: float = 1.25) -> tuple:
    """Name the fault or refuse, on the same two gates diagnose() uses.

    Not an import from features.diagnose: that function takes the full snapshot
    feature dict including sidebands, and half of it would be unfilled here. The
    two gates that matter for this comparison -- is anything there, and is the
    best candidate clear of the runner-up -- are reproduced, and a test pins
    them to the same constants.
    """
    cands = {k: v for k, v in ratios.items() if k in
             ("BPFO", "BPFI", "BSF2", "FTF")}
    best = max(cands, key=cands.get)
    top = cands[best]
    rest = sorted((v for k, v in cands.items() if k != best), reverse=True)
    runner = rest[0] if rest else 0.0
    if top < min_ratio:
        return "healthy", top
    if runner > 0 and top < margin * runner:
        return "ambiguous", top
    return {"BSF2": "BSF"}.get(best, best), top


LIMITS = (
    "The tacho phase is EXACT here because the generator knows the speed "
    "profile. A real keyphasor gives one pulse per revolution and the phase "
    "between pulses is interpolated, which adds error this study does not "
    "have.",
    "track_speed follows the strongest line in a speed window. On a machine "
    "whose 1x imbalance is not the dominant low-frequency content -- a gearbox "
    "with a strong mesh order, a fan with heavy blade-pass -- it will lock onto "
    "the wrong line and every angle after that is wrong.",
    "No real speed-varying data. CWRU runs at constant speed, so the run-up is "
    "simulated and only the CONSTANT-speed agreement check uses real "
    "measurements. A run-up rig record would test both halves at once.",
    "Angle is assumed monotone: no reversing drives.",
)
