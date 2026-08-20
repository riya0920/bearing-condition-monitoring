"""Vibration features: time domain, and the envelope spectrum that actually names faults.

WHY ENVELOPE ANALYSIS AND NOT AN FFT OF THE RAW SIGNAL

A bearing defect impulse is broadband and small. What the accelerometer records is
the machine's structural resonance (a few kHz) being rung by that impulse. So the
raw spectrum shows energy piled up around the resonance, and the fault information
is in the *rate at which the resonance is being re-excited* -- that is, in the
AMPLITUDE MODULATION of a high-frequency carrier, not in any line at BPFO.

Meanwhile the low-frequency end of the raw spectrum is dominated by 1x imbalance
and 2x misalignment, which are 10-100x larger than anything the bearing is doing.
Looking for a BPFO line there is looking for a candle next to a floodlight.

The classical fix (McFadden & Smith 1984, "Vibration monitoring of rolling element
bearings by the high-frequency resonance technique"):

    1. band-pass around the resonance          -> keep only the rung band
    2. take the analytic-signal magnitude       -> the envelope, i.e. the modulation
    3. FFT the envelope                         -> lines at BPFO / BPFI / BSF

Step 1 needs the resonance band. Choosing it by eye is the usual practice and does
not survive a fleet, so `select_band` picks it by SPECTRAL KURTOSIS (Antoni, 2006):
the band whose filtered time signal is most impulsive. That is a defensible,
automatable answer to the question every practitioner is asked -- "how did you pick
the filter band?"

TOLERANCE: rolling elements slip, so the observed line sits ~1-2% off the kinematic
value. Band energy is therefore integrated over a +/-2% window around the computed
frequency and its first two harmonics. Demanding an exact bin match finds nothing.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats

from bearing import FS


def time_domain(x: np.ndarray) -> dict:
    """RMS, kurtosis, crest factor -- and why the order of their response matters.

    RMS is total energy. A single small spall barely moves it: the defect adds a
    brief impulse per revolution to a signal dominated by continuous shaft content.
    RMS is a LATE indicator -- by the time it moves, the damage is distributed.

    KURTOSIS is the 4th standardised moment: it measures how heavy the tails are,
    i.e. how impulsive the signal is. A single spall is exactly an impulsiveness
    change with almost no energy change, so kurtosis moves FIRST. That is the whole
    "kurtosis leads RMS" result, and the lead time it buys is the P-F interval.

    Kurtosis then FALLS again as damage spreads: many defects blur into a
    continuous rough-running signal, which is closer to Gaussian. A monitoring
    system that treats kurtosis as monotonic will report the bearing recovering
    just before it seizes -- which is why the health index in health.py is built
    from band energies and uses kurtosis as an early-warning term, not as the score.
    """
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x**2)))
    return {
        "rms": rms,
        "peak": float(np.max(np.abs(x))),
        "crest_factor": float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0,
        "kurtosis": float(stats.kurtosis(x, fisher=False)),  # 3.0 for Gaussian
        "skewness": float(stats.skew(x)),
        "p2p": float(np.ptp(x)),
    }


def select_band(x: np.ndarray, fs: float = FS, n_bands: int = 16,
                f_min: float = 500.0, n_segments: int = 8) -> tuple[float, float, float]:
    """Pick the demodulation band by spectral kurtosis (Antoni 2006, simplified).

    Split the spectrum into bands, band-pass into each, keep the band whose
    envelope is most impulsive. Returns (low_hz, high_hz, kurtosis).

    Kurtosis is averaged over `n_segments` sub-windows rather than computed once
    over the whole snapshot. The single-shot version has enough variance that a
    noise band frequently beats the true resonance, which then sends the envelope
    analysis to a band with no fault information in it. Averaging over segments is
    closer to what a real kurtogram does, and it is the difference between this
    working and not working: the first version of this function did not do it and
    the resulting diagnosis was indistinguishable from chance.
    """
    nyq = fs / 2
    edges = np.linspace(f_min, nyq * 0.95, n_bands + 1)
    best = (f_min, edges[1], -np.inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        try:
            sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        except ValueError:
            continue
        env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
        seg = np.array_split(env, n_segments)
        k = float(np.mean([stats.kurtosis(sg, fisher=False) for sg in seg]))
        if k > best[2]:
            best = (float(lo), float(hi), k)
    return best


def envelope_spectrum(x: np.ndarray, band: tuple[float, float], fs: float = FS):
    """Band-pass, Hilbert envelope, remove the DC of the envelope, FFT."""
    nyq = fs / 2
    lo, hi = band
    sos = signal.butter(4, [max(lo, 1.0) / nyq, min(hi, nyq * 0.99) / nyq],
                        btype="band", output="sos")
    y = signal.sosfiltfilt(sos, np.asarray(x, dtype=float))
    env = np.abs(signal.hilbert(y))
    env = env - env.mean()  # the DC term would dominate every band-energy ratio
    n = len(env)
    spec = np.abs(np.fft.rfft(env * np.hanning(n))) / n
    freqs = np.fft.rfftfreq(n, 1 / fs)
    return freqs, spec, env


# Harmonic weights. A localised defect produces a periodic impulse train, so its
# envelope spectrum is a comb, and using only the fundamental throws away evidence
# and is fragile to where slip put that one line. The weights decay because the
# higher the harmonic, the more likely it collides with another fault frequency --
# see HARMONIC_COLLISION.
HARMONIC_WEIGHTS = (1.0, 0.6, 0.3)

# For this geometry (SKF 6205-like): BPFO x 3 = 10.754 x shaft, BPFI x 2 = 10.830 x
# shaft. They sit 0.70% apart -- INSIDE the slip tolerance any real detector must
# allow. So the third BPFO harmonic and the second BPFI harmonic are not separable
# by frequency at all, and a diagnosis leaning on harmonic energy will confuse an
# outer-race fault with an inner-race one. The discriminator that does work is
# sidebands: see `sideband_ratio` and `diagnose`.
HARMONIC_COLLISION = "BPFO*3 = 10.754x shaft vs BPFI*2 = 10.830x shaft: 0.70% apart"


def _peak_near(freqs: np.ndarray, spec: np.ndarray, f: float, tol_pct: float) -> float:
    if f <= 0 or f >= freqs[-1]:
        return 0.0
    w = f * tol_pct / 100.0
    m = (freqs >= f - w) & (freqs <= f + w)
    return float(spec[m].max()) if m.any() else 0.0


def _band_energy(freqs: np.ndarray, spec: np.ndarray, f0: float,
                 tol_pct: float = 1.5, harmonics: int = 3) -> float:
    """Weighted envelope-spectrum energy at f0 and its harmonics, within +/-tol_pct."""
    return float(sum(
        w * _peak_near(freqs, spec, f0 * h, tol_pct)
        for h, w in zip(range(1, harmonics + 1), HARMONIC_WEIGHTS[:harmonics])
    ))


def sideband_ratio(freqs: np.ndarray, spec: np.ndarray, f0: float, shaft_hz: float,
                   tol_pct: float = 1.5) -> float:
    """Energy at f0 +/- shaft_hz, relative to energy at f0 itself.

    This is the physical discriminator between inner- and outer-race faults, and it
    is geometry rather than statistics.

    An OUTER race defect is stationary in the load zone: every rolling element
    strikes it with the same force, so the impulse train has constant amplitude and
    the envelope spectrum shows a clean comb at BPFO with no sidebands.

    An INNER race defect rotates with the shaft, carrying the defect in and out of
    the load zone once per revolution. The impulse train is amplitude-modulated at
    shaft rate, and amplitude modulation puts sidebands at BPFI +/- 1x shaft.
    Strong sidebands mean the defect is moving, which means it is on the rotating
    race.

    So the answer to "your envelope spectrum shows energy at 3.58x shaft" is not
    "outer race". It is: 3.58x matches BPFO for this geometry, AND that line has no
    shaft-rate sidebands, AND it is not a harmonic of something lower. What would
    make me wrong: a speed estimate off by 2%, a bearing part number that differs
    from the drawing, or BPFI harmonics landing on the same line -- which for this
    geometry they do, at BPFO x 3.
    """
    center = _peak_near(freqs, spec, f0, tol_pct)
    if center <= 0:
        return 0.0
    side = (_peak_near(freqs, spec, f0 - shaft_hz, tol_pct)
            + _peak_near(freqs, spec, f0 + shaft_hz, tol_pct))
    return float(side / (2 * center))


def estimate_shaft_speed(x: np.ndarray, fs: float = FS,
                         search=(20.0, 40.0)) -> float:
    """Estimate shaft rate from the raw spectrum's dominant low-frequency line.

    Necessary because the fault frequencies are all proportional to shaft speed. A
    2% speed error moves BPFO by 2%, which is the same size as the slip tolerance
    -- so a system that assumes nameplate speed is spending its whole error budget
    before it starts. On a real machine this comes from a tacho or a keyphasor;
    estimating it from the signal is the fallback when neither is wired.
    """
    n = len(x)
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=float) * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    m = (freqs >= search[0]) & (freqs <= search[1])
    return float(freqs[m][np.argmax(spec[m])])


def snapshot_features(x: np.ndarray, geom, fs: float = FS,
                      band: tuple[float, float] | None = None) -> dict:
    """The full per-snapshot feature vector: time domain + physics-located energies."""
    out = time_domain(x)
    shaft_hz = estimate_shaft_speed(x, fs)
    out["shaft_hz_est"] = shaft_hz
    if band is None:
        lo, hi, sk = select_band(x, fs)
        band = (lo, hi)
        out["band_kurtosis"] = sk
    out["band_lo"], out["band_hi"] = band

    freqs, spec, env = envelope_spectrum(x, band, fs)
    out["env_kurtosis"] = float(stats.kurtosis(env, fisher=False))
    out["env_rms"] = float(np.sqrt(np.mean(env**2)))

    ff = geom.fault_frequencies(shaft_hz)
    # MEDIAN, not mean: the mean of the band includes the fault lines themselves,
    # so a strong fault inflates its own normaliser and the ratio understates it.
    broadband = float(np.median(spec[(freqs > 5) & (freqs < 1000)])) + 1e-12
    for name in ("BPFO", "BPFI", "BSF", "FTF", "shaft"):
        e = _band_energy(freqs, spec, ff[name])
        out[f"env_{name}"] = e
        # Ratio to the local broadband floor. The absolute number moves with
        # sensor mounting, gain, and load; the ratio is what transfers between
        # assets, and transferring between assets is the entire point of a fleet
        # health index.
        out[f"env_{name}_ratio"] = e / broadband
    for name in ("BPFO", "BPFI"):
        out[f"sb_{name}"] = sideband_ratio(freqs, spec, ff[name], shaft_hz)
    return out


def diagnose(feats: dict, margin: float = 1.25, sideband_threshold: float = 0.25,
             min_ratio: float = 4.0) -> tuple[str, float]:
    """Name the fault, or refuse to.

    Three gates, in order:

    1. IS ANYTHING THERE? If the best candidate is below `min_ratio` the answer is
       "healthy". A diagnosis engine that always names a race is a random
       part-number generator.
    2. SIDEBANDS FIRST. If the BPFI line carries shaft-rate sidebands above
       `sideband_threshold`, call it inner race regardless of which raw energy is
       larger -- because the BPFO x 3 / BPFI x 2 collision means raw energy is
       precisely the evidence that cannot separate them, and modulation is the
       evidence that can.
    3. MARGIN. Otherwise take the leading candidate only if it beats the runner-up
       by `margin`; below that, return "indeterminate".

    Returning "indeterminate" is a feature. "The outer race is spalled" and
    "something is wrong with this bearing" are different work orders, and the
    second is the honest answer more often than vendors admit.
    """
    cands = {k: feats[f"env_{k}_ratio"] for k in ("BPFO", "BPFI", "BSF")}
    order = sorted(cands.items(), key=lambda kv: -kv[1])
    top, second = order[0], order[1]
    if top[1] < min_ratio:
        return "healthy", top[1]
    if feats.get("sb_BPFI", 0.0) > sideband_threshold and cands["BPFI"] >= min_ratio:
        return "BPFI", feats["sb_BPFI"]
    ratio = top[1] / max(second[1], 1e-9)
    if ratio < margin:
        return "indeterminate", ratio
    return top[0], ratio
