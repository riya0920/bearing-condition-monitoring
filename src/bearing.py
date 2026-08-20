"""Bearing physics: fault frequencies from geometry, and a run-to-failure simulator.

The whole differentiator of this project is that the features come from bearing
kinematics rather than from a feature-selection routine. A rolling-element bearing
with a localised defect produces an impulse every time a rolling element passes the
defect, and the rate of that impulse train is fixed by the geometry:

    BPFO = (n/2) * f_r * (1 - (d/D) cos(phi))     outer race defect
    BPFI = (n/2) * f_r * (1 + (d/D) cos(phi))     inner race defect
    BSF  = (D/(2d)) * f_r * (1 - ((d/D) cos(phi))^2)   rolling element (ball)
    FTF  = (1/2) * f_r * (1 - (d/D) cos(phi))     cage

    n   = number of rolling elements
    d   = rolling element diameter
    D   = pitch diameter
    phi = contact angle
    f_r = shaft rotation frequency (Hz)

Reading these off a spectrum is the difference between "there is an anomaly" and
"the outer race is spalled, order the part". Note what they are NOT: harmonics of
shaft speed. BPFO is typically a non-integer multiple (3-8x) of shaft rate, which
is why it does not collide with imbalance/misalignment lines at 1x, 2x, 3x.

The catch that makes this a diagnosis rather than a lookup: these are KINEMATIC
frequencies assuming pure rolling. Real bearings slip, so the actual line sits
1-2% off the computed value and smears. A detector that demands an exact match
finds nothing.

DATA HONESTY: the vibration here is SYNTHESISED by `simulate_run_to_failure`,
not measured. It is a physically-structured simulation -- impulse trains at the
kinematic frequencies, exponentially decaying resonance ringing, speed jitter,
slip, and a degradation trajectory -- built so the ground truth (fault type,
onset cycle, failure cycle) is known and every claim is scoreable. It is NOT
CWRU or IMS data and no number here should be compared to a paper that uses them.
See README "what is NOT built".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 20_000  # sampling rate, Hz -- typical for a piezo accelerometer channel


@dataclass(frozen=True)
class BearingGeometry:
    """Defaults approximate a SKF 6205-2RS deep-groove ball bearing, the size used
    in most published bearing test rigs. The numbers are geometry, not calibration."""
    n_elements: int = 9
    ball_diameter_mm: float = 7.94
    pitch_diameter_mm: float = 39.04
    contact_angle_deg: float = 0.0

    @property
    def ratio(self) -> float:
        return (self.ball_diameter_mm / self.pitch_diameter_mm) * np.cos(
            np.deg2rad(self.contact_angle_deg)
        )

    def fault_frequencies(self, shaft_hz: float) -> dict[str, float]:
        n, r = self.n_elements, self.ratio
        return {
            "BPFO": (n / 2) * shaft_hz * (1 - r),
            "BPFI": (n / 2) * shaft_hz * (1 + r),
            "BSF": (self.pitch_diameter_mm / (2 * self.ball_diameter_mm))
            * shaft_hz * (1 - r**2),
            "FTF": 0.5 * shaft_hz * (1 - r),
            "shaft": shaft_hz,
        }

    def orders(self) -> dict[str, float]:
        """Fault frequencies as MULTIPLES of shaft speed -- the speed-independent
        form, and the one to quote when the machine is not running at a constant
        speed."""
        f = self.fault_frequencies(1.0)
        return {k: v for k, v in f.items()}


def _impulse_train(n_samples: int, fs: float, freq: float, rng: np.random.Generator,
                   amplitude: float, resonance_hz: float = 3500.0,
                   damping: float = 800.0, slip_pct: float = 1.5) -> np.ndarray:
    """Impulses at `freq`, each exciting a decaying structural resonance.

    Two pieces of realism that matter for whether envelope analysis works at all:
      * SLIP -- each inter-impulse interval is jittered by a small random
        percentage, which smears the spectral line. This is why raw-spectrum
        peak-picking at BPFO underperforms envelope analysis.
      * RESONANCE -- the defect impulse is broadband; what the accelerometer sees
        is the machine's structural resonance rung by it. The fault information is
        in the ENVELOPE of that high-frequency ringing, not in its carrier. That
        fact is the entire justification for the envelope method.
    """
    x = np.zeros(n_samples)
    if freq <= 0 or amplitude <= 0:
        return x
    period = fs / freq
    t = 0.0
    while t < n_samples:
        idx = int(t)
        if idx >= n_samples:
            break
        length = min(n_samples - idx, int(fs / damping * 8))
        tt = np.arange(length) / fs
        ring = np.exp(-damping * tt) * np.sin(2 * np.pi * resonance_hz * tt)
        x[idx : idx + length] += amplitude * (1 + 0.15 * rng.standard_normal()) * ring
        t += period * (1 + slip_pct / 100.0 * rng.standard_normal())
    return x


def simulate_snapshot(geom: BearingGeometry, shaft_hz: float, fault: str | None,
                      severity: float, rng: np.random.Generator,
                      duration_s: float = 0.5, noise: float = 0.35) -> np.ndarray:
    """One acquisition snapshot: shaft-rate content + defect impulses + noise."""
    n = int(FS * duration_s)
    t = np.arange(n) / FS
    freqs = geom.fault_frequencies(shaft_hz)

    # Always-present machine content: 1x imbalance, 2x misalignment, and a bit of
    # broadband. A healthy bearing is not a silent bearing, and a detector tuned on
    # silence fails on the first real machine.
    x = (1.0 * np.sin(2 * np.pi * shaft_hz * t + rng.uniform(0, 6.28))
         + 0.4 * np.sin(2 * np.pi * 2 * shaft_hz * t + rng.uniform(0, 6.28)))
    x += noise * rng.standard_normal(n)

    if fault and severity > 0:
        amp = 2.0 * severity
        x += _impulse_train(n, FS, freqs[fault], rng, amplitude=amp)
        if fault == "BPFI":
            # Inner-race defects rotate through the load zone, so their impulse
            # amplitude is modulated at shaft rate -- this is what produces the
            # +/- shaft-rate SIDEBANDS around BPFI that distinguish an inner-race
            # fault from an outer-race one on a spectrum.
            x *= 1 + 0.5 * severity * np.sin(2 * np.pi * shaft_hz * t)
    return x


def simulate_run_to_failure(geom: BearingGeometry, fault: str, n_cycles: int = 400,
                            onset: int = 200, shaft_hz: float = 29.95,
                            seed: int = 0, speed_jitter_pct: float = 1.0):
    """A degrading bearing, one snapshot per cycle.

    Degradation is flat until `onset`, then grows on an accelerating curve. That
    shape -- long quiet, then a knee -- is what the P-F interval describes and what
    makes lead time finite: there is nothing to detect before the knee.

    Shaft speed jitters cycle to cycle, so anything that hard-codes a frequency in
    Hz rather than estimating speed will drift off the fault line. Speed estimation
    is part of the job.
    """
    rng = np.random.default_rng(seed)
    snapshots, severities, speeds = [], [], []
    for c in range(n_cycles):
        if c < onset:
            sev = 0.0
        else:
            frac = (c - onset) / max(1, (n_cycles - onset))
            sev = frac**2.2  # accelerating: the knee of the P-F curve
        hz = shaft_hz * (1 + speed_jitter_pct / 100.0 * rng.standard_normal())
        snapshots.append(simulate_snapshot(geom, hz, fault, sev, rng))
        severities.append(sev)
        speeds.append(hz)
    return (np.asarray(snapshots), np.asarray(severities), np.asarray(speeds),
            {"fault": fault, "onset_cycle": onset, "failure_cycle": n_cycles - 1})


def simulate_healthy(geom: BearingGeometry, n_cycles: int = 400, shaft_hz: float = 29.95,
                     seed: int = 0, speed_jitter_pct: float = 1.0):
    """A bearing that never fails. Needed to count false alarms honestly: a
    false-alarm rate measured only on the healthy PREFIX of failing assets is
    measured on assets that are about to fail, which is not the fleet."""
    rng = np.random.default_rng(seed)
    snaps, speeds = [], []
    for _ in range(n_cycles):
        hz = shaft_hz * (1 + speed_jitter_pct / 100.0 * rng.standard_normal())
        snaps.append(simulate_snapshot(geom, hz, None, 0.0, rng))
        speeds.append(hz)
    return np.asarray(snaps), np.asarray(speeds)
