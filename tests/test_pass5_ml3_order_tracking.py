"""Pass 5: order tracking, and the speed-varying case.

The tests that matter split into two groups.

CORRECTNESS OF THE RESAMPLER. Angular resampling has several ways to be
silently wrong -- an off-by-one in the phase integration, an order axis scaled
by the wrong factor, a samples-per-rev low enough to alias one fault order onto
another -- and every one of them still produces a plausible spectrum with a
peak in it. The constant-speed identity is the strongest available check: on
constant speed the angle axis is a linear function of the time axis, so a line
at f Hz must land at order f / shaft_hz and nowhere else.

CORRECTNESS OF THE GENERATOR. If the simulator placed defect impulses at a
fixed FREQUENCY while the shaft accelerated, order tracking would have nothing
to recover and the whole study would be measuring its own generator. A test
pins the impulses to angle.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bearing as B              # noqa: E402
import features as F             # noqa: E402
import order_tracking as OT      # noqa: E402

GEOM = B.BearingGeometry()


# --- the speed profile and its phase -----------------------------------------

def test_a_constant_profile_integrates_to_linear_angle():
    sp = OT.sweep_profile(30.0, 30.0, 1.0)
    rev = OT.phase_from_speed(sp)
    assert rev[-1] == pytest.approx(30.0, rel=1e-3)
    # linear: the middle sample is halfway through the revolutions
    assert rev[len(rev) // 2] == pytest.approx(15.0, rel=2e-3)


def test_a_linear_run_up_turns_the_mean_number_of_revolutions():
    sp = OT.sweep_profile(20.0, 40.0, 2.0)
    rev = OT.phase_from_speed(sp)
    assert rev[-1] == pytest.approx(30.0 * 2.0, rel=1e-3)


def test_a_coast_down_is_not_a_reversed_run_up():
    """Exponential decay, so most of the record sits near the final speed. If
    the two shapes were the same one of them would not be implemented."""
    up = OT.sweep_profile(20.0, 40.0, 2.0, kind="linear")
    down = OT.sweep_profile(40.0, 20.0, 2.0, kind="coast")
    assert down[0] == pytest.approx(40.0, rel=0.02)
    assert down[-1] == pytest.approx(20.0, abs=2.0)
    # the coast is BELOW the straight line for most of the record
    line = np.linspace(40.0, 20.0, len(down))
    assert np.mean(down < line) > 0.7
    assert not np.allclose(down, up[::-1], rtol=0.05)


def test_an_unknown_sweep_is_refused():
    with pytest.raises(ValueError, match="unknown sweep"):
        OT.sweep_profile(20.0, 40.0, 1.0, kind="sinusoidal-ish")


# --- the resampler is right --------------------------------------------------

def test_a_tone_lands_at_the_order_it_must_land_at():
    """The identity the whole method rests on. A pure 300 Hz tone on a shaft
    turning at 30 Hz is order 10, exactly."""
    fs, shaft, tone = OT.FS, 30.0, 300.0
    t = np.arange(int(fs * 2)) / fs
    x = np.sin(2 * np.pi * tone * t)
    rev = t * shaft
    y, spr = OT.angular_resample(x, rev, samples_per_rev=128)
    orders, spec = OT.order_spectrum(y, spr)
    assert orders[np.argmax(spec)] == pytest.approx(tone / shaft, rel=0.02)


def test_a_swept_tone_lands_at_a_single_order_too():
    """The point of the exercise: a tone that tracks the shaft is a MOVING
    frequency and a STATIONARY order."""
    fs = OT.FS
    speed = OT.sweep_profile(20.0, 40.0, 2.0)
    rev = OT.phase_from_speed(speed)
    x = np.sin(2 * np.pi * 10.0 * rev)          # order 10, whatever the speed
    y, spr = OT.angular_resample(x, rev, samples_per_rev=128)
    orders, spec = OT.order_spectrum(y, spr)
    assert orders[np.argmax(spec)] == pytest.approx(10.0, rel=0.02)

    # ... and in the FREQUENCY domain the same signal has no single peak worth
    # the name: this is the failure being demonstrated.
    n = len(x)
    fspec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    peak = freqs[np.argmax(fspec)]
    band = fspec[(freqs > 190) & (freqs < 410)]
    assert 190 < peak < 410
    assert float(fspec.max()) < 3.0 * float(np.median(band[band > 0])) * 20


def test_the_order_axis_is_not_secretly_a_frequency_axis():
    """Scale the shaft speed and the order of a shaft-locked line must not
    move. If the axis were mislabelled Hz this test fails."""
    got = []
    for shaft in (20.0, 40.0):
        t = np.arange(int(OT.FS * 1.0)) / OT.FS
        rev = t * shaft
        x = np.sin(2 * np.pi * 7.0 * rev)
        o, s = OT.order_spectrum(*OT.angular_resample(x, rev, 128))
        got.append(o[np.argmax(s)])
    assert got[0] == pytest.approx(got[1], rel=0.02)
    assert got[0] == pytest.approx(7.0, rel=0.02)


def test_a_record_the_shaft_never_turned_is_refused():
    with pytest.raises(ValueError, match="did not turn"):
        OT.angular_resample(np.zeros(100), np.zeros(100))


def test_too_few_revolutions_is_refused_rather_than_returning_junk():
    t = np.arange(200) / OT.FS
    with pytest.raises(ValueError, match="revolutions"):
        OT.angular_resample(np.zeros(200), t * 1.0, samples_per_rev=8)


def test_samples_per_rev_sets_the_order_nyquist():
    t = np.arange(int(OT.FS * 1.0)) / OT.FS
    rev = t * 30.0
    o, _ = OT.order_spectrum(*OT.angular_resample(np.zeros(len(t)), rev, 256))
    assert o[-1] == pytest.approx(128.0, rel=0.01)


# --- the generator is honest -------------------------------------------------

def test_impulses_are_placed_by_angle_not_by_time():
    """If they were placed at a fixed frequency there would be nothing for
    angular resampling to recover, and the entire study would be measuring its
    own generator."""
    rng = np.random.default_rng(3)
    speed = OT.sweep_profile(20.0, 45.0, 2.0)
    rev = OT.phase_from_speed(speed)
    x = OT.simulate_sweep(GEOM, "BPFO", speed, 1.0, rng, noise=0.0,
                          slip_pct=0.0)
    o, s = OT.envelope_order_spectrum(x, rev, (2500.0, 4500.0))
    peak_order = o[(o > 1.0) & (o < 20.0)][
        np.argmax(s[(o > 1.0) & (o < 20.0)])]
    assert peak_order == pytest.approx(GEOM.orders()["BPFO"], rel=0.03)


def test_the_resonance_does_not_sweep_with_the_shaft():
    """It is a property of the structure, not the rotor. Getting this backwards
    would make the run-up case artificially easy for the envelope band."""
    rng = np.random.default_rng(4)
    speed = OT.sweep_profile(20.0, 45.0, 2.0)
    x = OT.simulate_sweep(GEOM, "BPFO", speed, 1.0, rng, noise=0.0,
                          resonance_hz=3500.0, damping=800.0)
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / OT.FS)
    hi = (freqs > 1500) & (freqs < 6000)
    assert freqs[hi][np.argmax(spec[hi])] == pytest.approx(3500.0, rel=0.15)


def test_a_healthy_sweep_has_no_fault_order():
    rng = np.random.default_rng(5)
    speed = OT.sweep_profile(20.0, 45.0, 2.0)
    rev = OT.phase_from_speed(speed)
    x = OT.simulate_sweep(GEOM, None, speed, 0.0, rng)
    o, s = OT.envelope_order_spectrum(x, rev, (2500.0, 4500.0))
    assert OT.call(OT.order_ratios(o, s, GEOM))[0] == "healthy"


# --- the speed tracker -------------------------------------------------------

def test_the_tracker_follows_a_run_up():
    rng = np.random.default_rng(6)
    speed = OT.sweep_profile(22.0, 38.0, 2.0)
    x = OT.simulate_sweep(GEOM, "BPFO", speed, 1.0, rng)
    est = OT.track_speed(x, OT.FS, search=(12.0, 55.0))
    assert np.mean(np.abs(est - speed) / speed) < 0.05


def test_the_tracker_refuses_a_window_off_the_spectrogram():
    with pytest.raises(ValueError, match="search window"):
        OT.track_speed(np.random.default_rng(0).standard_normal(8192),
                       OT.FS, search=(50_000.0, 60_000.0))


# --- scoring -----------------------------------------------------------------

def test_the_two_gates_match_the_ones_diagnose_uses():
    """`call` is a reimplementation, not an import -- features.diagnose wants a
    full snapshot feature dict and half of it would be unfilled here. The
    constants have to stay in step."""
    import inspect
    src = inspect.getsource(F.diagnose)
    assert "min_ratio: float = 6.0" in src
    assert "margin: float = 1.25" in src
    assert OT.call.__defaults__ == (6.0, 1.25)


def test_call_refuses_when_nothing_is_there():
    flat = {k: 1.0 for k in ("BPFO", "BPFI", "BSF2", "FTF", "shaft", "BSF")}
    assert OT.call(flat)[0] == "healthy"


def test_call_says_ambiguous_rather_than_guessing():
    r = {"BPFO": 20.0, "BPFI": 19.0, "BSF2": 1.0, "FTF": 1.0}
    assert OT.call(r)[0] == "ambiguous"


def test_call_maps_bsf2_back_to_a_ball_fault():
    r = {"BPFO": 2.0, "BPFI": 2.0, "BSF2": 40.0, "FTF": 1.0}
    assert OT.call(r)[0] == "BSF"


def test_the_ratio_floor_is_a_median_not_a_mean():
    """A mean floor lets a strong fault inflate its own normaliser and
    understate itself. Same reasoning as features.snapshot_features."""
    import inspect
    assert "np.median" in inspect.getsource(OT._ratio_at)


# --- the constant-speed identity, end to end ---------------------------------

def test_the_two_domains_agree_at_constant_speed():
    """The check the whole study rests on. Same signal, same band, same
    scoring; only the axis differs."""
    rng = np.random.default_rng(7)
    speed = OT.sweep_profile(29.95, 29.95, 2.0)
    x = OT.simulate_sweep(GEOM, "BPFI", speed, 1.0, rng)
    rev = OT.phase_from_speed(speed)
    freqs, spec, _ = F.envelope_spectrum(x, (2500.0, 4500.0), OT.FS)
    fixed = OT.frequency_ratios(freqs, spec, GEOM, 29.95)
    o, s = OT.envelope_order_spectrum(x, rev, (2500.0, 4500.0))
    order = OT.order_ratios(o, s, GEOM)
    assert OT.call(fixed)[0] == OT.call(order)[0] == "BPFI"
    a = np.log1p([fixed[k] for k in ("BPFO", "BPFI", "BSF2", "FTF")])
    b = np.log1p([order[k] for k in ("BPFO", "BPFI", "BSF2", "FTF")])
    assert float(np.corrcoef(a, b)[0, 1]) > 0.9


def test_order_tracking_wins_on_a_wide_sweep():
    """The finding, at the smallest scale that shows it."""
    rng = np.random.default_rng(8)
    speed = OT.sweep_profile(15.0, 45.0, 2.0)
    rev = OT.phase_from_speed(speed)
    x = OT.simulate_sweep(GEOM, "BPFO", speed, 1.0, rng)
    freqs, spec, _ = F.envelope_spectrum(x, (2500.0, 4500.0), OT.FS)
    fixed = OT.frequency_ratios(freqs, spec, GEOM, float(speed.mean()))
    o, s = OT.envelope_order_spectrum(x, rev, (2500.0, 4500.0))
    order = OT.order_ratios(o, s, GEOM)
    assert order["BPFO"] > 5 * fixed["BPFO"]


def test_the_limits_are_stated():
    j = " ".join(OT.LIMITS)
    assert "No real speed-varying data" in j
    assert "tacho phase is EXACT" in j
