"""A Tennessee-Eastman-style process, and residual-based (model-based) features.

WHY THE PROCESS SIDE IS A DIFFERENT PROBLEM FROM THE BEARING SIDE, and why the
spec asks for both. A bearing gives you one signal with physics in its FREQUENCY
content: the fault frequencies come from geometry and you go and look at them. A
chemical process gives you fifty signals with physics in their RELATIONSHIPS:
reactor pressure is a function of temperature and feed rate, level is the
integral of in minus out, and no single tag tells you anything.

That difference changes what a detector has to do. On a bearing, feature
engineering IS the physics. On a process, the physics is a set of CONSTRAINTS
between variables, and a fault is a violation of a constraint -- which is
invisible in every individual variable.

THE FAILURE MODE THIS EXISTS TO DEMONSTRATE. Every variable can sit comfortably
inside its own control limits while the process is in a state it has never been
in. Reactor temperature normal. Feed rate normal. Cooling-water flow normal. But
that temperature at that feed rate with that cooling flow is a combination that
has never occurred, because it means the heat exchanger has fouled. A wall of
univariate charts shows fifty green lines, and the plant is drifting toward a
shutdown.

THE SIMULATED PROCESS. Not the real Downs-and-Vinnel TE model -- that is a
FORTRAN plant model with 50 states and it is not reimplementable here honestly.
This is a TE-STYLE process: a reactor / separator / stripper loop with the
structure that matters for monitoring, which is a set of variables coupled by
physical relationships plus a recycle that propagates a disturbance around the
loop. It is named as a style rather than the thing, everywhere.

FAULTS, chosen so the T2/SPE distinction has something to bite on:

  IDV-1-style  a STEP in feed composition. The process moves to a new operating
               point along its normal correlation structure -- relationships
               hold, magnitudes shift. This is a T2 fault.

  IDV-4-style  heat-exchanger FOULING, a slow drift. The relationship between
               cooling flow and reactor temperature changes: the same flow no
               longer removes the same heat. Magnitudes can stay in range while
               the CORRELATION breaks. This is an SPE fault, and it is the one
               univariate charts cannot see.

  IDV-13-style a slow drift in reaction kinetics, which moves both.

  valve stick  a control valve that stops responding. The controller compensates
               elsewhere, so the loop looks stable and the relationships are
               wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The measured tags, in the order the matrices use them everywhere.
TAGS = [
    "feed_a_flow", "feed_d_flow", "feed_e_flow", "recycle_flow",
    "reactor_temp", "reactor_press", "reactor_level",
    "sep_temp", "sep_level", "sep_underflow",
    "stripper_level", "stripper_temp", "stripper_flow",
    "cool_water_flow", "cool_water_outlet_temp", "compressor_work",
    "purge_rate", "product_rate",
]

FAULTS = {
    "none": "no disturbance",
    "feed_composition_step": ("IDV-1 style: a step in feed A composition. The "
                              "process moves to a new operating point ALONG its "
                              "normal correlation structure"),
    "heat_exchanger_fouling": ("IDV-4 style: cooling capacity degrades slowly. "
                               "The COOLING-FLOW-to-TEMPERATURE relationship "
                               "changes while both stay in range"),
    "kinetics_drift": ("IDV-13 style: slow drift in reaction kinetics; moves "
                       "magnitudes and relationships together"),
    "valve_stick": ("a control valve stops responding; the controller "
                    "compensates elsewhere, so the loop looks stable and the "
                    "relationships are wrong"),
}


@dataclass
class ProcessRun:
    x: np.ndarray                  # (n_samples, n_tags)
    fault: str
    fault_start: int               # sample index where the disturbance begins
    tags: list = field(default_factory=lambda: list(TAGS))


def simulate(n: int = 2000, fault: str = "none", fault_start: int | None = None,
             severity: float = 1.0, seed: int = 0) -> ProcessRun:
    """Generate a run. Relationships are enforced, not sampled independently.

    The construction matters: every derived variable is computed FROM its drivers
    plus measurement noise, so the covariance structure is a consequence of the
    process rather than something imposed on it. A generator that draws each tag
    from its own distribution and then correlates them post hoc cannot produce a
    fault that breaks a relationship while leaving magnitudes intact -- which is
    the entire case this module exists to make.
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {sorted(FAULTS)}")
    rng = np.random.default_rng(seed)
    fs = n // 3 if fault_start is None else int(fault_start)
    t = np.arange(n, dtype=float)
    active = (t >= fs).astype(float)
    ramp = np.clip((t - fs) / max(n - fs, 1), 0, 1) * active

    # --- independent drivers -------------------------------------------
    # PRODUCTION RATE swings over a realistic range, and this is the single most
    # important property of the generator. A plant that sits at one operating
    # point has narrow marginal limits on every tag, so ANY fault immediately
    # leaves them and univariate charts are sufficient -- which is exactly what
    # the first version of this simulator accidentally built, and it made the
    # multivariate case evaporate.
    #
    # A plant that changes rate all day has WIDE marginal limits and a TIGHT
    # correlation structure, and that gap is the entire operating region where
    # multivariate monitoring earns its place.
    rate = 1.0 + 0.09 * np.sin(2 * np.pi * t / 340.0) \
        + 0.05 * np.sin(2 * np.pi * t / 97.0)
    wander = np.cumsum(rng.normal(0, 0.02, n))
    wander -= wander.mean()

    feed_a = 0.25 + 0.01 * rng.normal(0, 1, n) + 0.02 * wander
    feed_d = 3664.0 * rate + 20.0 * rng.normal(0, 1, n) + 15 * wander
    feed_e = 4509.0 * rate + 25.0 * rng.normal(0, 1, n) + 18 * wander

    # A feed-composition STEP shifts the operating point; relationships hold.
    if fault == "feed_composition_step":
        feed_a = feed_a + 0.018 * severity * active

    # --- reaction and heat balance -------------------------------------
    # Kinetics: conversion rises with temperature and with A availability.
    kinetics = 1.0
    if fault == "kinetics_drift":
        kinetics = 1.0 - 0.030 * severity * ramp

    heat_release = (0.55 * feed_d + 0.45 * feed_e) * (0.9 + 0.4 * feed_a) * kinetics

    # Cooling. The heat-exchanger fouling fault degrades EFFECTIVENESS: the same
    # flow removes less heat. Flow itself stays in range -- which is precisely
    # why a chart on the flow tag sees nothing.
    # Per-sample, because fouling is a RAMP: it has to be indexed inside the
    # control loop below rather than multiplied in as a scalar.
    effectiveness = np.ones(n)
    if fault == "heat_exchanger_fouling":
        effectiveness = 1.0 - 0.055 * severity * ramp

    # The controller acts on temperature error by moving cooling water. Under a
    # stuck valve it cannot.
    cool_flow = np.empty(n)
    temp = np.empty(n)
    setpoint = 120.4
    cw = 94.6
    for i in range(n):
        removed = cw * float(effectiveness[i]) * 0.0125
        temp[i] = setpoint + (heat_release[i] - 3800.0) / 900.0 - (removed - 1.18) * 9.0
        temp[i] += rng.normal(0, 0.06)
        err = temp[i] - setpoint
        if fault == "valve_stick" and i >= fs:
            # Frozen. The load keeps changing, so the cooling flow is now wrong
            # for the load -- a broken relationship, with both tags still inside
            # their own (wide) limits for a good while.
            pass
        else:
            cw = float(np.clip(cw + 2.2 * err, 60.0, 130.0))
        cool_flow[i] = cw + rng.normal(0, 0.15)

    # --- everything downstream is a function of the above ---------------
    press = (2700.0 + 0.10 * (feed_d + feed_e) + 3.4 * (temp - setpoint)
             + rng.normal(0, 3.0, n))
    level = 75.0 + 0.004 * (feed_d - feed_e) + 0.8 * (temp - setpoint) \
        + rng.normal(0, 0.25, n)
    recycle = 9.35 + 0.0009 * (feed_d + feed_e) + 0.05 * (temp - setpoint) \
        + rng.normal(0, 0.08, n)

    sep_temp = 80.1 + 0.42 * (temp - setpoint) + rng.normal(0, 0.15, n)
    sep_level = 50.0 + 0.010 * (press - 2700.0) + rng.normal(0, 0.4, n)
    sep_under = 25.16 + 0.0016 * feed_d + 0.09 * (temp - setpoint) \
        + rng.normal(0, 0.12, n)

    strip_level = 50.0 + 0.35 * (sep_level - 50.0) + rng.normal(0, 0.5, n)
    strip_temp = 65.7 + 0.28 * (sep_temp - 80.1) + rng.normal(0, 0.2, n)
    strip_flow = 22.9 + 0.5 * (sep_under - 25.16) + rng.normal(0, 0.15, n)

    cw_out = 77.3 + 0.30 * (temp - setpoint) + 0.045 * (100.0 - cool_flow) \
        + rng.normal(0, 0.12, n)
    comp_work = 341.4 + 0.06 * (press - 2700.0) + 1.4 * (recycle - 9.35) \
        + rng.normal(0, 0.7, n)
    purge = 0.337 + 0.00004 * (press - 2700.0) + rng.normal(0, 0.004, n)
    product = (22.9 + 0.45 * (strip_flow - 22.9) - 0.02 * (temp - setpoint) ** 2
               + rng.normal(0, 0.16, n))

    x = np.column_stack([
        feed_a, feed_d, feed_e, recycle, temp, press, level,
        sep_temp, sep_level, sep_under, strip_level, strip_temp, strip_flow,
        cool_flow, cw_out, comp_work, purge, product])
    return ProcessRun(x=x.astype(float), fault=fault, fault_start=fs)


# ---------------------------------------------------------------------------
# residual-based (model-based) features
# ---------------------------------------------------------------------------

class ResidualModel:
    """One linear model per signal, predicting it from every OTHER signal.

    This is the "model-based monitoring" the spec names, and the reason it is
    worth the trouble is the same reason the process side is hard: a fault is a
    broken RELATIONSHIP, and a residual is exactly the quantity that measures
    one. If reactor temperature is normally predictable from feed rate and
    cooling flow to within 0.1 degrees, then a 2-degree residual means the
    relationship has changed -- even though 2 degrees may be well inside the
    temperature's own control limits.

    RIDGE, NOT ORDINARY LEAST SQUARES, and the reason is structural rather than
    numerical. Process variables are heavily collinear by construction -- that is
    what makes them a process rather than eighteen independent sensors -- so the
    normal equations are ill-conditioned and an OLS coefficient can be enormous
    and unstable. A model whose coefficients swing between refits produces
    residuals that move for reasons that have nothing to do with the plant.

    WHAT THIS CANNOT DO. A linear model of a nonlinear process is wrong
    everywhere except near the operating point it was fitted at. That is
    acceptable here because monitoring is a LOCAL question -- has this changed
    from where it was -- and it is exactly why a residual monitor must be refitted
    after a deliberate operating-point change, and must NOT be refitted in
    response to a fault.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.coefs: list = []
        self.resid_sd: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ResidualModel":
        x = np.asarray(x, dtype=float)
        self.mu, self.sd = x.mean(0), x.std(0)
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        z = (x - self.mu) / self.sd
        n_tags = z.shape[1]
        self.coefs = []
        res = np.empty_like(z)
        for j in range(n_tags):
            others = [k for k in range(n_tags) if k != j]
            a = z[:, others]
            g = a.T @ a + self.alpha * np.eye(len(others))
            w = np.linalg.solve(g, a.T @ z[:, j])
            self.coefs.append((others, w))
            res[:, j] = z[:, j] - a @ w
        self.resid_sd = np.where(res.std(0) < 1e-9, 1.0, res.std(0))
        return self

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """Standardised residuals: how far each signal is from where the OTHERS
        say it should be."""
        if self.mu is None:
            raise RuntimeError("fit() first")
        z = (np.asarray(x, dtype=float) - self.mu) / self.sd
        out = np.empty_like(z)
        for j, (others, w) in enumerate(self.coefs):
            out[:, j] = z[:, j] - z[:, others] @ w
        return out / self.resid_sd

    def r2(self, x: np.ndarray) -> np.ndarray:
        """Per-signal R². A signal nothing predicts has no relationship to break.

        Reported because it decides whether a residual is worth monitoring at
        all: at R² = 0.05 the residual is just the signal again, and a residual
        chart on it is a univariate chart with extra steps.
        """
        z = (np.asarray(x, dtype=float) - self.mu) / self.sd
        out = np.empty(z.shape[1])
        for j, (others, w) in enumerate(self.coefs):
            pred = z[:, others] @ w
            ss_res = float(((z[:, j] - pred) ** 2).sum())
            ss_tot = float(((z[:, j] - z[:, j].mean()) ** 2).sum())
            out[j] = 1.0 - ss_res / max(ss_tot, 1e-12)
        return out


def univariate_limits(train: np.ndarray, k: float = 3.0) -> tuple:
    """Per-tag 3-sigma limits, so the multivariate case has something to beat."""
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu - k * sd, mu + k * sd


def univariate_violations(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Rows where ANY tag is outside its own limits -- the wall of charts."""
    return ((x < lo) | (x > hi)).any(axis=1)
