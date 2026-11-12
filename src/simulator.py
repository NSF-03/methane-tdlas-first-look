"""Part 2: forward model of one 1-second absorbance spectrum on the measured wavelength grid.

Noise-free absorbance at C ppm is (C / 50) * T, where T is the HITRAN absorbance for
50 ppm (Figure 6) interpolated onto the 317-point measured grid. Added on top:
white detector noise, an optional smooth baseline change between gas and reference
scans, and an optional interference fringe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import data
from .data import REF_PPM

# Per-point white noise, AU: the s.d. of the residual column of the authors' fit
# (Figure_9, 2.06e-3 AU) rounded to 2.0e-3.
SIGMA_A = 2.0e-3
SIGMA_B = 2.0e-3  # s.d. of each baseline coefficient, AU


@dataclass(frozen=True)
class Conditions:
    sigma_a: float = SIGMA_A                # white noise per point, AU
    sigma_b: float = 0.0                    # baseline coefficients b0, b1, b2 ~ N(0, sigma_b^2), AU
    fringe_amp: tuple = (0.0, 0.0)          # fringe amplitude, uniform range, AU
    fringe_period: tuple = (0.15, 0.60)     # fringe period, uniform range, nm


SCENARIOS = {
    "A": Conditions(),
    "B": Conditions(sigma_b=SIGMA_B, fringe_amp=(0.0, 2.0e-3), fringe_period=(0.15, 0.60)),
    # Hypothetical stress test; the paper reports no fringes for this sensor.
    "C": Conditions(sigma_b=SIGMA_B, fringe_amp=(6.0e-3, 6.0e-3), fringe_period=(0.05, 0.12)),
}


class Simulator:
    def __init__(self):
        meas = data.measured_absorbance()
        hit = data.hitran_curve()
        self.wavelength = meas["wavelength_nm"]
        # np.interp holds the end values; the measured grid extends about 0.002 nm
        # below and 0.001 nm above the HITRAN range.
        self.template = np.interp(self.wavelength, hit["wavelength_nm"], hit["absorbance"])
        lo, hi = self.wavelength[0], self.wavelength[-1]
        self.x = 2 * (self.wavelength - lo) / (hi - lo) - 1
        raw = data.raw_signals()
        if not np.allclose(raw["wavelength_nm"], self.wavelength):
            raise ValueError("Figure_5 and Figure_6 grids differ")
        self.i_air = raw["air"]

    def clean(self, conc_ppm):
        """Noise-free absorbance, one row per concentration."""
        return np.outer(np.atleast_1d(conc_ppm) / REF_PPM, self.template)

    def spectra(self, conc_ppm, cond: Conditions, rng: np.random.Generator):
        """Simulated absorbance spectra, one row per concentration."""
        conc = np.atleast_1d(np.asarray(conc_ppm, dtype=float))
        n, m = conc.size, self.wavelength.size
        a = self.clean(conc) + rng.normal(0.0, cond.sigma_a, (n, m))
        if cond.sigma_b > 0:
            b = rng.normal(0.0, cond.sigma_b, (n, 3))
            a += b[:, [0]] + b[:, [1]] * self.x + b[:, [2]] * self.x ** 2
        if cond.fringe_amp[1] > 0:
            amp = rng.uniform(*cond.fringe_amp, n)
            period = rng.uniform(*cond.fringe_period, n)
            phase = rng.uniform(0.0, 2 * np.pi, n)
            # Wavelength is measured from the grid start; with a uniform random phase this
            # is the same distribution as a * sin(2 pi lambda / p + phi).
            dl = self.wavelength - self.wavelength[0]
            a += amp[:, None] * np.sin(2 * np.pi * dl[None, :] / period[:, None] + phase[:, None])
        return a

    def raw_signal(self, absorbance):
        """Raw-signal view for illustration: I = I_air * exp(-A), with I_air measured (Figure 5)."""
        return self.i_air * np.exp(-np.asarray(absorbance))
