"""Ways of turning one absorbance spectrum into a methane concentration (ppm)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import REF_PPM

N_LINES = 7


# E1 -------------------------------------------------------------------------
def single_point(spectra, template):
    """E1: absorbance at the template peak divided by the template peak, times 50 ppm."""
    i = int(np.argmax(template))
    return REF_PPM * np.asarray(spectra)[..., i] / template[i]


# E2 -------------------------------------------------------------------------
def line_fit_design(template, x, n_poly=3):
    """Columns [template, 1, x, x^2] (n_poly baseline terms)."""
    return np.column_stack([template] + [x ** k for k in range(n_poly)])


def line_fit(spectra, template, x):
    """E2: linear least squares on [template, 1, x, x^2]; concentration = 50 * template coefficient."""
    weights = np.linalg.pinv(line_fit_design(template, x))[0]
    return REF_PPM * np.asarray(spectra) @ weights


def coefficient_sd(template, x, sigma, n_poly=3):
    """Standard deviation (ppm) of the template coefficient under white noise of s.d. sigma (AU)."""
    X = line_fit_design(template, x, n_poly)
    return REF_PPM * sigma * np.sqrt(np.linalg.inv(X.T @ X)[0, 0])


# Seven-Lorentzian non-linear fit (Part 1) -----------------------------------------
def lorentzians(params, wavelength, ref, n_base=1):
    """Sum of N_LINES Lorentzians (peak-height form) plus a polynomial baseline.

    params = [centre offsets from ref (nm) x7, half widths (nm) x7, peak heights (AU) x7,
    baseline coefficients x n_base]; the baseline uses wavelength scaled to [-1, 1].
    """
    c, g, a = params[:N_LINES], params[N_LINES:2 * N_LINES], params[2 * N_LINES:3 * N_LINES]
    u = (wavelength - ref)[:, None] - c[None, :]
    curve = (a * g ** 2 / (u ** 2 + g ** 2)).sum(axis=1)
    xs = 2 * (wavelength - wavelength.min()) / np.ptp(wavelength) - 1
    for k in range(n_base):
        curve = curve + params[3 * N_LINES + k] * xs ** k
    return curve


def fit_lorentzians(wavelength, absorbance, centres0, width0=0.05, n_base=1):
    """Levenberg-Marquardt fit of seven Lorentzians and a baseline to one spectrum."""
    ref = float(np.mean(wavelength))
    p0 = np.concatenate([np.asarray(centres0) - ref, np.full(N_LINES, width0),
                         np.full(N_LINES, 0.5 * np.mean(absorbance)), np.zeros(n_base)])
    sol = least_squares(lambda p: lorentzians(p, wavelength, ref, n_base) - absorbance, p0,
                        method="lm", x_scale="jac", max_nfev=20000)
    curve = lorentzians(sol.x, wavelength, ref, n_base)
    p = sol.x
    return {
        "centres_nm": p[:N_LINES] + ref,
        "half_widths_nm": np.abs(p[N_LINES:2 * N_LINES]),
        "heights_AU": p[2 * N_LINES:3 * N_LINES],
        "baseline_AU": p[3 * N_LINES:],
        "params": p, "ref_nm": ref, "n_base": n_base,
        "curve": curve, "residual": absorbance - curve,
        "status": int(sol.status), "nfev": int(sol.nfev),
    }


# E3 -------------------------------------------------------------------------
def make_network(seed):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300,
                     n_iter_no_change=20, random_state=seed),
    )


def train_network(spectra, conc_ppm, seed):
    """E3: small multilayer perceptron trained to map a spectrum to concentration / 50 ppm."""
    net = make_network(seed)
    net.fit(spectra, np.asarray(conc_ppm) / REF_PPM)
    return net


def network_predict(net, spectra):
    return REF_PPM * net.predict(np.atleast_2d(spectra))
