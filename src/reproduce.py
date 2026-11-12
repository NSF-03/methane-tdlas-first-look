"""Part 1: reproduce the numbers reported by Davis et al. (2023) from their open data."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from . import data, estimators
from .data import PAPER, REF_PPM

ALLAN_TAUS_S = (1.0, 20.0, 40.0)
ALLAN_SEARCH_S = (5.0, 200.0)  # the curve is noisy at very long times


def absorbance_check() -> dict:
    """(a) Absorbance recomputed from the raw signals against the published absorbance."""
    raw, pub = data.raw_signals(), data.measured_absorbance()
    if not np.allclose(raw["wavelength_nm"], pub["wavelength_nm"]):
        raise ValueError("Figure_5 and Figure_6 grids differ")
    ratio = raw["gas"] / raw["air"]
    a_ln, a_lin, a_pub = -np.log(ratio), 1.0 - ratio, pub["absorbance"]
    i = int(np.argmax(a_pub))
    return {
        "ln_corr_with_published": np.corrcoef(a_ln, a_pub)[0, 1],
        "ln_mean_offset_AU": np.mean(a_ln - a_pub),
        "ln_peak_AU": a_ln[i],
        "linear_max_abs_diff_AU": np.max(np.abs(a_lin - a_pub)),
        "published_peak_AU": a_pub[i],
        "published_peak_wavelength_nm": pub["wavelength_nm"][i],
        "ln_over_linear_at_peak": a_ln[i] / a_pub[i],
    }


def calibration_fit(cal: dict) -> dict:
    """(b) Straight line through the Figure 7 points (unweighted)."""
    c, a = cal["conc_ppm"], cal["absorbance"]
    slope, intercept = np.polyfit(c, a, 1)
    r2 = 1 - np.sum((a - (slope * c + intercept)) ** 2) / np.sum((a - a.mean()) ** 2)
    zero = c == 0
    return {
        "slope_AU_per_ppm": slope, "intercept_AU": intercept, "r2": r2,
        "zero_row_absorbance_AU": float(a[zero][0]),
        "zero_row_absorbance_err_AU": float(cal["absorbance_err"][zero][0]),
    }


def allan_check(slope: float, peak_factor: float) -> dict:
    """(c) Allan-Werle curve at fixed averaging times, in AU and ppb."""
    al = data.allan()
    tau, dev = al["tau_s"], al["adev_AU"]
    out = {}
    for t in ALLAN_TAUS_S:
        i = int(np.argmin(np.abs(tau - t)))
        out[f"{t:g}s"] = {"tau_s": tau[i], "AU": dev[i], "ppb_slope": 1e3 * dev[i] / slope,
                          "ppb_peak": 1e3 * dev[i] / peak_factor}
    win = (tau >= ALLAN_SEARCH_S[0]) & (tau <= ALLAN_SEARCH_S[1])
    j = np.flatnonzero(win)[np.argmin(dev[win])]
    out["min_5_200s"] = {"tau_s": tau[j], "AU": dev[j], "ppb_slope": 1e3 * dev[j] / slope,
                         "ppb_peak": 1e3 * dev[j] / peak_factor}
    return out


def curvature_seeds(n=estimators.N_LINES) -> np.ndarray:
    """Alternative start: the n strongest maxima of -d2A/dlambda2 of the HITRAN curve."""
    h = data.hitran_curve()
    wl, a = h["wavelength_nm"], h["absorbance"]
    d2 = np.gradient(np.gradient(a, wl), wl)
    pk, props = find_peaks(-d2, prominence=0)
    best = pk[np.argsort(props["prominences"])[::-1][:n]]
    return np.sort(wl[best])


def even_seeds(wl, n=estimators.N_LINES) -> np.ndarray:
    return wl.min() + (np.arange(n) + 0.5) * np.ptp(wl) / n


def fit_published_spectrum(n_base=1, seeds="even") -> dict:
    """(d) Seven-Lorentzian Levenberg-Marquardt fit to the published 50 ppm absorbance."""
    pub = data.measured_absorbance()
    wl = pub["wavelength_nm"]
    c0 = even_seeds(wl) if seeds == "even" else curvature_seeds()
    return estimators.fit_lorentzians(wl, pub["absorbance"], c0, n_base=n_base)


def lorentzian_check() -> dict:
    au = data.authors_fit()
    pub = data.measured_absorbance()
    if not np.allclose(au["absorbance"], pub["absorbance"]):
        raise ValueError("Figure_9 data column differs from Figure_6")
    fit = fit_published_spectrum()
    alt = fit_published_spectrum(seeds="curvature")
    lin = fit_published_spectrum(n_base=2)
    mine, theirs = fit["curve"], au["fit"]
    offset = np.mean(theirs - au["absorbance"])
    res = fit["residual"] - fit["residual"].mean()
    return {
        "my_fit": {k: fit[k] for k in ("centres_nm", "half_widths_nm", "heights_AU", "baseline_AU",
                                       "params", "ref_nm", "n_base", "status", "nfev")},
        "my_residual_sd_AU": np.std(fit["residual"], ddof=1),
        "my_residual_lag1_autocorr": np.corrcoef(res[:-1], res[1:])[0, 1],
        "my_peak_AU": mine.max(),
        "my_negative_heights": int(np.sum(fit["heights_AU"] < 0)),
        "authors_residual_sd_AU": np.std(au["residual"], ddof=1),
        "authors_peak_AU": theirs.max(),
        "authors_fit_minus_data_mean_AU": offset,
        "authors_residual_vs_fit_minus_data_max_abs_AU":
            np.max(np.abs(au["residual"] - (theirs - au["absorbance"] - offset))),
        "rms_mine_vs_authors_AU": np.sqrt(np.mean((mine - theirs) ** 2)),
        "rms_mine_vs_authors_offset_removed_AU": np.std(mine - theirs),
        "curvature_start_max_abs_curve_diff_AU": np.max(np.abs(alt["curve"] - mine)),
        "curvature_start_residual_sd_AU": np.std(alt["residual"], ddof=1),
        "linear_baseline_residual_sd_AU": np.std(lin["residual"], ddof=1),
        "linear_baseline_max_abs_height_AU": np.max(np.abs(lin["heights_AU"])),
    }


def run() -> dict:
    absb = absorbance_check()
    cal = calibration_fit(data.calibration())
    peak_factor = absb["published_peak_AU"] / REF_PPM  # AU per ppm from the 50 ppm peak
    z = PAPER["zero_error_AU"]
    lod = {
        "slope_ppm": z / cal["slope_AU_per_ppm"],
        "slope_file_zero_ppm": cal["zero_row_absorbance_err_AU"] / cal["slope_AU_per_ppm"],
        "peak_ppm": z / peak_factor,
        "peak_factor_AU_per_ppm": peak_factor,
    }
    nd = PAPER["absorbance_noise_density_AU_per_rtHz"]
    nea = {"1s_AU": nd * np.sqrt(PAPER["nebw_1s_Hz"]), "20s_AU": nd * np.sqrt(PAPER["nebw_20s_Hz"])}
    vib = data.vibration_residuals()
    out = {
        "paper": PAPER,
        "absorbance": absb,
        "calibration": cal,
        "lod": lod,
        "nea_from_paper_inputs": nea,
        "allan": allan_check(cal["slope_AU_per_ppm"], peak_factor),
        "lorentzian": lorentzian_check(),
        "vibration": {"normal_sd_V": np.std(vib["normal_V"], ddof=1),
                      "vibration_sd_V": np.std(vib["vibration_V"], ddof=1)},
    }
    data.save_json(out, "reproduction.json")
    return out


if __name__ == "__main__":
    run()
