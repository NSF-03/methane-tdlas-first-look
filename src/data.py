"""Loading helpers for the open data of Davis et al. (2023).

Paths are relative to the repository root. The data files list wavelength in
descending order; every spectrum returned here is sorted by ascending wavelength.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "davis2023"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Values stated in the text of Davis et al. (2023). Used only for comparison.
PAPER = {
    "effective_pathlength_cm": 70.1,
    "single_point_wavelength_nm": 3312.85,
    "zero_error_AU": 2.2e-3,
    "single_point_lod_ppm": 1.1,
    "absorbance_noise_density_AU_per_rtHz": 8.4e-5,  # one measurement subtracted from a second
    "nebw_1s_Hz": 200.0,
    "nebw_20s_Hz": 10.0,
    "nea_1s_AU": 1.2e-3,
    "nea_20s_AU": 2.7e-4,
    "allan_40s_AU": 4.3e-4,
    "allan_40s_ppb": 220.0,
    "fit_lod_ppb": 180.0,
    "fit_lod_number_of_fits": 20,
    "vibration_residual_sd_normal_V": 5.8e-4,
    "vibration_residual_sd_vibration_V": 4.2e-4,
}

REF_PPM = 50.0  # concentration of the measured spectrum and of the HITRAN curve


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def _ascending(wavelength, *columns):
    wavelength = np.asarray(wavelength, dtype=float)
    order = np.argsort(wavelength)
    return (wavelength[order],) + tuple(np.asarray(c, dtype=float)[order] for c in columns)


def raw_signals() -> dict:
    """Figure 5: detector signals for 50 ppm methane and for synthetic air."""
    d = _read("Figure_5")
    wl, gas, air = _ascending(d["Wavelength (nm)"], d["50ppm methane in synthetic air"],
                              d["Synthetic air"])
    return {"wavelength_nm": wl, "gas": gas, "air": air}


def measured_absorbance() -> dict:
    """Figure 6: published absorbance of 50 ppm methane on the measured grid."""
    d = _read("Figure_6")[["Detected Absorption Wavelength (nm)", "Detected Absorption (AU)"]].dropna()
    wl, a = _ascending(d.iloc[:, 0], d.iloc[:, 1])
    return {"wavelength_nm": wl, "absorbance": a}


def hitran_curve() -> dict:
    """Figure 6: HITRAN absorbance for 50 ppm at 25 C and atmospheric pressure."""
    d = _read("Figure_6")[["HITRAN Wavelength (nm)", "HITRAN data"]].dropna()
    wl, a = _ascending(d.iloc[:, 0], d.iloc[:, 1])
    return {"wavelength_nm": wl, "absorbance": a}


def calibration() -> dict:
    """Figure 7: single-point peak absorbance against concentration."""
    d = _read("Figure_7")
    return {
        "conc_ppm": d["Concentration (ppm)"].to_numpy(float),
        "absorbance": d["Absorbance (AU)"].to_numpy(float),
        "conc_err_ppm": d["Concentration error (ppm)"].to_numpy(float),
        "absorbance_err": d["Absorbance error (AU)"].to_numpy(float),
    }


def allan() -> dict:
    """Figure 8: Allan-Werle deviation (already computed by the authors); zero rows dropped."""
    d = _read("Figure_8")
    d = d[d["Allan Deviation (AU)"] > 0]
    return {"tau_s": d["Averaging Time (s)"].to_numpy(float),
            "adev_AU": d["Allan Deviation (AU)"].to_numpy(float)}


def authors_fit() -> dict:
    """Figure 9: published absorbance, the authors' seven-Lorentzian fit and its residual."""
    d = _read("Figure_9")
    wl, a, fit, res = _ascending(d["Wavelength (nm)"], d["Detected Absorption (AU)"],
                                 d["Lorentzian Fit (AU)"], d["Residual (AU)"])
    return {"wavelength_nm": wl, "absorbance": a, "fit": fit, "residual": res}


def vibration_residuals() -> dict:
    """Figure 11: residuals of the ramp waveforms without and with vibration."""
    d = _read("Figure_11")
    return {"time_ms": d["Time (ms)"].to_numpy(float),
            "normal_V": d["No Vibration Residual (V)"].to_numpy(float),
            "vibration_V": d["Vibration Present Residual (V)"].to_numpy(float)}


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(obj: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(_plain(obj), indent=2) + "\n")
    return path


def load_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())
