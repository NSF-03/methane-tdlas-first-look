"""Run the whole study in order and print the two summary tables.

    python run_all.py
"""
from __future__ import annotations

import platform
import time

import matplotlib
import numpy
import pandas
import scipy
import sklearn

from src import data, experiment, figures, reproduce


def sci(v, sig=3):
    """1.858e-3 -> '1.86e-3'."""
    m, e = f"{v:.{sig - 1}e}".split("e")
    return f"{m}e{int(e)}"


def table_part1(rep: dict) -> str:
    p, ab, cal, lod = rep["paper"], rep["absorbance"], rep["calibration"], rep["lod"]
    nea, al, lz, vib = rep["nea_from_paper_inputs"], rep["allan"], rep["lorentzian"], rep["vibration"]
    rows = [
        ("Published absorbance vs 1 − I_gas/I_air, max. difference (AU)",
         "absorbance defined as ΔI/I₀ (Eq. 2)", sci(ab["linear_max_abs_diff_AU"], 2)),
        ("Published absorbance vs −ln(I_gas/I_air): correlation; mean offset (AU)", "–",
         f"{ab['ln_corr_with_published']:.5f}; +{sci(ab['ln_mean_offset_AU'])}"),
        ("Peak absorbance of the 50 ppm spectrum (AU)", "used to derive the 70.1 cm pathlength",
         f"{ab['published_peak_AU']:.4f} at {ab['published_peak_wavelength_nm']:.2f} nm"),
        ("Figure 7 line: slope (AU/ppm); intercept (AU); R²", "linear within error",
         f"{sci(cal['slope_AU_per_ppm'])}; {sci(cal['intercept_AU'])}; {cal['r2']:.3f}"),
        ("Zero error (AU)", sci(p["zero_error_AU"], 2),
         f"{sci(cal['zero_row_absorbance_err_AU'])} (0 ppm row of Figure 7)"),
        ("Single-point limit of detection (ppm)", f"{p['single_point_lod_ppm']:.1f}",
         f"{lod['slope_ppm']:.2f} (2.2e-3 AU / Figure 7 slope); "
         f"{lod['peak_ppm']:.2f} (2.2e-3 AU / 50 ppm peak per ppm)"),
        ("Single-point NEA, 1 s (AU)", sci(p["nea_1s_AU"], 2),
         f"{sci(nea['1s_AU'])} from the paper's noise density and bandwidth; "
         f"Allan-Werle curve at 1 s: {sci(al['1s']['AU'])}"),
        ("Single-point NEA, 20 s (AU)", sci(p["nea_20s_AU"], 2),
         f"{sci(nea['20s_AU'])} likewise; Allan-Werle curve at 20 s: {sci(al['20s']['AU'])}"),
        ("Allan-Werle deviation at 40 s (AU)", sci(p["allan_40s_AU"], 2),
         f"{sci(al['40s']['AU'])}; minimum over 5–200 s: {sci(al['min_5_200s']['AU'])} "
         f"at {al['min_5_200s']['tau_s']:.0f} s"),
        ("Allan-Werle deviation at 40 s (ppb)", f"{p['allan_40s_ppb']:.0f}",
         f"{al['40s']['ppb_slope']:.0f} (Figure 7 slope); {al['40s']['ppb_peak']:.0f} (50 ppm peak)"),
        ("Seven-Lorentzian fit: residual s.d. (AU); curve peak (AU)", "–",
         f"authors' {sci(lz['authors_residual_sd_AU'])}; {lz['authors_peak_AU']:.4f}. "
         f"Mine {sci(lz['my_residual_sd_AU'])}; {lz['my_peak_AU']:.4f}"),
        ("Vibration test residual s.d., normal; with vibration (V)",
         f"{sci(p['vibration_residual_sd_normal_V'], 2)}; {sci(p['vibration_residual_sd_vibration_V'], 2)}",
         f"{sci(vib['normal_sd_V'])}; {sci(vib['vibration_sd_V'])}"),
    ]
    lines = ["| Quantity | Paper | Reproduced from open data |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def table_part3(exp: dict, conc="2") -> str:
    res = exp["results"]
    head = "| Scenario (values in ppm) | E1 bias | E1 s.d. | E2 bias | E2 s.d. | E3 bias | E3 s.d. |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    names = {"A": "A: white noise", "B": "B: + baseline + fringes (trained)",
             "C": "C: short-period fringes (not trained)"}
    for s in ("A", "B", "C"):
        cells = []
        for e in ("E1", "E2", "E3"):
            r = res[s][e][conc]
            cells += [f"{r['bias_ppm']:+.2f}", f"{r['sd_ppm']:.2f}"]
        lines.append(f"| {names[s]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    t0 = time.perf_counter()
    rep = reproduce.run()
    exp = experiment.run()
    figures.make_all()
    runtime = time.perf_counter() - t0
    data.save_json({
        "runtime_s": runtime,
        "python": platform.python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "pandas": pandas.__version__,
    }, "run_info.json")
    print("Part 1: paper value / reproduced from open data\n")
    print(table_part1(rep))
    print("\nPart 3: simulated 1-s spectra at 2 ppm, values in ppm "
          f"({exp['settings']['n_test']} spectra per cell)\n")
    print(table_part3(exp))
    print(f"\nTotal runtime {runtime:.1f} s")


if __name__ == "__main__":
    main()
