"""Part 3: compare three ways of turning one simulated 1-s spectrum into a concentration.

E1 single point, E2 linear line fit with a quadratic baseline, E3 a small neural network.
"""
from __future__ import annotations

import time

import numpy as np

from . import data, estimators, reproduce
from .data import PAPER, REF_PPM
from .simulator import SCENARIOS, SIGMA_A, Simulator

N_TRAIN = 50_000
N_TEST = 2_000
TEST_PPM = (0.0, 2.0, 10.0, 50.0)
TRAIN_RANGE_PPM = (0.0, 50.0)
NETWORK_FOR = {"A": "A", "B": "B", "C": "B"}  # scenario C is never seen in training
SEED_TRAIN_DATA = {"A": 101, "B": 102}
SEED_NETWORK = {"A": 201, "B": 202}
SEED_TEST = 301


def summarise(est, true_ppm) -> dict:
    est = np.asarray(est)
    err = est - true_ppm
    sd = np.std(est, ddof=1)
    return {"mean_ppm": est.mean(), "bias_ppm": err.mean(), "sd_ppm": sd,
            "rmse_ppm": np.sqrt(np.mean(err ** 2)), "bias_se_ppm": sd / np.sqrt(est.size)}


def train_networks(sim: Simulator) -> tuple[dict, dict]:
    nets, info = {}, {}
    for name in ("A", "B"):
        rng = np.random.default_rng(SEED_TRAIN_DATA[name])
        conc = rng.uniform(*TRAIN_RANGE_PPM, N_TRAIN)
        x = sim.spectra(conc, SCENARIOS[name], rng)
        t0 = time.perf_counter()
        nets[name] = estimators.train_network(x, conc, SEED_NETWORK[name])
        info[name] = {"train_seconds": time.perf_counter() - t0,
                      "epochs": int(nets[name][-1].n_iter_), "n_train": N_TRAIN}
    return nets, info


def run() -> dict:
    t_start = time.perf_counter()
    sim = Simulator()
    T, x = sim.template, sim.x
    nets, net_info = train_networks(sim)

    table = {}
    for si, (scen, cond) in enumerate(SCENARIOS.items()):
        table[scen] = {"E1": {}, "E2": {}, "E3": {}}
        for ci, c in enumerate(TEST_PPM):
            rng = np.random.default_rng([SEED_TEST, si, ci])
            s = sim.spectra(np.full(N_TEST, c), cond, rng)
            key = f"{c:g}"
            table[scen]["E1"][key] = summarise(estimators.single_point(s, T), c)
            table[scen]["E2"][key] = summarise(estimators.line_fit(s, T, x), c)
            table[scen]["E3"][key] = summarise(
                estimators.network_predict(nets[NETWORK_FOR[scen]], s), c)

    # Theory under white noise (scenario A) and consistency checks.
    e2_theory = estimators.coefficient_sd(T, x, SIGMA_A, n_poly=3)
    e2_sim = [table["A"]["E2"][f"{c:g}"]["sd_ppm"] for c in TEST_PPM]
    e1_sim = [table["A"]["E1"][f"{c:g}"]["sd_ppm"] for c in TEST_PPM]
    n20 = PAPER["fit_lod_number_of_fits"]
    al = data.allan()
    allan_1s = float(al["adev_AU"][np.argmin(np.abs(al["tau_s"] - 1.0))])
    theory = {
        "sigma_a_AU": SIGMA_A,
        "e1_sd_ppm": REF_PPM * SIGMA_A / T.max(),
        "e2_sd_ppm": e2_theory,
        "template_plus_constant_sd_ppm": estimators.coefficient_sd(T, x, SIGMA_A, n_poly=1),
        "template_only_sd_ppm": estimators.coefficient_sd(T, x, SIGMA_A, n_poly=0),
        "e2_sim_over_theory_min": min(e2_sim) / e2_theory,
        "e2_sim_over_theory_max": max(e2_sim) / e2_theory,
        "sd_relative_se_n_test": 1 / np.sqrt(2 * (N_TEST - 1)),
        "sd_relative_se_n20": 1 / np.sqrt(2 * (n20 - 1)),
        # Same formulas with sigma set to the 1-s single-point Allan-Werle deviation (Figure 8).
        "allan_1s_AU": allan_1s,
        "e2_sd_ppm_at_allan_1s": estimators.coefficient_sd(T, x, allan_1s, n_poly=3),
        "template_plus_constant_sd_ppm_at_allan_1s":
            estimators.coefficient_sd(T, x, allan_1s, n_poly=1),
    }
    checks = {
        "e2_sd_A_2ppm_ppb": 1e3 * table["A"]["E2"]["2"]["sd_ppm"],
        "paper_fit_lod_ppb": PAPER["fit_lod_ppb"],
        "e1_sd_A_2ppm_ppm": table["A"]["E1"]["2"]["sd_ppm"],
        "e1_sd_A_range_ppm": [min(e1_sim), max(e1_sim)],
        "paper_single_point_lod_ppm": PAPER["single_point_lod_ppm"],
    }

    # Real-data check on the one published 50 ppm spectrum.
    pub = data.measured_absorbance()
    if not np.allclose(pub["wavelength_nm"], sim.wavelength):
        raise ValueError("grid mismatch")
    a = pub["absorbance"]
    fit = reproduce.fit_published_spectrum()
    by_terms = {}
    for n in range(4):  # template fit with 0 to 3 baseline terms; n = 3 is E2
        X = estimators.line_fit_design(T, x, n)
        coef, *_ = np.linalg.lstsq(X, a, rcond=None)
        by_terms[str(n)] = {"ppm": REF_PPM * coef[0], "residual_sd_AU": np.std(a - X @ coef, ddof=1)}
    real = {
        "E1_ppm": float(estimators.single_point(a, T)),
        "E2_ppm": float(estimators.line_fit(a, T, x)),
        "lorentzian_curve_peak_ppm": REF_PPM * fit["curve"].max() / T.max(),
        "E3_network_A_ppm": float(estimators.network_predict(nets["A"], a)[0]),
        "E3_network_B_ppm": float(estimators.network_predict(nets["B"], a)[0]),
        "template_fit_by_baseline_terms": by_terms,
        "note": "Lorentzian estimate = 50 ppm x peak of the fitted curve / peak of the HITRAN template.",
    }

    out = {
        "settings": {"n_train": N_TRAIN, "n_test": N_TEST, "test_ppm": TEST_PPM,
                     "train_range_ppm": TRAIN_RANGE_PPM, "network_for": NETWORK_FOR,
                     "hidden_layers": [64, 32],
                     "scenarios": {k: vars(v) for k, v in SCENARIOS.items()},
                     "seeds": {"train_data": SEED_TRAIN_DATA, "network": SEED_NETWORK,
                               "test": SEED_TEST}},
        "template": {"peak_AU": T.max(), "peak_wavelength_nm": sim.wavelength[np.argmax(T)]},
        "networks": net_info,
        "results": table,
        "theory_white_noise": theory,
        "consistency": checks,
        "real_spectrum": real,
        "runtime_s": time.perf_counter() - t_start,
    }
    data.save_json(out, "experiment.json")
    return out


if __name__ == "__main__":
    run()
