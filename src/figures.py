"""Figures 1 to 4, drawn from the data files and from results/*.json."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter  # noqa: E402

from . import data, estimators  # noqa: E402
from .simulator import SCENARIOS, Simulator  # noqa: E402

# Okabe-Ito colours (colour-blind safe); black is used for measured data.
INK, GREY = "#000000", "#7f7f7f"
BLUE, VERMILION, GREEN, SKY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7"
EST_COLOUR = {"E1": BLUE, "E2": VERMILION, "E3": GREEN}
EST_LABEL = {"E1": "E1 single point", "E2": "E2 line fit", "E3": "E3 neural network"}
SCEN_LABEL = {"A": "A\nwhite noise\n(E3 trained on A)",
              "B": "B\n+ baseline + fringes\n(E3 trained on B)",
              "C": "C\nshort-period fringes\n(E3 trained on B)"}
DPI = 200
SEED_EXAMPLES = 401

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "legend.frameon": False, "axes.formatter.useoffset": False,
})


def sci_tex(v, sig=3):
    m, e = f"{v:.{sig - 1}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def _save(fig, name):
    data.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(data.FIGURES_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig1(rep):
    raw, pub, hit, au = data.raw_signals(), data.measured_absorbance(), data.hitran_curve(), data.authors_fit()
    wl = pub["wavelength_nm"]
    lf = rep["lorentzian"]["my_fit"]
    mine = estimators.lorentzians(np.array(lf["params"]), wl, lf["ref_nm"], lf["n_base"])
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(7.5, 9.4),
                           gridspec_kw={"height_ratios": [1, 1.6, 0.55, 0.55]})
    ax[0].plot(wl, raw["air"], color=INK, lw=1, label="Synthetic air")
    ax[0].plot(wl, raw["gas"], color=VERMILION, lw=1, label="50 ppm methane in synthetic air")
    ax[0].set_ylabel("Detector signal\n(arb. units)")
    ax[0].legend(loc="upper left")

    ax[1].plot(wl, pub["absorbance"], color=INK, lw=0.8, label=r"Published absorbance ($1 - I_{gas}/I_{air}$)")
    ax[1].plot(wl, -np.log(raw["gas"] / raw["air"]), color=SKY, lw=0.8,
               label=r"$-\ln(I_{gas}/I_{air})$ from the raw signals")
    ax[1].plot(hit["wavelength_nm"], hit["absorbance"], color=PURPLE, lw=1.8, label="HITRAN, 50 ppm")
    ax[1].plot(wl, au["fit"], color=VERMILION, lw=1.3, label="Authors' seven-Lorentzian fit")
    ax[1].plot(wl, mine, color=BLUE, lw=1.3, ls="--", label="My seven-Lorentzian fit")
    ax[1].set_ylabel("Absorbance (AU)")
    ax[1].set_ylim(0, 0.125)
    ax[1].legend(loc="upper left", fontsize=8, ncol=2)

    for a, fit, colour, label in ((ax[2], au["fit"], VERMILION, "Published minus authors' fit"),
                                  (ax[3], mine, BLUE, "Published minus my fit")):
        a.axhline(0, color=GREY, lw=0.8)
        a.plot(wl, pub["absorbance"] - fit, color=colour, lw=0.9)
        a.text(0.01, 0.96, label, transform=a.transAxes, va="top", fontsize=8)
        a.set_ylabel("Residual (AU)")
        a.set_ylim(-0.012, 0.012)
    ax[3].set_xlabel("Wavelength (nm)")
    fig.align_ylabels(ax)
    _save(fig, "fig1_reproduction.png")


def fig2(rep):
    cal, al = data.calibration(), data.allan()
    c = rep["calibration"]
    slope, icpt = c["slope_AU_per_ppm"], c["intercept_AU"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.errorbar(cal["conc_ppm"], cal["absorbance"], xerr=cal["conc_err_ppm"], yerr=cal["absorbance_err"],
                fmt="o", ms=4, color=INK, ecolor=GREY, capsize=2, label="Figure_7 data")
    xx = np.array([0.0, 52.0])
    a1.plot(xx, slope * xx + icpt, color=VERMILION, lw=1.3,
            label=f"Least-squares line\nslope {sci_tex(slope)} AU/ppm, $R^2$ = {c['r2']:.3f}")
    a1.set_xlabel("Methane concentration (ppm)")
    a1.set_ylabel("Single-point peak absorbance (AU)")
    a1.legend(loc="upper left")

    ppm = al["adev_AU"] / slope
    a2.loglog(al["tau_s"], ppm, color=INK, lw=0.9, label="Figure_8, converted with the Figure_7 slope")
    offsets = {"1s": ((10, 10), "left"), "20s": ((-10, -30), "right"), "40s": ((14, -34), "left")}
    for key, (xy, ha) in offsets.items():
        p = rep["allan"][key]
        v = p["AU"] / slope
        a2.plot(p["tau_s"], v, "o", color=VERMILION, ms=6, zorder=3,
                label="This work, at 1, 20 and 40 s" if key == "1s" else None)
        a2.annotate(f"{p['tau_s']:g} s: {v:.2f} ppm", (p["tau_s"], v), xytext=xy,
                    textcoords="offset points", ha=ha, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.6})
    paper_ppm = rep["paper"]["allan_40s_ppb"] / 1e3
    a2.plot(40, paper_ppm, "D", mfc="none", mec=BLUE, ms=7, mew=1.4, zorder=4,
            label=f"Paper: {rep['paper']['allan_40s_ppb']:.0f} ppb at 40 s")
    a2.set_ylim(0.1, 20)
    a2.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10]))
    a2.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    a2.yaxis.set_minor_formatter(NullFormatter())
    a2.set_xlabel("Averaging time (s)")
    a2.set_ylabel("Allan-Werle deviation (ppm)")
    sec = a2.secondary_yaxis("right", functions=(lambda y: y * slope, lambda y: y / slope))
    sec.set_ylabel("Allan-Werle deviation (AU)")
    a2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig2_calibration_allan.png")


def fig3(exp):
    res, th = exp["results"], exp["theory_white_noise"]
    scen, ests, w = ["A", "B", "C"], ["E1", "E2", "E3"], 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for j, e in enumerate(ests):
        xs = np.arange(3) + (j - 1) * w
        sd = [res[s][e]["2"]["sd_ppm"] for s in scen]
        a1.bar(xs, sd, w * 0.92, color=EST_COLOUR[e], label=EST_LABEL[e], zorder=2)
        for xi, v in zip(xs, sd):
            a1.text(xi, v * 1.06, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        bias = [res[s][e]["2"]["bias_ppm"] for s in scen]
        se = [res[s][e]["2"]["bias_se_ppm"] for s in scen]
        a2.errorbar(xs, bias, yerr=2 * np.array(se), fmt="o", ms=6, color=EST_COLOUR[e],
                    capsize=3, label=EST_LABEL[e], zorder=3)
    b = th["template_only_sd_ppm"]
    a1.hlines(b, -0.45, 0.45, colors=INK, linestyles=":", lw=1.4, zorder=3,
              label=f"Template-only fit, white-noise s.d. ({b:.2f} ppm)")
    a1.set_yscale("log")
    a1.set_ylim(0.05, 8)
    a1.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
    a1.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_ylabel("Standard deviation at 2 ppm (ppm)")
    a1.set_xticks(range(3), [SCEN_LABEL[s] for s in scen])
    a1.legend(loc="upper left", fontsize=8)
    a2.axhline(0, color=GREY, lw=0.8)
    a2.set_ylabel("Bias at 2 ppm (ppm), error bars ±2 s.e.")
    a2.set_xticks(range(3), [SCEN_LABEL[s] for s in scen])
    a2.legend(loc="upper left")
    lim = max(abs(v) for v in a2.get_ylim())
    a2.set_ylim(-lim * 1.2, lim * 1.2)
    fig.tight_layout()
    _save(fig, "fig3_estimators.png")


def fig4():
    sim = Simulator()
    rng = np.random.default_rng(SEED_EXAMPLES)
    wl = sim.wavelength
    fig = plt.figure(figsize=(10, 8.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.48, wspace=0.12)
    titles = {"A": "A: white noise", "B": "B: + baseline + fringes", "C": "C: short-period fringes (stress test)"}
    first = {}
    examples = {}
    for r, c_ppm in enumerate((2.0, 50.0)):
        for k, (s, cond) in enumerate(SCENARIOS.items()):
            ax = fig.add_subplot(gs[r, k], sharey=first.get(r))
            first.setdefault(r, ax)
            spec = sim.spectra([c_ppm], cond, rng)[0]
            examples[(s, c_ppm)] = spec
            ax.plot(wl, spec, color=INK, lw=0.6, label="Simulated 1-s spectrum")
            ax.plot(wl, sim.clean(c_ppm)[0], color=VERMILION, lw=1.4, label="Noise-free")
            ax.set_xticks([3312.6, 3313.0, 3313.4])
            ax.set_xlabel("Wavelength (nm)", fontsize=8)
            ax.text(0.03, 0.95, f"{c_ppm:g} ppm", transform=ax.transAxes, va="top", fontsize=9,
                    fontweight="bold")
            if r == 0:
                ax.set_title(titles[s], fontsize=9)
            if k == 0:
                ax.set_ylabel("Absorbance (AU)")
            else:
                ax.tick_params(labelleft=False)
            if r == 0 and k == 0:
                ax.legend(loc="lower right", fontsize=7.5)
    ax = fig.add_subplot(gs[2, :])
    raw = data.raw_signals()
    ax.plot(wl, raw["air"], color=INK, lw=1, label="Measured synthetic air ($I_{air}$)")
    ax.plot(wl, raw["gas"], color=VERMILION, lw=1, label="Measured 50 ppm")
    ax.plot(wl, sim.raw_signal(examples[("A", 50.0)]), color=BLUE, lw=0.7,
            label=r"Simulated 50 ppm, scenario A: $I_{air}\exp(-A)$")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Detector signal (arb. units)")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, "fig4_example_spectra.png")


def make_all():
    rep = data.load_json("reproduction.json")
    exp = data.load_json("experiment.json")
    fig1(rep)
    fig2(rep)
    fig3(exp)
    fig4()


if __name__ == "__main__":
    make_all()
