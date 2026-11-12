"""Basic checks. Run from the repository root with: python -m unittest -v"""
import unittest

import numpy as np

from src import data, estimators, reproduce
from src.simulator import SCENARIOS, Simulator


class TestBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulator()

    def test_template_finite_and_peaks_near_3312_87_nm(self):
        t = self.sim.template
        self.assertEqual(t.shape, self.sim.wavelength.shape)
        self.assertTrue(np.all(np.isfinite(t)))
        self.assertAlmostEqual(self.sim.wavelength[np.argmax(t)], 3312.87, delta=0.01)

    def test_noise_free_spectrum_returns_its_concentration(self):
        for c in (0.0, 2.0, 10.0, 50.0):
            spectrum = self.sim.clean(c)[0]
            self.assertAlmostEqual(float(estimators.single_point(spectrum, self.sim.template)), c, places=9)
            self.assertAlmostEqual(float(estimators.line_fit(spectrum, self.sim.template, self.sim.x)), c,
                                   places=6)

    def test_simulator_reproducible_with_seed(self):
        conc = np.full(5, 2.0)
        for cond in SCENARIOS.values():
            a = self.sim.spectra(conc, cond, np.random.default_rng(1))
            b = self.sim.spectra(conc, cond, np.random.default_rng(1))
            c = self.sim.spectra(conc, cond, np.random.default_rng(2))
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.allclose(a, c))

    def test_calibration_slope_within_5_percent(self):
        slope = reproduce.calibration_fit(data.calibration())["slope_AU_per_ppm"]
        self.assertLess(abs(slope / 1.858e-3 - 1), 0.05)


if __name__ == "__main__":
    unittest.main()
