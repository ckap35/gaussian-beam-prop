import unittest

import numpy as np

from src.pi_phase_kick import focal_field


class FocusTests(unittest.TestCase):
    def setUp(self):
        self.wavelength = 632.8e-9
        self.focal_length = 0.15
        self.radius = 1.5e-3
        self.size = 256
        self.dx = 12e-3 / self.size
        x = (np.arange(self.size) - self.size // 2) * self.dx
        self.X, Y = np.meshgrid(x, x, indexing="xy")
        self.pupil = np.exp(-(self.X**2 + Y**2) / self.radius**2)

    def propagate(self, pupil):
        return focal_field(pupil, self.dx, self.wavelength, self.focal_length)

    def test_unmasked_focus_matches_analytic_gaussian(self):
        intensity = np.abs(self.propagate(self.pupil))**2
        frequency = np.fft.fftshift(np.fft.fftfreq(self.size, self.dx))
        focal_x = self.wavelength * self.focal_length * frequency
        waist = self.wavelength * self.focal_length / (np.pi * self.radius)
        expected = np.exp(-2 * (focal_x / waist)**2)
        actual = intensity[self.size // 2] / intensity.max()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)

    def test_fourier_scaling_conserves_integrated_power(self):
        focal_dx = self.wavelength * self.focal_length / (self.size * self.dx)
        pupil_power = np.sum(np.abs(self.pupil)**2) * self.dx**2
        focal_power = np.sum(np.abs(self.propagate(self.pupil))**2) * focal_dx**2
        self.assertAlmostEqual(focal_power / pupil_power, 1)

    def test_phase_step_preserves_power_and_suppresses_center(self):
        unmasked = np.abs(self.propagate(self.pupil))**2
        mask = np.where(self.X >= 0, 1, -1)
        masked = np.abs(self.propagate(self.pupil * mask))**2
        self.assertAlmostEqual(masked.sum() / unmasked.sum(), 1)
        mid = self.size // 2
        self.assertLess(masked[mid, mid] / unmasked[mid, mid], 1e-3)


if __name__ == "__main__":
    unittest.main()
