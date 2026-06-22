"""Synthetic tests for gaussianity_diagnostics."""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from gaussianity_diagnostics import (
    OUTPUT_FIELDS,
    _anderson_rejection_flags,
    _pointwise_diagnostics,
    _qq_r2,
    compute_pointwise_gaussianity,
    plot_gaussianity_map,
    plot_point_histogram_and_qq,
    plot_rejection_mask,
    run_synthetic_test,
)


class GaussianityDiagnosticsTests(unittest.TestCase):
    def test_numpy_output_shape_and_fields(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(100, 3, 4))
        result = compute_pointwise_gaussianity(data)

        self.assertEqual(set(result), set(OUTPUT_FIELDS))
        for values in result.values():
            self.assertEqual(values.shape, (3, 4))

    def test_nan_and_zero_variance_handling(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(50, 2))
        data[:40, 0] = np.nan
        data[:, 1] = 3.0
        result = compute_pointwise_gaussianity(data, min_samples=20)

        for field in OUTPUT_FIELDS:
            self.assertTrue(np.isnan(result[field][0]))
            self.assertTrue(np.isnan(result[field][1]))

    def test_helpers_distinguish_normal_and_mixture_samples(self):
        rng = np.random.default_rng(3)
        normal = rng.normal(size=1000)
        mixture = np.concatenate(
            [rng.normal(-2, 0.5, 500), rng.normal(2, 0.5, 500)]
        )

        self.assertGreater(_qq_r2(normal), _qq_r2(mixture))
        normal_ad, normal_flags, _ = _anderson_rejection_flags(normal)
        mixture_ad, mixture_flags, _ = _anderson_rejection_flags(mixture)
        self.assertLess(normal_ad, mixture_ad)
        self.assertEqual(mixture_flags[5.0], 1.0)

        normal_diagnostics = _pointwise_diagnostics(normal)
        mixture_diagnostics = _pointwise_diagnostics(mixture)
        self.assertEqual(normal_diagnostics["gaussian_flag"], 1.0)
        self.assertEqual(mixture_diagnostics["gaussian_flag"], 0.0)

    def test_synthetic_field_classification(self):
        synthetic = run_synthetic_test(seed=4)
        self.assertGreater(
            synthetic["gaussian_compatible_fraction"],
            synthetic["non_gaussian_compatible_fraction"],
        )
        self.assertGreater(synthetic["gaussian_compatible_fraction"], 0.5)
        self.assertLess(synthetic["non_gaussian_compatible_fraction"], 0.1)

    def test_plotting_utilities(self):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(100, 4, 5))
        result = compute_pointwise_gaussianity(data)

        ax_map = plot_gaussianity_map(
            result,
            highlight_below=0.05,
            marker_points=[(1, 2)],
        )
        ax_mask = plot_rejection_mask(
            result,
            method="anderson",
            alpha=0.05,
            marker_points=[(1, 2)],
        )
        fig_point, axes_point = plot_point_histogram_and_qq(
            data, (1, 2)
        )

        self.assertIsNotNone(ax_map)
        self.assertIsNotNone(ax_mask)
        self.assertEqual(len(axes_point), 2)

        import matplotlib.pyplot as plt

        plt.close(ax_map.figure)
        plt.close(ax_mask.figure)
        plt.close(fig_point)


if __name__ == "__main__":
    unittest.main()
