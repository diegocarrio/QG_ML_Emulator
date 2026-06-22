"""Tests for PCA/EOF projection Gaussianity diagnostics."""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pca_gaussianity_diagnostics import (
    _anderson_diagnostics,
    _compute_pca_svd,
    _prepare_ensemble_matrix,
    _projection_diagnostics,
    _qq_r2,
    _synthetic_ensemble_fields,
    compare_pca_gaussianity_results,
    compute_pca_projection_gaussianity,
    plot_eof_pattern,
    plot_leading_eof_patterns,
    plot_pc_histogram_and_qq,
    plot_pca_reconstruction,
    plot_pca_gaussianity_summary,
    summarize_pca_gaussianity,
    reconstruct_pca_field,
)


class PcaGaussianityDiagnosticsTests(unittest.TestCase):
    def test_prepare_matrix_handles_missing_and_constant_features(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(100, 3, 4))
        data[:20, 0, 0] = np.nan  # Removed at 95% threshold.
        data[:, 0, 1] = 2.0  # Constant feature removed.
        data[:2, 1, 1] = np.nan  # Retained and mean-filled.

        prepared = _prepare_ensemble_matrix(
            data,
            min_valid_fraction=0.95,
        )
        self.assertEqual(prepared.matrix.shape, (100, 10))
        self.assertFalse(prepared.valid_feature_mask[0, 0])
        self.assertFalse(prepared.valid_feature_mask[0, 1])
        self.assertTrue(np.all(np.isfinite(prepared.matrix)))

    def test_pca_shapes_and_variance(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(80, 15))
        X -= X.mean(axis=0)
        result = _compute_pca_svd(X, n_components=5)

        self.assertEqual(result["scores"].shape, (80, 5))
        self.assertEqual(result["eofs_valid"].shape, (5, 15))
        self.assertTrue(
            np.all(np.diff(result["explained_variance"]) <= 0)
        )
        self.assertLessEqual(
            result["cumulative_explained_variance_ratio"][-1],
            1.0,
        )

    def test_projection_helpers_detect_mixture(self):
        rng = np.random.default_rng(3)
        normal = rng.normal(size=1000)
        mixture = np.concatenate(
            [rng.normal(-2, 0.5, 500), rng.normal(2, 0.5, 500)]
        )
        self.assertGreater(_qq_r2(normal), _qq_r2(mixture))
        self.assertLess(
            _anderson_diagnostics(normal, 0.05)["statistic"],
            _anderson_diagnostics(mixture, 0.05)["statistic"],
        )

        scores = np.column_stack([normal, mixture])
        diagnostics = _projection_diagnostics(scores)
        self.assertEqual(diagnostics["anderson_reject_5"][1], 1.0)
        self.assertEqual(diagnostics["gaussian_flag"][1], 0.0)

    def test_end_to_end_synthetic_classification(self):
        gaussian, non_gaussian = _synthetic_ensemble_fields(
            seed=4,
            n_members=600,
            shape=(10, 12),
        )
        gaussian_result = compute_pca_projection_gaussianity(
            gaussian,
            n_components=6,
            return_eofs=True,
        )
        non_gaussian_result = compute_pca_projection_gaussianity(
            non_gaussian,
            n_components=6,
            return_eofs=True,
        )

        self.assertEqual(gaussian_result["scores"].shape, (600, 6))
        self.assertEqual(gaussian_result["eofs"].shape, (6, 10, 12))
        self.assertEqual(
            gaussian_result["valid_feature_mask"].shape,
            (10, 12),
        )
        self.assertGreater(
            np.mean(gaussian_result["gaussian_flag"]),
            np.mean(non_gaussian_result["gaussian_flag"]),
        )
        self.assertEqual(non_gaussian_result["anderson_reject_5"][0], 1.0)

    def test_comparison_summary_and_plots(self):
        gaussian, non_gaussian = _synthetic_ensemble_fields(
            seed=5,
            n_members=300,
            shape=(8, 8),
        )
        ref = compute_pca_projection_gaussianity(
            gaussian,
            n_components=4,
            return_eofs=True,
        )
        test = compute_pca_projection_gaussianity(
            non_gaussian,
            n_components=4,
            return_eofs=True,
        )
        comparison = compare_pca_gaussianity_results(ref, test)
        summary = summarize_pca_gaussianity(test, n_leading=4)
        fig_summary, _ = plot_pca_gaussianity_summary(test)
        fig_pc, _ = plot_pc_histogram_and_qq(test, component=0)
        ax_eof = plot_eof_pattern(test, component=0)
        reconstruction = reconstruct_pca_field(
            test,
            member=0,
            n_components=4,
        )
        fig_reconstruction, axes_reconstruction = plot_pca_reconstruction(
            test,
            non_gaussian,
            member=0,
            n_components=4,
        )
        fig_leading, axes_leading = plot_leading_eof_patterns(
            test,
            n_components=4,
            ncols=2,
        )

        self.assertEqual(
            comparison["delta_anderson_statistic"].shape,
            (4,),
        )
        self.assertEqual(comparison["anderson_reject_test"].shape, (4,))
        self.assertEqual(summary["n_leading"], 4)
        self.assertEqual(axes_leading.shape, (2, 2))
        self.assertEqual(reconstruction.shape, (8, 8))
        self.assertEqual(axes_reconstruction.shape, (2, 2))

        import matplotlib.pyplot as plt

        plt.close(fig_summary)
        plt.close(fig_pc)
        plt.close(ax_eof.figure)
        plt.close(fig_leading)
        plt.close(fig_reconstruction)

    def test_full_rank_reconstruction_recovers_centered_input(self):
        rng = np.random.default_rng(6)
        data = rng.normal(size=(30, 4, 5))
        result = compute_pca_projection_gaussianity(
            data,
            n_components=29,
            return_eofs=True,
        )
        reconstructed = reconstruct_pca_field(
            result,
            member=3,
            n_components=result["scores"].shape[1],
        )
        np.testing.assert_allclose(reconstructed, data[3], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
