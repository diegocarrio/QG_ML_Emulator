"""Gaussianity diagnostics for PCA/EOF projections of ensemble fields.

If a full ensemble state vector is multivariate Gaussian, every linear
projection of that state is also Gaussian. This module projects ensemble
anomalies onto leading PCA/EOF directions and applies univariate diagnostics
to the resulting principal-component (PC) scores.

Rejecting normality for one or more PC scores is evidence against multivariate
Gaussianity in the sampled subspace. Conversely, not rejecting normality does
not prove that the full ensemble is Gaussian.

Missing-value handling
----------------------
Features with a finite-value fraction below ``min_valid_fraction`` are removed.
Occasional NaNs in retained features are filled with that feature's ensemble
mean. Constant features are removed because they contain no PCA variance.

NumPy input returns a dictionary. If xarray is installed, xarray DataArray
input returns an xarray Dataset preserving member and non-member coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:  # Optional dependency.
    import xarray as xr
except ImportError:  # pragma: no cover - environment-dependent.
    xr = None


ANDERSON_LEVELS = np.array([15.0, 10.0, 5.0, 2.5, 1.0])
LEVEL_SUFFIXES = {
    15.0: "15",
    10.0: "10",
    5.0: "5",
    2.5: "2p5",
    1.0: "1",
}


@dataclass
class _PreparedEnsemble:
    """Internal representation of an ensemble matrix and its metadata."""

    matrix: np.ndarray
    valid_feature_mask: np.ndarray
    original_feature_shape: tuple[int, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    member_axis: int
    member_name: str | None
    member_coordinate: Any
    feature_dims: tuple[str, ...] | None
    feature_coordinates: dict[str, Any] | None
    input_was_xarray: bool


def _normalize_axis_index(axis: int, ndim: int) -> int:
    if not isinstance(axis, (int, np.integer)):
        raise TypeError("member_dim must be an integer axis or xarray dimension.")
    if axis < -ndim or axis >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return int(axis % ndim)


def _qq_r2(sample: np.ndarray) -> float:
    """Return the linear-fit R² of a normal Q-Q plot."""

    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 3 or np.ptp(sample) == 0:
        return np.nan
    _, (_, _, correlation) = stats.probplot(sample, dist="norm", fit=True)
    return float(correlation**2)


def _nearest_anderson_level(alpha: float) -> float:
    """Return the available Anderson-Darling level nearest to ``alpha``."""

    return float(ANDERSON_LEVELS[np.argmin(np.abs(ANDERSON_LEVELS - 100 * alpha))])


def _anderson_diagnostics(
    sample: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Return Anderson-Darling statistic, critical values, and rejection flags."""

    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 3 or np.ptp(sample) == 0:
        return {
            "statistic": np.nan,
            "significance_levels": ANDERSON_LEVELS.copy(),
            "critical_values": np.full(ANDERSON_LEVELS.shape, np.nan),
            "rejection_flags": np.full(ANDERSON_LEVELS.shape, np.nan),
            "selected_level": _nearest_anderson_level(alpha),
            "selected_reject": np.nan,
        }

    # The critical-value interface is intentionally used because the requested
    # output includes SciPy's 15%, 10%, 5%, 2.5%, and 1% rejection levels.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="As of SciPy 1.17.*",
            category=FutureWarning,
        )
        result = stats.anderson(sample, dist="norm")

    available = {
        float(level): float(critical)
        for level, critical in zip(
            result.significance_level,
            result.critical_values,
        )
    }
    critical_values = np.full(ANDERSON_LEVELS.shape, np.nan)
    rejection_flags = np.full(ANDERSON_LEVELS.shape, np.nan)
    for index, requested_level in enumerate(ANDERSON_LEVELS):
        for available_level, critical_value in available.items():
            if np.isclose(requested_level, available_level):
                critical_values[index] = critical_value
                rejection_flags[index] = float(result.statistic > critical_value)
                break

    selected_level = _nearest_anderson_level(alpha)
    selected_index = int(np.argmin(np.abs(ANDERSON_LEVELS - selected_level)))
    return {
        "statistic": float(result.statistic),
        "significance_levels": ANDERSON_LEVELS.copy(),
        "critical_values": critical_values,
        "rejection_flags": rejection_flags,
        "selected_level": selected_level,
        "selected_reject": rejection_flags[selected_index],
    }


def _prepare_ensemble_matrix(
    data: Any,
    member_dim: int | str = 0,
    min_valid_fraction: float = 0.95,
    center: bool = True,
    standardize_features: bool = False,
) -> _PreparedEnsemble:
    """Convert an ensemble field to a clean ``(member, feature)`` matrix.

    Features below ``min_valid_fraction`` are discarded. NaNs in retained
    features are replaced by feature means. Constant features are discarded.
    Standardization divides each retained feature by its sample standard
    deviation and necessarily centers it, even when ``center=False``.
    """

    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")

    is_xarray = xr is not None and isinstance(data, xr.DataArray)
    if is_xarray:
        if isinstance(member_dim, str):
            if member_dim not in data.dims:
                raise ValueError(f"Unknown member dimension: {member_dim!r}")
            member_axis = data.get_axis_num(member_dim)
            member_name = member_dim
        else:
            member_axis = _normalize_axis_index(member_dim, data.ndim)
            member_name = data.dims[member_axis]

        values = np.asarray(data.values, dtype=float)
        feature_dims = tuple(dim for dim in data.dims if dim != member_name)
        feature_coordinates = {
            dim: data.coords[dim]
            for dim in feature_dims
            if dim in data.coords
        }
        member_coordinate = (
            data.coords[member_name]
            if member_name in data.coords
            else np.arange(data.sizes[member_name])
        )
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim < 2:
            raise ValueError("data must include member and feature dimensions.")
        if isinstance(member_dim, str):
            raise TypeError("member_dim must be an integer for NumPy input.")
        member_axis = _normalize_axis_index(member_dim, values.ndim)
        member_name = None
        member_coordinate = np.arange(values.shape[member_axis])
        feature_dims = None
        feature_coordinates = None

    member_first = np.moveaxis(values, member_axis, 0)
    n_members = member_first.shape[0]
    if n_members < 3:
        raise ValueError("At least three ensemble members are required.")

    original_feature_shape = member_first.shape[1:]
    flattened = member_first.reshape(n_members, -1)
    finite_fraction = np.mean(np.isfinite(flattened), axis=0)
    valid_flat_mask = finite_fraction >= min_valid_fraction

    if not np.any(valid_flat_mask):
        raise ValueError(
            "No features satisfy min_valid_fraction; lower the threshold "
            "or inspect the missing data."
        )

    retained = flattened[:, valid_flat_mask]
    feature_means = np.nanmean(retained, axis=0)
    if not np.all(np.isfinite(feature_means)):
        raise ValueError("Retained features contain no finite ensemble values.")

    missing_rows, missing_columns = np.where(~np.isfinite(retained))
    retained = retained.copy()
    retained[missing_rows, missing_columns] = feature_means[missing_columns]

    feature_std = np.std(retained, axis=0, ddof=1)
    nonconstant = np.isfinite(feature_std) & (
        feature_std > np.finfo(float).eps
    )
    if np.count_nonzero(nonconstant) < 2:
        raise ValueError(
            "Too few nonconstant valid features remain for PCA (need at least 2)."
        )

    selected_flat_indices = np.flatnonzero(valid_flat_mask)[nonconstant]
    final_flat_mask = np.zeros(flattened.shape[1], dtype=bool)
    final_flat_mask[selected_flat_indices] = True

    retained = retained[:, nonconstant]
    feature_means = feature_means[nonconstant]
    feature_std = feature_std[nonconstant]

    # PCA standardization convention includes centering.
    if center or standardize_features:
        retained = retained - feature_means
    feature_scales = feature_std if standardize_features else np.ones_like(feature_std)
    if standardize_features:
        retained = retained / feature_scales

    return _PreparedEnsemble(
        matrix=retained,
        valid_feature_mask=final_flat_mask.reshape(original_feature_shape),
        original_feature_shape=original_feature_shape,
        feature_means=feature_means,
        feature_scales=feature_scales,
        member_axis=member_axis,
        member_name=member_name,
        member_coordinate=member_coordinate,
        feature_dims=feature_dims,
        feature_coordinates=feature_coordinates,
        input_was_xarray=is_xarray,
    )


def _compute_pca_svd(
    X: np.ndarray,
    n_components: int = 20,
) -> dict[str, np.ndarray]:
    """Compute PCA with economy SVD without forming a covariance matrix."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_members, n_features).")
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        raise ValueError("The ensemble anomaly matrix has zero total variance.")

    tolerance = (
        np.finfo(float).eps
        * max(X.shape)
        * singular_values[0]
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    if numerical_rank == 0:
        raise ValueError("The ensemble anomaly matrix has zero numerical rank.")

    retained_components = min(n_components, numerical_rank)
    n_members = X.shape[0]
    all_explained_variance = singular_values**2 / (n_members - 1)
    total_variance = float(np.sum(all_explained_variance))
    if total_variance <= 0:
        raise ValueError("The ensemble anomaly matrix has zero total variance.")

    explained_variance = all_explained_variance[:retained_components]
    explained_variance_ratio = explained_variance / total_variance
    scores = (
        U[:, :retained_components]
        * singular_values[:retained_components]
    )

    return {
        "scores": scores,
        "eofs_valid": Vt[:retained_components],
        "singular_values": singular_values[:retained_components],
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": np.cumsum(
            explained_variance_ratio
        ),
        "total_variance": np.array(total_variance),
        "numerical_rank": np.array(numerical_rank),
    }


def _projection_diagnostics(
    scores: np.ndarray,
    alpha: float = 0.05,
    compute_qq_r2: bool = True,
    skew_threshold: float | None = 0.5,
    kurtosis_threshold: float | None = 1.0,
    qq_r2_threshold: float | None = 0.98,
) -> dict[str, np.ndarray]:
    """Compute univariate diagnostics for every PC score distribution."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (member, component).")

    n_components = scores.shape[1]
    output = {
        "pc_mean": np.full(n_components, np.nan),
        "pc_std": np.full(n_components, np.nan),
        "pc_skewness": np.full(n_components, np.nan),
        "pc_excess_kurtosis": np.full(n_components, np.nan),
        "anderson_statistic": np.full(n_components, np.nan),
        "anderson_critical_values": np.full(
            (n_components, ANDERSON_LEVELS.size),
            np.nan,
        ),
        "anderson_rejection_flags": np.full(
            (n_components, ANDERSON_LEVELS.size),
            np.nan,
        ),
        "qq_r2": np.full(n_components, np.nan),
        "gaussian_flag": np.full(n_components, np.nan),
    }

    selected_level = _nearest_anderson_level(alpha)
    selected_index = int(np.argmin(np.abs(ANDERSON_LEVELS - selected_level)))
    for component in range(n_components):
        sample = scores[:, component]
        output["pc_mean"][component] = np.mean(sample)
        output["pc_std"][component] = np.std(sample, ddof=1)
        output["pc_skewness"][component] = stats.skew(sample, bias=False)
        output["pc_excess_kurtosis"][component] = stats.kurtosis(
            sample,
            fisher=True,
            bias=False,
        )

        anderson = _anderson_diagnostics(sample, alpha)
        output["anderson_statistic"][component] = anderson["statistic"]
        output["anderson_critical_values"][component] = anderson[
            "critical_values"
        ]
        output["anderson_rejection_flags"][component] = anderson[
            "rejection_flags"
        ]
        if compute_qq_r2:
            output["qq_r2"][component] = _qq_r2(sample)

        criteria = [
            not bool(
                output["anderson_rejection_flags"][
                    component, selected_index
                ]
            )
        ]
        if skew_threshold is not None:
            criteria.append(
                abs(output["pc_skewness"][component]) <= skew_threshold
            )
        if kurtosis_threshold is not None:
            criteria.append(
                abs(output["pc_excess_kurtosis"][component])
                <= kurtosis_threshold
            )
        if compute_qq_r2 and qq_r2_threshold is not None:
            criteria.append(
                output["qq_r2"][component] >= qq_r2_threshold
            )

        # Operational compatibility flag, not a proof of Gaussianity.
        output["gaussian_flag"][component] = float(all(criteria))

    for index, level in enumerate(ANDERSON_LEVELS):
        suffix = LEVEL_SUFFIXES[float(level)]
        output[f"anderson_reject_{suffix}"] = output[
            "anderson_rejection_flags"
        ][:, index]
        output[f"anderson_critical_{suffix}"] = output[
            "anderson_critical_values"
        ][:, index]

    output["anderson_significance_levels"] = ANDERSON_LEVELS.copy()
    output["anderson_selected_level"] = np.array(selected_level)
    return output


def _restore_eofs(
    eofs_valid: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> np.ndarray:
    """Restore EOF vectors to their original non-member feature shape."""

    n_components = eofs_valid.shape[0]
    restored = np.full(
        (n_components, *valid_feature_mask.shape),
        np.nan,
        dtype=float,
    )
    restored_flat = restored.reshape(n_components, -1)
    restored_flat[:, valid_feature_mask.ravel()] = eofs_valid
    return restored


def _restore_feature_values(
    values_valid: np.ndarray,
    valid_feature_mask: np.ndarray,
) -> np.ndarray:
    """Restore feature values to the original non-member shape."""

    restored = np.full(valid_feature_mask.shape, np.nan, dtype=float)
    restored.ravel()[valid_feature_mask.ravel()] = values_valid
    return restored


def _format_output_as_xarray_or_dict(
    prepared: _PreparedEnsemble,
    pca: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    *,
    alpha: float,
    center: bool,
    standardize_features: bool,
    min_valid_fraction: float,
    skew_threshold: float | None,
    kurtosis_threshold: float | None,
    qq_r2_threshold: float | None,
    return_eofs: bool,
    return_scores: bool,
) -> dict[str, Any] | Any:
    """Format PCA diagnostics as a dictionary or xarray Dataset."""

    n_components = pca["scores"].shape[1]
    eofs = _restore_eofs(
        pca["eofs_valid"],
        prepared.valid_feature_mask,
    )
    feature_mean_field = _restore_feature_values(
        prepared.feature_means,
        prepared.valid_feature_mask,
    )
    feature_scale_field = _restore_feature_values(
        prepared.feature_scales,
        prepared.valid_feature_mask,
    )
    metadata = {
        "alpha": alpha,
        "anderson_alpha_used": _nearest_anderson_level(alpha) / 100,
        "center": center,
        "standardize_features": standardize_features,
        "min_valid_fraction": min_valid_fraction,
        "skew_threshold": skew_threshold,
        "kurtosis_threshold": kurtosis_threshold,
        "qq_r2_threshold": qq_r2_threshold,
        "n_members": prepared.matrix.shape[0],
        "n_valid_features": int(np.sum(prepared.valid_feature_mask)),
        "numerical_rank": int(pca["numerical_rank"]),
        "interpretation": (
            "gaussian_flag=1 means compatible with Gaussianity under the "
            "configured projection diagnostics; it is not proof of "
            "multivariate normality."
        ),
    }

    common = {
        "explained_variance": pca["explained_variance"],
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "cumulative_explained_variance_ratio": pca[
            "cumulative_explained_variance_ratio"
        ],
        "pc_mean": diagnostics["pc_mean"],
        "pc_std": diagnostics["pc_std"],
        "pc_skewness": diagnostics["pc_skewness"],
        "pc_excess_kurtosis": diagnostics["pc_excess_kurtosis"],
        "anderson_statistic": diagnostics["anderson_statistic"],
        "anderson_critical_values": diagnostics[
            "anderson_critical_values"
        ],
        "anderson_significance_levels": diagnostics[
            "anderson_significance_levels"
        ],
        "anderson_reject_15": diagnostics["anderson_reject_15"],
        "anderson_reject_10": diagnostics["anderson_reject_10"],
        "anderson_reject_5": diagnostics["anderson_reject_5"],
        "anderson_reject_2p5": diagnostics["anderson_reject_2p5"],
        "anderson_reject_1": diagnostics["anderson_reject_1"],
        "qq_r2": diagnostics["qq_r2"],
        "gaussian_flag": diagnostics["gaussian_flag"],
    }

    if not prepared.input_was_xarray:
        output: dict[str, Any] = dict(common)
        if return_scores:
            output["scores"] = pca["scores"]
        if return_eofs:
            output["eofs"] = eofs
        output["valid_feature_mask"] = prepared.valid_feature_mask
        output["original_feature_shape"] = prepared.original_feature_shape
        output["feature_means"] = prepared.feature_means
        output["feature_scales"] = prepared.feature_scales
        output["feature_mean_field"] = feature_mean_field
        output["feature_scale_field"] = feature_scale_field
        output["metadata"] = metadata
        return output

    component_coord = np.arange(n_components)
    level_coord = diagnostics["anderson_significance_levels"]
    data_vars: dict[str, Any] = {}
    for field, values in common.items():
        values = np.asarray(values)
        if field == "anderson_significance_levels":
            data_vars[field] = (("significance_level",), values)
        elif field == "anderson_critical_values":
            data_vars[field] = (
                ("component", "significance_level"),
                values,
            )
        else:
            data_vars[field] = (("component",), values)

    member_dim = prepared.member_name or "member"
    coords: dict[str, Any] = {
        "component": component_coord,
        "significance_level": level_coord,
    }
    if return_scores:
        coords[member_dim] = prepared.member_coordinate
        data_vars["scores"] = (
            (member_dim, "component"),
            pca["scores"],
        )
    if return_eofs:
        eof_dims = ("component", *prepared.feature_dims)
        data_vars["eofs"] = (eof_dims, eofs)
        coords.update(prepared.feature_coordinates or {})

    data_vars["valid_feature_mask"] = (
        prepared.feature_dims,
        prepared.valid_feature_mask,
    )
    data_vars["feature_mean_field"] = (
        prepared.feature_dims,
        feature_mean_field,
    )
    data_vars["feature_scale_field"] = (
        prepared.feature_dims,
        feature_scale_field,
    )
    coords.update(prepared.feature_coordinates or {})
    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset.attrs.update(metadata)
    dataset.attrs["original_feature_shape"] = prepared.original_feature_shape
    return dataset


def compute_pca_projection_gaussianity(
    data: Any,
    member_dim: int | str = 0,
    n_components: int = 20,
    alpha: float = 0.05,
    standardize_features: bool = False,
    center: bool = True,
    min_valid_fraction: float = 0.95,
    compute_qq_r2: bool = True,
    skew_threshold: float | None = 0.5,
    kurtosis_threshold: float | None = 1.0,
    qq_r2_threshold: float | None = 0.98,
    return_eofs: bool = False,
    return_scores: bool = True,
) -> dict[str, Any] | Any:
    """Assess ensemble Gaussianity through leading PCA/EOF projections."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    prepared = _prepare_ensemble_matrix(
        data,
        member_dim=member_dim,
        min_valid_fraction=min_valid_fraction,
        center=center,
        standardize_features=standardize_features,
    )
    pca = _compute_pca_svd(prepared.matrix, n_components=n_components)
    diagnostics = _projection_diagnostics(
        pca["scores"],
        alpha=alpha,
        compute_qq_r2=compute_qq_r2,
        skew_threshold=skew_threshold,
        kurtosis_threshold=kurtosis_threshold,
        qq_r2_threshold=qq_r2_threshold,
    )
    return _format_output_as_xarray_or_dict(
        prepared,
        pca,
        diagnostics,
        alpha=alpha,
        center=center,
        standardize_features=standardize_features,
        min_valid_fraction=min_valid_fraction,
        skew_threshold=skew_threshold,
        kurtosis_threshold=kurtosis_threshold,
        qq_r2_threshold=qq_r2_threshold,
        return_eofs=return_eofs,
        return_scores=return_scores,
    )


def _result_array(result: Any, field: str) -> np.ndarray:
    if isinstance(result, Mapping):
        if field not in result:
            raise KeyError(f"Result does not contain {field!r}.")
        return np.asarray(result[field])
    if xr is not None and isinstance(result, xr.Dataset):
        if field not in result:
            raise KeyError(f"Result does not contain {field!r}.")
        return np.asarray(result[field].values)
    raise TypeError("result must be a dictionary or xarray Dataset.")


def _result_metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result.get("metadata", {}))
    if xr is not None and isinstance(result, xr.Dataset):
        return dict(result.attrs)
    return {}


def plot_pca_gaussianity_summary(result: Any):
    """Plot explained variance and Gaussianity diagnostics by PC."""

    variance_ratio = _result_array(result, "explained_variance_ratio")
    cumulative = _result_array(
        result,
        "cumulative_explained_variance_ratio",
    )
    anderson = _result_array(result, "anderson_statistic")
    skewness = _result_array(result, "pc_skewness")
    kurtosis = _result_array(result, "pc_excess_kurtosis")
    qq_r2 = _result_array(result, "qq_r2")
    metadata = _result_metadata(result)
    alpha_used = float(metadata.get("anderson_alpha_used", 0.05))
    reject_field = {
        0.15: "anderson_reject_15",
        0.10: "anderson_reject_10",
        0.05: "anderson_reject_5",
        0.025: "anderson_reject_2p5",
        0.01: "anderson_reject_1",
    }.get(alpha_used, "anderson_reject_5")
    rejection = _result_array(result, reject_field)

    components = np.arange(1, variance_ratio.size + 1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)

    axes[0, 0].bar(components, variance_ratio, color="steelblue")
    axes[0, 0].set_ylabel("Explained variance ratio")
    axes[0, 0].set_title("Explained variance by component")

    axes[0, 1].plot(components, cumulative, marker="o")
    axes[0, 1].axhline(0.9, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Cumulative variance ratio")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_title("Cumulative explained variance")

    axes[1, 0].plot(components, anderson, marker="o", color="darkred")
    axes[1, 0].set_ylabel("Anderson-Darling statistic")
    axes[1, 0].set_title("Anderson-Darling departure from normality")

    axes[1, 1].bar(
        components,
        rejection,
        color=np.where(rejection > 0.5, "crimson", "royalblue"),
    )
    axes[1, 1].set_yticks([0, 1], ["Do not reject", "Reject"])
    axes[1, 1].set_title(f"Anderson rejection (alpha={alpha_used:g})")

    axes[2, 0].plot(
        components,
        skewness,
        marker="o",
        label="Skewness",
    )
    axes[2, 0].plot(
        components,
        kurtosis,
        marker="s",
        label="Excess kurtosis",
    )
    axes[2, 0].axhline(0, color="black", linewidth=0.8)
    axes[2, 0].set_ylabel("Moment diagnostic")
    axes[2, 0].set_title("Shape diagnostics")
    axes[2, 0].legend()

    axes[2, 1].plot(components, qq_r2, marker="o", color="purple")
    qq_threshold = metadata.get("qq_r2_threshold", 0.98)
    if qq_threshold is not None:
        axes[2, 1].axhline(
            qq_threshold,
            color="gray",
            linestyle="--",
            linewidth=1,
        )
    axes[2, 1].set_ylim(min(0.9, np.nanmin(qq_r2) - 0.01), 1.002)
    axes[2, 1].set_ylabel("Q-Q R²")
    axes[2, 1].set_title("Normal Q-Q linearity")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("PC component")
    fig.tight_layout()
    return fig, axes


def plot_pc_histogram_and_qq(result: Any, component: int = 0):
    """Plot a histogram and normal Q-Q plot for one PC score sample."""

    scores = _result_array(result, "scores")
    if scores.ndim != 2:
        raise ValueError("scores must have shape (member, component).")
    if component < 0 or component >= scores.shape[1]:
        raise IndexError("component is outside the available PC range.")

    sample = scores[:, component]
    mean = _result_array(result, "pc_mean")[component]
    std = _result_array(result, "pc_std")[component]
    variance_ratio = _result_array(
        result,
        "explained_variance_ratio",
    )[component]
    anderson = _result_array(result, "anderson_statistic")[component]
    skewness = _result_array(result, "pc_skewness")[component]
    kurtosis = _result_array(
        result,
        "pc_excess_kurtosis",
    )[component]
    qq_r2 = _result_array(result, "qq_r2")[component]
    metadata = _result_metadata(result)
    alpha_used = float(metadata.get("anderson_alpha_used", 0.05))
    reject_field = {
        0.15: "anderson_reject_15",
        0.10: "anderson_reject_10",
        0.05: "anderson_reject_5",
        0.025: "anderson_reject_2p5",
        0.01: "anderson_reject_1",
    }.get(alpha_used, "anderson_reject_5")
    rejected = bool(_result_array(result, reject_field)[component])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(
        sample,
        bins=30,
        density=True,
        alpha=0.7,
        color="royalblue",
        edgecolor="black",
        label=f"PC {component + 1} scores",
    )
    x_values = np.linspace(np.min(sample), np.max(sample), 300)
    axes[0].plot(
        x_values,
        stats.norm.pdf(x_values, mean, std),
        color="darkred",
        linestyle="--",
        linewidth=2,
        label="Fitted Gaussian",
    )
    axes[0].set_title("(a) PC score histogram")
    axes[0].set_xlabel("PC score")
    axes[0].set_ylabel("Probability density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    stats.probplot(sample, dist="norm", plot=axes[1])
    axes[1].set_title("(b) Normal Q-Q plot")
    axes[1].grid(alpha=0.3)

    interpretation = (
        "evidence against Gaussianity"
        if rejected
        else "compatible with Gaussianity at selected AD level"
    )
    fig.suptitle(
        f"PC {component + 1}: variance={variance_ratio:.2%}, "
        f"AD={anderson:.3f} ({interpretation}), "
        f"skew={skewness:+.3f}, excess kurtosis={kurtosis:+.3f}, "
        f"Q-Q R²={qq_r2:.5f}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes


def plot_eof_pattern(
    result: Any,
    component: int = 0,
    selection: Any = None,
    *,
    ax=None,
    cmap: str = "RdBu_r",
):
    """Plot an EOF pattern, optionally slicing dimensions above two.

    For EOFs with more than two feature dimensions, ``selection`` may be a
    tuple applied after the component dimension, for example ``(0, slice(None),
    slice(None))`` for ``(level, y, x)`` EOFs. If omitted, leading dimensions
    are indexed at zero until a two-dimensional field remains.
    """

    eofs = _result_array(result, "eofs")
    if component < 0 or component >= eofs.shape[0]:
        raise IndexError("component is outside the available EOF range.")
    pattern = eofs[component]
    if selection is not None:
        pattern = pattern[selection]
    while pattern.ndim > 2:
        pattern = pattern[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if pattern.ndim == 2:
        image = ax.imshow(
            pattern,
            origin="lower",
            cmap=cmap,
            aspect="auto",
        )
        ax.figure.colorbar(image, ax=ax, label="EOF loading")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    elif pattern.ndim == 1:
        ax.plot(pattern)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("EOF loading")
    else:
        raise ValueError("Selected EOF pattern must be one- or two-dimensional.")
    ax.set_title(f"EOF pattern: component {component + 1}")
    return ax


def reconstruct_pca_field(
    result: Any,
    member: int,
    n_components: int | None = None,
    *,
    add_mean: bool = True,
) -> np.ndarray:
    """Reconstruct one ensemble field from retained PCA components.

    PCA is fitted to centered anomalies by default. Therefore, a physical
    field reconstruction is

    ``ensemble_mean + sum(score_k * EOF_k)``.

    With all nonzero components, this recovers the mean-filled retained
    features up to numerical precision. A truncated reconstruction recovers
    only the variance represented by those components.
    """

    scores = _result_array(result, "scores")
    eofs = _result_array(result, "eofs")
    valid_mask = _result_array(result, "valid_feature_mask").astype(bool)
    feature_mean = _result_array(result, "feature_mean_field")
    feature_scale = _result_array(result, "feature_scale_field")
    metadata = _result_metadata(result)

    if member < 0 or member >= scores.shape[0]:
        raise IndexError("member is outside the available ensemble range.")
    if n_components is None:
        n_components = scores.shape[1]
    if n_components < 1 or n_components > scores.shape[1]:
        raise ValueError(
            f"n_components must be between 1 and {scores.shape[1]}."
        )

    valid_eofs = eofs[:n_components].reshape(n_components, -1)[
        :, valid_mask.ravel()
    ]
    reconstructed_valid = scores[member, :n_components] @ valid_eofs

    if bool(metadata.get("standardize_features", False)):
        reconstructed_valid = (
            reconstructed_valid
            * feature_scale.ravel()[valid_mask.ravel()]
        )

    # Centering is implicit when standardizing features.
    was_centered = bool(metadata.get("center", True)) or bool(
        metadata.get("standardize_features", False)
    )
    if add_mean and was_centered:
        reconstructed_valid = (
            reconstructed_valid
            + feature_mean.ravel()[valid_mask.ravel()]
        )

    reconstruction = np.full(valid_mask.shape, np.nan, dtype=float)
    reconstruction.ravel()[valid_mask.ravel()] = reconstructed_valid
    return reconstruction


def plot_pca_reconstruction(
    result: Any,
    original_data: Any,
    member: int = 0,
    n_components: int = 6,
    member_dim: int | str = 0,
    selection: Any = None,
    *,
    cmap: str = "RdBu_r",
    title: str | None = None,
):
    """Plot original field, ensemble mean, PCA reconstruction, and residual."""

    original = np.asarray(original_data)
    if xr is not None and isinstance(original_data, xr.DataArray):
        if isinstance(member_dim, str):
            member_axis = original_data.get_axis_num(member_dim)
        else:
            member_axis = _normalize_axis_index(member_dim, original_data.ndim)
    else:
        if isinstance(member_dim, str):
            raise TypeError("member_dim must be an integer for NumPy input.")
        member_axis = _normalize_axis_index(member_dim, original.ndim)

    member_first = np.moveaxis(original, member_axis, 0)
    if member < 0 or member >= member_first.shape[0]:
        raise IndexError("member is outside the available ensemble range.")

    original_field = np.asarray(member_first[member], dtype=float)
    mean_field = _result_array(result, "feature_mean_field")
    reconstruction = reconstruct_pca_field(
        result,
        member=member,
        n_components=n_components,
        add_mean=True,
    )

    if selection is not None:
        original_field = original_field[selection]
        mean_field = mean_field[selection]
        reconstruction = reconstruction[selection]
    while original_field.ndim > 2:
        original_field = original_field[0]
        mean_field = mean_field[0]
        reconstruction = reconstruction[0]
    if original_field.ndim != 2:
        raise ValueError("Selected reconstruction fields must be two-dimensional.")

    residual = original_field - reconstruction
    valid_mask = np.isfinite(original_field) & np.isfinite(reconstruction)
    rmse = float(np.sqrt(np.mean(residual[valid_mask] ** 2)))
    original_variance = float(np.var(original_field[valid_mask]))
    residual_variance = float(np.var(residual[valid_mask]))
    captured_fraction = (
        1.0 - residual_variance / original_variance
        if original_variance > 0
        else np.nan
    )

    field_limit = float(
        np.nanpercentile(
            np.abs(
                np.concatenate(
                    [
                        original_field.ravel(),
                        mean_field.ravel(),
                        reconstruction.ravel(),
                    ]
                )
            ),
            99,
        )
    )
    residual_limit = float(np.nanpercentile(np.abs(residual), 99))
    field_limit = field_limit or 1.0
    residual_limit = residual_limit or 1.0

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
    )
    fields = [
        (original_field, f"Original PV: member {member}", field_limit),
        (mean_field, "Ensemble-mean PV", field_limit),
        (
            reconstruction,
            f"Mean + first {n_components} PCs",
            field_limit,
        ),
        (
            residual,
            f"Residual: original − reconstruction\nRMSE={rmse:.3f}",
            residual_limit,
        ),
    ]
    images = []
    for ax, (field, panel_title, limit) in zip(axes.flat, fields):
        image = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        images.append(image)
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    fig.colorbar(
        images[0],
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        shrink=0.8,
        label="PV",
    )
    fig.colorbar(
        images[3],
        ax=axes[1, 1],
        shrink=0.8,
        label="PV residual",
    )
    if title is None:
        title = (
            f"PCA reconstruction with {n_components} components "
            f"(member {member}, captured sample variance≈{captured_fraction:.1%})"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig, axes


def plot_leading_eof_patterns(
    result: Any,
    n_components: int = 6,
    selection: Any = None,
    *,
    ncols: int = 3,
    cmap: str = "RdBu_r",
    color_percentile: float = 99.0,
    title: str = "Leading PCA/EOF patterns",
):
    """Plot the leading EOF patterns ranked by explained variance.

    Each panel reports the explained variance ratio and whether the
    Anderson-Darling test rejects Gaussianity at the configured significance
    level. A shared symmetric color scale makes EOF amplitudes comparable.

    For EOFs with more than two feature dimensions, ``selection`` follows the
    same convention as :func:`plot_eof_pattern`.
    """

    eofs = _result_array(result, "eofs")
    variance_ratio = _result_array(result, "explained_variance_ratio")
    metadata = _result_metadata(result)
    alpha_used = float(metadata.get("anderson_alpha_used", 0.05))
    reject_field = {
        0.15: "anderson_reject_15",
        0.10: "anderson_reject_10",
        0.05: "anderson_reject_5",
        0.025: "anderson_reject_2p5",
        0.01: "anderson_reject_1",
    }.get(alpha_used, "anderson_reject_5")
    rejection = _result_array(result, reject_field)

    n = min(n_components, eofs.shape[0])
    if n < 1:
        raise ValueError("At least one EOF component is required.")
    if ncols < 1:
        raise ValueError("ncols must be at least 1.")

    patterns = []
    for component in range(n):
        pattern = eofs[component]
        if selection is not None:
            pattern = pattern[selection]
        while pattern.ndim > 2:
            pattern = pattern[0]
        if pattern.ndim != 2:
            raise ValueError(
                "Leading EOF summary requires a two-dimensional selected pattern."
            )
        patterns.append(pattern)

    if not 0 < color_percentile <= 100:
        raise ValueError("color_percentile must be in (0, 100].")
    finite_amplitudes = np.concatenate(
        [np.abs(pattern[np.isfinite(pattern)]) for pattern in patterns]
    )
    color_limit = float(np.percentile(finite_amplitudes, color_percentile))
    if color_limit == 0:
        color_limit = 1.0

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for component, (ax, pattern) in enumerate(
        zip(axes.flat, patterns)
    ):
        image = ax.imshow(
            pattern,
            origin="lower",
            cmap=cmap,
            vmin=-color_limit,
            vmax=color_limit,
            aspect="auto",
        )
        status = (
            "AD reject"
            if rejection[component] > 0.5
            else "AD compatible"
        )
        status_color = "crimson" if rejection[component] > 0.5 else "darkgreen"
        ax.set_title(
            f"PC {component + 1}: {variance_ratio[component]:.2%} variance\n"
            f"{status} at α={alpha_used:g}",
            color=status_color,
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    if image is not None:
        fig.colorbar(
            image,
            ax=list(axes.flat[:n]),
            shrink=0.78,
            pad=0.02,
            location="right",
            label="EOF loading",
        )
    return fig, axes


def compare_pca_gaussianity_results(
    result_ref: Any,
    result_test: Any,
) -> dict[str, np.ndarray] | Any:
    """Compare PCA Gaussianity diagnostics from a reference and test ensemble."""

    n_components = min(
        _result_array(result_ref, "explained_variance_ratio").size,
        _result_array(result_test, "explained_variance_ratio").size,
    )
    metadata_ref = _result_metadata(result_ref)
    alpha_used = float(metadata_ref.get("anderson_alpha_used", 0.05))
    reject_field = {
        0.15: "anderson_reject_15",
        0.10: "anderson_reject_10",
        0.05: "anderson_reject_5",
        0.025: "anderson_reject_2p5",
        0.01: "anderson_reject_1",
    }.get(alpha_used, "anderson_reject_5")

    fields = {
        "delta_explained_variance_ratio": (
            _result_array(result_test, "explained_variance_ratio")[:n_components]
            - _result_array(result_ref, "explained_variance_ratio")[:n_components]
        ),
        "delta_anderson_statistic": (
            _result_array(result_test, "anderson_statistic")[:n_components]
            - _result_array(result_ref, "anderson_statistic")[:n_components]
        ),
        "delta_skewness": (
            _result_array(result_test, "pc_skewness")[:n_components]
            - _result_array(result_ref, "pc_skewness")[:n_components]
        ),
        "delta_excess_kurtosis": (
            _result_array(
                result_test,
                "pc_excess_kurtosis",
            )[:n_components]
            - _result_array(
                result_ref,
                "pc_excess_kurtosis",
            )[:n_components]
        ),
        "delta_qq_r2": (
            _result_array(result_test, "qq_r2")[:n_components]
            - _result_array(result_ref, "qq_r2")[:n_components]
        ),
        "gaussian_flag_ref": _result_array(
            result_ref,
            "gaussian_flag",
        )[:n_components],
        "gaussian_flag_test": _result_array(
            result_test,
            "gaussian_flag",
        )[:n_components],
        "anderson_reject_ref": _result_array(
            result_ref,
            reject_field,
        )[:n_components],
        "anderson_reject_test": _result_array(
            result_test,
            reject_field,
        )[:n_components],
    }
    fields["compatible_fraction_ref"] = np.array(
        np.nanmean(fields["gaussian_flag_ref"])
    )
    fields["compatible_fraction_test"] = np.array(
        np.nanmean(fields["gaussian_flag_test"])
    )

    if (
        xr is not None
        and isinstance(result_ref, xr.Dataset)
        and isinstance(result_test, xr.Dataset)
    ):
        return xr.Dataset(
            {
                field: (("component",), values)
                for field, values in fields.items()
                if np.asarray(values).ndim == 1
            },
            coords={"component": np.arange(n_components)},
            attrs={
                "compatible_fraction_ref": float(
                    fields["compatible_fraction_ref"]
                ),
                "compatible_fraction_test": float(
                    fields["compatible_fraction_test"]
                ),
            },
        )
    return fields


def summarize_pca_gaussianity(
    result: Any,
    n_leading: int = 10,
) -> dict[str, Any]:
    """Print and return a concise interpretation of leading PC diagnostics."""

    variance_ratio = _result_array(result, "explained_variance_ratio")
    cumulative = _result_array(
        result,
        "cumulative_explained_variance_ratio",
    )
    anderson = _result_array(result, "anderson_statistic")
    gaussian_flag = _result_array(result, "gaussian_flag")
    metadata = _result_metadata(result)
    alpha_used = float(metadata.get("anderson_alpha_used", 0.05))
    reject_field = {
        0.15: "anderson_reject_15",
        0.10: "anderson_reject_10",
        0.05: "anderson_reject_5",
        0.025: "anderson_reject_2p5",
        0.01: "anderson_reject_1",
    }.get(alpha_used, "anderson_reject_5")
    rejection = _result_array(result, reject_field)

    n = min(n_leading, variance_ratio.size)
    rejected_indices = np.flatnonzero(rejection[:n] > 0.5)
    incompatible_indices = np.flatnonzero(gaussian_flag[:n] < 0.5)
    strongest = np.argsort(anderson[:n])[::-1][: min(3, n)]
    broadly_compatible = incompatible_indices.size <= max(1, int(0.2 * n))

    summary = {
        "n_leading": n,
        "explained_variance_fraction": float(cumulative[n - 1]),
        "anderson_rejected_count": int(rejected_indices.size),
        "anderson_rejected_components": (rejected_indices + 1).tolist(),
        "compatible_count": int(np.sum(gaussian_flag[:n] > 0.5)),
        "strongest_non_gaussian_components": (strongest + 1).tolist(),
        "broadly_compatible": broadly_compatible,
    }

    print(
        f"First {n} PCs explain "
        f"{summary['explained_variance_fraction']:.1%} of ensemble variance."
    )
    print(
        f"Anderson-Darling rejects Gaussianity for "
        f"{summary['anderson_rejected_count']}/{n} leading PCs "
        f"at alpha={alpha_used:g}."
    )
    print(
        "Strongest Anderson-Darling departures: PCs "
        + ", ".join(map(str, summary["strongest_non_gaussian_components"]))
        + "."
    )
    if broadly_compatible:
        print(
            "The dominant ensemble subspace appears broadly compatible with "
            "Gaussianity under these diagnostics, without proving full "
            "multivariate normality."
        )
    else:
        print(
            "Several dominant projections show evidence against Gaussianity; "
            "the leading ensemble subspace is not broadly compatible with "
            "Gaussianity under these diagnostics."
        )
    return summary


def _synthetic_ensemble_fields(
    seed: int = 42,
    n_members: int = 1000,
    shape: tuple[int, int] = (16, 16),
) -> tuple[np.ndarray, np.ndarray]:
    """Create correlated Gaussian and regime-mixture synthetic fields."""

    rng = np.random.default_rng(seed)
    y = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
    x = np.linspace(0, 2 * np.pi, shape[1], endpoint=False)
    xx, yy = np.meshgrid(x, y)
    modes = np.stack(
        [
            np.sin(xx),
            np.cos(yy),
            np.sin(xx + yy),
            np.cos(2 * xx - yy),
            np.sin(2 * yy),
        ]
    )
    modes = modes.reshape(modes.shape[0], -1)
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)

    gaussian_coefficients = rng.normal(
        scale=[4.0, 3.0, 2.0, 1.5, 1.0],
        size=(n_members, modes.shape[0]),
    )
    gaussian = gaussian_coefficients @ modes
    gaussian += rng.normal(scale=0.2, size=gaussian.shape)

    non_gaussian_coefficients = gaussian_coefficients.copy()
    regimes = rng.choice([-1.0, 1.0], size=n_members)
    non_gaussian_coefficients[:, 0] = (
        5.0 * regimes + rng.normal(scale=0.7, size=n_members)
    )
    non_gaussian_coefficients[:, 1] = (
        rng.lognormal(mean=0.0, sigma=0.6, size=n_members)
        - np.exp(0.6**2 / 2)
    ) * 2.0
    non_gaussian = non_gaussian_coefficients @ modes
    non_gaussian += rng.normal(scale=0.2, size=non_gaussian.shape)

    return (
        gaussian.reshape(n_members, *shape),
        non_gaussian.reshape(n_members, *shape),
    )


def _print_component_table(result: Mapping[str, Any], n: int = 5) -> None:
    print("PC  VarRatio   AD stat   Reject5%   Skew    ExKurt   Q-Q R2")
    for component in range(min(n, len(result["explained_variance_ratio"]))):
        print(
            f"{component + 1:>2}  "
            f"{result['explained_variance_ratio'][component]:>8.3f}  "
            f"{result['anderson_statistic'][component]:>8.3f}  "
            f"{int(result['anderson_reject_5'][component]):>8}  "
            f"{result['pc_skewness'][component]:>7.3f}  "
            f"{result['pc_excess_kurtosis'][component]:>7.3f}  "
            f"{result['qq_r2'][component]:>7.4f}"
        )


if __name__ == "__main__":
    gaussian_data, non_gaussian_data = _synthetic_ensemble_fields()
    gaussian_result = compute_pca_projection_gaussianity(
        gaussian_data,
        n_components=8,
        return_eofs=True,
    )
    non_gaussian_result = compute_pca_projection_gaussianity(
        non_gaussian_data,
        n_components=8,
        return_eofs=True,
    )

    print("\nGaussian correlated ensemble")
    _print_component_table(gaussian_result)
    summarize_pca_gaussianity(gaussian_result, n_leading=8)

    print("\nNon-Gaussian regime-mixture ensemble")
    _print_component_table(non_gaussian_result)
    summarize_pca_gaussianity(non_gaussian_result, n_leading=8)

    plot_pca_gaussianity_summary(gaussian_result)
    plot_pca_gaussianity_summary(non_gaussian_result)
    plot_pc_histogram_and_qq(non_gaussian_result, component=0)
    plot_eof_pattern(non_gaussian_result, component=0)
    plt.show()
