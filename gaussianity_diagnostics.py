"""Pointwise Gaussianity diagnostics for ensemble fields.

The routines in this module describe samples as *compatible with Gaussianity*
or as showing *evidence against Gaussianity*. Statistical tests do not prove
that a distribution is Gaussian.

Flags are returned as floating-point arrays:

* ``1.0`` means True/reject/compatible, depending on the field.
* ``0.0`` means False/do not reject/not compatible.
* ``NaN`` means that the diagnostic was not available (for example, because
  fewer than ``min_samples`` valid members were present or variance was zero).

This representation preserves missing diagnostic information, unlike a plain
Boolean array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:  # xarray is an optional dependency.
    import xarray as xr
except ImportError:  # pragma: no cover - exercised when xarray is unavailable.
    xr = None


ANDERSON_LEVELS = (15.0, 10.0, 5.0, 2.5, 1.0)
OUTPUT_FIELDS = (
    "mean",
    "std",
    "skewness",
    "excess_kurtosis",
    "shapiro_pvalue",
    "anderson_statistic",
    "anderson_reject_15",
    "anderson_reject_10",
    "anderson_reject_5",
    "anderson_reject_2p5",
    "anderson_reject_1",
    "qq_r2",
    "gaussian_flag",
)


def _normalize_axis_index(axis: int, ndim: int) -> int:
    """Normalize a possibly negative axis without relying on NumPy internals."""

    if not isinstance(axis, (int, np.integer)):
        raise TypeError("member_dim must be an integer axis or dimension name.")
    if axis < -ndim or axis >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return int(axis % ndim)


def _qq_r2(sample: np.ndarray) -> float:
    """Return R² for a normal Q-Q plot linear fit.

    Values close to one indicate that empirical sample quantiles align well
    with theoretical Gaussian quantiles.
    """

    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2 or np.ptp(sample) == 0:
        return np.nan

    _, (_, _, correlation) = stats.probplot(sample, dist="norm", fit=True)
    return float(correlation**2)


def _anderson_rejection_flags(
    sample: np.ndarray,
) -> tuple[float, dict[float, float], dict[float, float]]:
    """Compute the Anderson-Darling normality statistic and rejection flags.

    A flag is True when the statistic is larger than SciPy's critical value at
    that significance level, which is evidence against Gaussianity.
    """

    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2 or np.ptp(sample) == 0:
        missing = {level: np.nan for level in ANDERSON_LEVELS}
        return np.nan, missing.copy(), missing

    # SciPy 1.17+ warns that this critical-value API will eventually change.
    # We intentionally use it because the requested diagnostics require the
    # tabulated 15%, 10%, 5%, 2.5%, and 1% rejection thresholds.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="As of SciPy 1.17.*",
            category=FutureWarning,
        )
        result = stats.anderson(sample, dist="norm")
    critical_values = {
        float(level): float(critical)
        for level, critical in zip(result.significance_level, result.critical_values)
    }

    flags: dict[float, float] = {}
    selected_critical_values: dict[float, float] = {}
    for requested_level in ANDERSON_LEVELS:
        available_level = min(
            critical_values,
            key=lambda level: abs(level - requested_level),
        )
        if not np.isclose(available_level, requested_level):
            flags[requested_level] = np.nan
            selected_critical_values[requested_level] = np.nan
            continue

        critical = critical_values[available_level]
        flags[requested_level] = float(result.statistic > critical)
        selected_critical_values[requested_level] = critical

    return float(result.statistic), flags, selected_critical_values


def _nearest_anderson_level(alpha: float) -> float:
    """Return the SciPy Anderson-Darling level nearest to ``alpha``."""

    alpha_percent = 100.0 * alpha
    return min(ANDERSON_LEVELS, key=lambda level: abs(level - alpha_percent))


def _pointwise_diagnostics(
    sample: np.ndarray,
    *,
    alpha: float = 0.05,
    min_samples: int = 20,
    use_shapiro: bool = True,
    compute_qq_r2: bool = True,
    skew_threshold: float | None = 0.5,
    kurtosis_threshold: float | None = 1.0,
    qq_r2_threshold: float | None = 0.98,
) -> dict[str, float]:
    """Compute all requested diagnostics for one ensemble sample."""

    sample = np.asarray(sample, dtype=float)
    valid = sample[np.isfinite(sample)]
    missing = {field: np.nan for field in OUTPUT_FIELDS}

    if valid.size < min_samples or np.ptp(valid) == 0:
        return missing

    diagnostics = {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "skewness": float(stats.skew(valid, bias=False)),
        "excess_kurtosis": float(
            stats.kurtosis(valid, fisher=True, bias=False)
        ),
        "shapiro_pvalue": np.nan,
        "anderson_statistic": np.nan,
        "anderson_reject_15": np.nan,
        "anderson_reject_10": np.nan,
        "anderson_reject_5": np.nan,
        "anderson_reject_2p5": np.nan,
        "anderson_reject_1": np.nan,
        "qq_r2": np.nan,
        "gaussian_flag": np.nan,
    }

    if use_shapiro:
        diagnostics["shapiro_pvalue"] = float(stats.shapiro(valid).pvalue)

    anderson_statistic, anderson_flags, _ = _anderson_rejection_flags(valid)
    diagnostics["anderson_statistic"] = anderson_statistic
    diagnostics["anderson_reject_15"] = anderson_flags[15.0]
    diagnostics["anderson_reject_10"] = anderson_flags[10.0]
    diagnostics["anderson_reject_5"] = anderson_flags[5.0]
    diagnostics["anderson_reject_2p5"] = anderson_flags[2.5]
    diagnostics["anderson_reject_1"] = anderson_flags[1.0]

    if compute_qq_r2:
        diagnostics["qq_r2"] = _qq_r2(valid)

    criteria: list[bool] = []
    if use_shapiro:
        criteria.append(diagnostics["shapiro_pvalue"] > alpha)

    anderson_level = _nearest_anderson_level(alpha)
    anderson_key = {
        15.0: "anderson_reject_15",
        10.0: "anderson_reject_10",
        5.0: "anderson_reject_5",
        2.5: "anderson_reject_2p5",
        1.0: "anderson_reject_1",
    }[anderson_level]
    anderson_reject = diagnostics[anderson_key]
    if np.isfinite(anderson_reject):
        criteria.append(not bool(anderson_reject))

    if skew_threshold is not None:
        criteria.append(abs(diagnostics["skewness"]) <= skew_threshold)
    if kurtosis_threshold is not None:
        criteria.append(
            abs(diagnostics["excess_kurtosis"]) <= kurtosis_threshold
        )
    if compute_qq_r2 and qq_r2_threshold is not None:
        criteria.append(diagnostics["qq_r2"] >= qq_r2_threshold)

    diagnostics["gaussian_flag"] = float(all(criteria)) if criteria else np.nan
    return diagnostics


@dataclass(frozen=True)
class GaussianityDiagnostics:
    """Configuration and methods for pointwise ensemble diagnostics."""

    alpha: float = 0.05
    min_samples: int = 20
    use_shapiro: bool = True
    compute_qq_r2: bool = True
    skew_threshold: float | None = 0.5
    kurtosis_threshold: float | None = 1.0
    qq_r2_threshold: float | None = 0.98

    def compute(
        self,
        data: Any,
        member_dim: int | str = 0,
    ) -> dict[str, np.ndarray] | Any:
        """Compute diagnostics along the ensemble/member dimension."""

        return compute_pointwise_gaussianity(
            data,
            member_dim=member_dim,
            alpha=self.alpha,
            min_samples=self.min_samples,
            use_shapiro=self.use_shapiro,
            compute_qq_r2=self.compute_qq_r2,
            skew_threshold=self.skew_threshold,
            kurtosis_threshold=self.kurtosis_threshold,
            qq_r2_threshold=self.qq_r2_threshold,
        )

    @staticmethod
    def plot_gaussianity_map(result: Any, field: str = "shapiro_pvalue", **kwargs):
        return plot_gaussianity_map(result, field=field, **kwargs)

    @staticmethod
    def plot_rejection_mask(
        result: Any,
        method: str = "anderson",
        alpha: float = 0.05,
        **kwargs,
    ):
        return plot_rejection_mask(
            result,
            method=method,
            alpha=alpha,
            **kwargs,
        )

    @staticmethod
    def plot_point_histogram_and_qq(
        data: Any,
        index_tuple: Sequence[int],
        member_dim: int | str = 0,
        **kwargs,
    ):
        return plot_point_histogram_and_qq(
            data,
            index_tuple=index_tuple,
            member_dim=member_dim,
            **kwargs,
        )


def compute_pointwise_gaussianity(
    data: Any,
    member_dim: int | str = 0,
    alpha: float = 0.05,
    min_samples: int = 20,
    use_shapiro: bool = True,
    compute_qq_r2: bool = True,
    skew_threshold: float | None = 0.5,
    kurtosis_threshold: float | None = 1.0,
    qq_r2_threshold: float | None = 0.98,
) -> dict[str, np.ndarray] | Any:
    """Compute Gaussianity diagnostics at every non-member grid point.

    Parameters
    ----------
    data:
        NumPy array or xarray DataArray.
    member_dim:
        Ensemble axis number. For xarray input, a dimension name is also
        accepted.
    alpha:
        Significance level for Shapiro-Wilk and the nearest available
        Anderson-Darling critical level.
    min_samples:
        Minimum number of finite ensemble values required at a grid point.
    skew_threshold, kurtosis_threshold, qq_r2_threshold:
        Optional, deliberately moderate thresholds used only by
        ``gaussian_flag``. Set any threshold to ``None`` to disable it.

    Returns
    -------
    dict or xarray.Dataset
        NumPy input returns a dictionary of NumPy arrays. xarray input returns
        a Dataset preserving all non-member dimensions and coordinates.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3.")

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
        values = np.asarray(data.values)
        output_dims = tuple(dim for dim in data.dims if dim != member_name)
        output_coords = {
            dim: data.coords[dim]
            for dim in output_dims
            if dim in data.coords
        }
    else:
        values = np.asarray(data)
        if values.ndim == 0:
            raise ValueError("data must have at least one dimension.")
        if isinstance(member_dim, str):
            raise TypeError("member_dim must be an integer for NumPy input.")
        member_axis = _normalize_axis_index(member_dim, values.ndim)

    member_first = np.moveaxis(values, member_axis, 0)
    output_shape = member_first.shape[1:]
    flattened = member_first.reshape(member_first.shape[0], -1)
    output = {
        field: np.full(flattened.shape[1], np.nan, dtype=float)
        for field in OUTPUT_FIELDS
    }

    for point_index in range(flattened.shape[1]):
        diagnostics = _pointwise_diagnostics(
            flattened[:, point_index],
            alpha=alpha,
            min_samples=min_samples,
            use_shapiro=use_shapiro,
            compute_qq_r2=compute_qq_r2,
            skew_threshold=skew_threshold,
            kurtosis_threshold=kurtosis_threshold,
            qq_r2_threshold=qq_r2_threshold,
        )
        for field, value in diagnostics.items():
            output[field][point_index] = value

    output = {
        field: values_for_field.reshape(output_shape)
        for field, values_for_field in output.items()
    }

    if not is_xarray:
        return output

    dataset = xr.Dataset(
        {
            field: xr.DataArray(
                values_for_field,
                dims=output_dims,
                coords=output_coords,
            )
            for field, values_for_field in output.items()
        }
    )
    dataset.attrs.update(
        {
            "member_dimension": member_name,
            "alpha": alpha,
            "anderson_alpha_used": _nearest_anderson_level(alpha) / 100.0,
            "min_samples": min_samples,
            "skew_threshold": skew_threshold,
            "kurtosis_threshold": kurtosis_threshold,
            "qq_r2_threshold": qq_r2_threshold,
            "interpretation": (
                "gaussian_flag=1 means compatible with Gaussianity under "
                "the configured diagnostics; it is not proof of normality."
            ),
        }
    )
    return dataset


def _get_result_field(result: Any, field: str) -> np.ndarray:
    if isinstance(result, Mapping):
        if field not in result:
            raise KeyError(f"Unknown result field: {field!r}")
        return np.asarray(result[field])
    if xr is not None and isinstance(result, xr.Dataset):
        if field not in result:
            raise KeyError(f"Unknown result field: {field!r}")
        return np.asarray(result[field].values)
    raise TypeError("result must be a diagnostics dictionary or xarray Dataset.")


def _prepare_map_field(values: np.ndarray, field: str) -> np.ndarray:
    values = np.asarray(values).squeeze()
    if values.ndim != 2:
        raise ValueError(
            f"{field!r} must be two-dimensional after squeezing; "
            f"received shape {values.shape}."
        )
    return values


def plot_gaussianity_map(
    result: Any,
    field: str = "shapiro_pvalue",
    *,
    ax=None,
    cmap: str = "viridis",
    highlight_below: float | None = None,
    marker_points: Sequence[tuple[int, int]] = (),
):
    """Plot one two-dimensional pointwise diagnostic field.

    ``highlight_below`` overlays red points where the field is below the
    supplied threshold. This is useful for highlighting Shapiro-Wilk
    rejections, for example with ``highlight_below=0.05``.

    ``marker_points`` contains ``(y, x)`` grid coordinates to mark with cyan
    stars.
    """

    values = _prepare_map_field(_get_result_field(result, field), field)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, origin="lower", cmap=cmap, aspect="auto")

    if highlight_below is not None:
        rejected_y, rejected_x = np.where(
            np.isfinite(values) & (values < highlight_below)
        )
        ax.scatter(
            rejected_x,
            rejected_y,
            s=8,
            color="red",
            alpha=0.75,
            linewidths=0,
            label=f"{field} < {highlight_below:g}",
        )

    for y, x in marker_points:
        ax.plot(
            x,
            y,
            marker="*",
            color="cyan",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=14,
            linestyle="none",
            label=f"Selected point (y={y}, x={x})",
        )

    ax.set_title(field.replace("_", " ").title())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.figure.colorbar(image, ax=ax, label=field)
    if highlight_below is not None or marker_points:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")
    return ax


def plot_rejection_mask(
    result: Any,
    method: str = "anderson",
    alpha: float = 0.05,
    *,
    ax=None,
    marker_points: Sequence[tuple[int, int]] = (),
):
    """Plot evidence-against-Gaussianity as a two-dimensional mask."""

    method = method.lower()
    if method == "anderson":
        level = _nearest_anderson_level(alpha)
        suffix = str(level).replace(".0", "").replace(".", "p")
        field = f"anderson_reject_{suffix}"
        mask = _get_result_field(result, field)
        title = f"Anderson-Darling rejection mask (alpha={level / 100:g})"
    elif method == "shapiro":
        field = "shapiro_pvalue"
        pvalues = _get_result_field(result, field)
        mask = np.where(np.isfinite(pvalues), pvalues <= alpha, np.nan)
        title = f"Shapiro-Wilk rejection mask (alpha={alpha:g})"
    else:
        raise ValueError("method must be 'anderson' or 'shapiro'.")

    mask = _prepare_map_field(mask, field)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(mask, origin="lower", cmap="coolwarm", vmin=0, vmax=1)

    for y, x in marker_points:
        ax.plot(
            x,
            y,
            marker="*",
            color="cyan",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=14,
            linestyle="none",
            label=f"Selected point (y={y}, x={x})",
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    colorbar = ax.figure.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Do not reject", "Reject"])
    if marker_points:
        ax.legend(loc="upper right")
    return ax


def _extract_point_sample(
    data: Any,
    index_tuple: Sequence[int],
    member_dim: int | str,
) -> np.ndarray:
    if xr is not None and isinstance(data, xr.DataArray):
        if isinstance(member_dim, str):
            member_name = member_dim
        else:
            member_axis = _normalize_axis_index(member_dim, data.ndim)
            member_name = data.dims[member_axis]
        non_member_dims = [dim for dim in data.dims if dim != member_name]
        if len(index_tuple) != len(non_member_dims):
            raise ValueError(
                "index_tuple must contain one index for each non-member "
                "dimension."
            )
        indexers = {
            dim: index for dim, index in zip(non_member_dims, index_tuple)
        }
        return np.asarray(data.isel(indexers).values, dtype=float)

    values = np.asarray(data)
    if isinstance(member_dim, str):
        raise TypeError("member_dim must be an integer for NumPy input.")
    member_axis = _normalize_axis_index(member_dim, values.ndim)
    non_member_axes = [
        axis for axis in range(values.ndim) if axis != member_axis
    ]
    if len(index_tuple) != len(non_member_axes):
        raise ValueError(
            "index_tuple must contain one index for each non-member dimension."
        )
    selection: list[Any] = [slice(None)] * values.ndim
    for axis, index in zip(non_member_axes, index_tuple):
        selection[axis] = index
    return np.asarray(values[tuple(selection)], dtype=float)


def plot_point_histogram_and_qq(
    data: Any,
    index_tuple: Sequence[int],
    member_dim: int | str = 0,
    *,
    bins: int = 30,
    alpha: float = 0.05,
    min_samples: int = 20,
    figsize: tuple[float, float] = (12, 5),
):
    """Plot a point histogram, fitted Gaussian PDF, and normal Q-Q plot."""

    sample = _extract_point_sample(data, index_tuple, member_dim)
    valid = sample[np.isfinite(sample)]
    diagnostics = _pointwise_diagnostics(
        valid,
        alpha=alpha,
        min_samples=min_samples,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if valid.size < min_samples or np.ptp(valid) == 0:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Diagnostics unavailable:\ninsufficient samples or zero variance",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return fig, axes

    axes[0].hist(
        valid,
        bins=bins,
        density=True,
        alpha=0.7,
        color="royalblue",
        edgecolor="black",
        label="Ensemble sample",
    )
    x_values = np.linspace(valid.min(), valid.max(), 300)
    axes[0].plot(
        x_values,
        stats.norm.pdf(
            x_values,
            diagnostics["mean"],
            diagnostics["std"],
        ),
        color="darkred",
        linestyle="--",
        linewidth=2,
        label="Fitted Gaussian",
    )
    axes[0].set_title("(a) Histogram")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Probability density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    stats.probplot(valid, dist="norm", plot=axes[1])
    axes[1].set_title("(b) Normal Q-Q plot")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Point {tuple(index_tuple)} | "
        f"Shapiro p={diagnostics['shapiro_pvalue']:.3e}, "
        f"Anderson={diagnostics['anderson_statistic']:.3f}, "
        f"skew={diagnostics['skewness']:+.3f}, "
        f"excess kurtosis={diagnostics['excess_kurtosis']:+.3f}, "
        f"Q-Q R²={diagnostics['qq_r2']:.5f}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes


def run_synthetic_test(
    seed: int = 42,
    members: int = 500,
    shape: tuple[int, int] = (8, 8),
) -> dict[str, Any]:
    """Run a small Gaussian-versus-mixture demonstration.

    The returned rates summarize how often each field is classified as
    compatible with Gaussianity under the configured combined criteria.
    """

    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(members, *shape))

    component = rng.random(size=(members, *shape)) < 0.5
    mixture = np.where(
        component,
        rng.normal(loc=-1.5, scale=0.7, size=(members, *shape)),
        rng.normal(loc=1.5, scale=0.7, size=(members, *shape)),
    )

    gaussian_result = compute_pointwise_gaussianity(gaussian)
    mixture_result = compute_pointwise_gaussianity(mixture)
    gaussian_rate = float(np.nanmean(gaussian_result["gaussian_flag"]))
    mixture_rate = float(np.nanmean(mixture_result["gaussian_flag"]))

    return {
        "gaussian_data": gaussian,
        "non_gaussian_data": mixture,
        "gaussian_result": gaussian_result,
        "non_gaussian_result": mixture_result,
        "gaussian_compatible_fraction": gaussian_rate,
        "non_gaussian_compatible_fraction": mixture_rate,
    }
