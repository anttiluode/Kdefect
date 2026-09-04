"""Small-sample diagnostics used by the pilot, never dressed up as discovery."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .vortices import Vortex, periodic_distance_matrix

RealArray = NDArray[np.floating]


def counting_cumulants(values: ArrayLike) -> dict[str, float]:
    """Return population cumulants kappa_1 through kappa_4."""

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {f"kappa_{order}": float("nan") for order in range(1, 5)}
    mean = float(np.mean(x))
    centered = x - mean
    second = float(np.mean(centered**2))
    third = float(np.mean(centered**3))
    fourth = float(np.mean(centered**4) - 3.0 * second**2)
    return {
        "kappa_1": mean,
        "kappa_2": second,
        "kappa_3": third,
        "kappa_4": fourth,
    }


def nearest_neighbor_distances(
    vortices: Sequence[Vortex], box_size: float
) -> RealArray:
    """Nearest-neighbor distances using periodic minimum images."""

    if len(vortices) < 2:
        return np.asarray([], dtype=float)
    positions = np.array([[v.x, v.y] for v in vortices], dtype=float)
    distances = periodic_distance_matrix(positions, positions, box_size)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def normalized_nn_spacings(vortices: Sequence[Vortex], box_size: float) -> RealArray:
    """Nearest-neighbor spacings normalized to unit sample mean."""

    distances = nearest_neighbor_distances(vortices, box_size)
    if len(distances) == 0 or float(np.mean(distances)) == 0.0:
        return np.asarray([], dtype=float)
    return distances / np.mean(distances)


def poisson_nn_ks(spacings: ArrayLike) -> float:
    """One-sample KS distance from the 2D Poisson nearest-neighbor law.

    With unit mean spacing the reference CDF is
    ``1-exp(-pi*s^2/4)``.  This is a diagnostic only; pooled vortices from
    separate runs are not treated as independent for inferential p-values.
    """

    sample = np.sort(np.asarray(spacings, dtype=float))
    if sample.size == 0:
        return float("nan")
    cdf = 1.0 - np.exp(-np.pi * sample**2 / 4.0)
    upper = np.arange(1, sample.size + 1) / sample.size
    lower = np.arange(0, sample.size) / sample.size
    return float(max(np.max(np.abs(upper - cdf)), np.max(np.abs(cdf - lower))))


def density_spectrum_diagnostics(field: NDArray[np.complexfloating], dx: float) -> dict[str, float]:
    """Radially averaged spectrum of density fluctuations on a square grid."""

    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("density spectrum expects one square two-dimensional field")
    size = field.shape[0]
    density = np.abs(field) ** 2
    density = density - np.mean(density)
    power = np.abs(np.fft.fft2(density)) ** 2
    wave = 2.0 * np.pi * np.fft.fftfreq(size, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    magnitude = np.sqrt(kx**2 + ky**2)
    fundamental = 2.0 * np.pi / (size * dx)
    shell = np.rint(magnitude / fundamental).astype(int)
    shell_power = np.bincount(shell.ravel(), weights=power.ravel())
    shell_count = np.bincount(shell.ravel())
    radial = np.divide(
        shell_power,
        shell_count,
        out=np.zeros_like(shell_power, dtype=float),
        where=shell_count > 0,
    )
    radial[0] = 0.0
    peak_shell = int(np.argmax(radial))
    nonzero_power = power[magnitude > 0.0]
    nonzero_k = magnitude[magnitude > 0.0]
    total = float(np.sum(nonzero_power))
    centroid = float(np.sum(nonzero_k * nonzero_power) / total) if total > 0.0 else float("nan")
    nyquist = np.pi / dx
    high_fraction = (
        float(np.sum(power[magnitude >= 0.5 * nyquist]) / total)
        if total > 0.0
        else float("nan")
    )
    return {
        "radial_peak_k": float(peak_shell * fundamental),
        "spectral_centroid_k": centroid,
        "high_k_power_fraction": high_fraction,
    }


def loglog_slope(x: ArrayLike, y: ArrayLike) -> float:
    """Ordinary least-squares slope in log-log coordinates."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = (x_array > 0.0) & (y_array > 0.0) & np.isfinite(x_array) & np.isfinite(y_array)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x_array[valid]), np.log(y_array[valid]), 1)[0])


def bootstrap_loglog_slope(
    x: ArrayLike,
    samples_by_x: Sequence[ArrayLike],
    *,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap whole realizations within each quench-time group."""

    x_array = np.asarray(x, dtype=float)
    groups = [np.asarray(group, dtype=float) for group in samples_by_x]
    if len(groups) != len(x_array) or any(len(group) == 0 for group in groups):
        raise ValueError("one nonempty realization group is required for each x")
    rng = np.random.default_rng(seed)
    slopes = np.empty(draws, dtype=float)
    for draw in range(draws):
        means = [
            float(np.mean(rng.choice(group, size=len(group), replace=True)))
            for group in groups
        ]
        slopes[draw] = loglog_slope(x_array, means)
    finite = slopes[np.isfinite(slopes)]
    if len(finite) == 0:
        return {"median": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
    low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
    return {"median": float(median), "low_95": float(low), "high_95": float(high)}


def paired_bootstrap_mean_difference(
    reference: ArrayLike,
    treatment: ArrayLike,
    *,
    draws: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the paired mean difference ``treatment-reference``."""

    left = np.asarray(reference, dtype=float)
    right = np.asarray(treatment, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("reference and treatment must be equal nonempty vectors")
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) == 0:
        return {
            "finite_pairs": 0,
            "observed": float("nan"),
            "median": float("nan"),
            "low_95": float("nan"),
            "high_95": float("nan"),
        }
    differences = right - left
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    bootstrapped = np.mean(differences[indices], axis=1)
    low, median, high = np.quantile(bootstrapped, [0.025, 0.5, 0.975])
    return {
        "finite_pairs": len(differences),
        "observed": float(np.mean(differences)),
        "median": float(median),
        "low_95": float(low),
        "high_95": float(high),
    }


def bootstrap_loglog_slope_difference(
    x: ArrayLike,
    reference_by_x: Sequence[ArrayLike],
    treatment_by_x: Sequence[ArrayLike],
    *,
    draws: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of treatment minus reference log-log slopes."""

    x_array = np.asarray(x, dtype=float)
    left = [np.asarray(group, dtype=float) for group in reference_by_x]
    right = [np.asarray(group, dtype=float) for group in treatment_by_x]
    if len(left) != len(x_array) or len(right) != len(x_array):
        raise ValueError("one pair of groups is required for every x")
    if any(
        a.shape != b.shape or a.ndim != 1 or len(a) == 0
        for a, b in zip(left, right, strict=True)
    ):
        raise ValueError("paired groups must be equal nonempty vectors")
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=float)
    for draw in range(draws):
        reference_means: list[float] = []
        treatment_means: list[float] = []
        for reference_group, treatment_group in zip(left, right, strict=True):
            indices = rng.integers(0, len(reference_group), size=len(reference_group))
            reference_means.append(float(np.mean(reference_group[indices])))
            treatment_means.append(float(np.mean(treatment_group[indices])))
        differences[draw] = loglog_slope(x_array, treatment_means) - loglog_slope(
            x_array, reference_means
        )
    finite = differences[np.isfinite(differences)]
    if len(finite) == 0:
        return {
            "observed": float("nan"),
            "median": float("nan"),
            "low_95": float("nan"),
            "high_95": float("nan"),
        }
    observed = loglog_slope(x_array, [np.mean(group) for group in right]) - loglog_slope(
        x_array, [np.mean(group) for group in left]
    )
    low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
    return {
        "observed": float(observed),
        "median": float(median),
        "low_95": float(low),
        "high_95": float(high),
    }


def charge_form_factor(
    vortices: Sequence[Vortex],
    box_size: float,
    modes: Sequence[tuple[int, int]] = ((1, 0), (0, 1), (1, 1), (1, -1)),
) -> dict[str, float]:
    """Low-k charge structure factor |sum q exp(-ik.r)|^2/N."""

    if not vortices:
        return {f"{nx},{ny}": float("nan") for nx, ny in modes}
    positions = np.array([[v.x, v.y] for v in vortices], dtype=float)
    charges = np.array([v.charge for v in vortices], dtype=float)
    result: dict[str, float] = {}
    for nx, ny in modes:
        wavevector = 2.0 * np.pi * np.array([nx, ny], dtype=float) / box_size
        amplitude = np.sum(charges * np.exp(-1j * (positions @ wavevector)))
        result[f"{nx},{ny}"] = float(abs(amplitude) ** 2 / len(vortices))
    return result
