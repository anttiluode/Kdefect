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
) -> dict[str, float]:
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
