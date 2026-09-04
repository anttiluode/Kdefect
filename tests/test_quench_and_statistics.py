from __future__ import annotations

import numpy as np
import pytest

from kdefect.quench import quench_epsilon
from kdefect.statistics import (
    bootstrap_loglog_slope_difference,
    counting_cumulants,
    density_spectrum_diagnostics,
    loglog_slope,
    paired_bootstrap_mean_difference,
    poisson_nn_ks,
)


def test_quench_is_linear_and_clipped() -> None:
    assert quench_epsilon(-20.0, 10.0, 0.5) == pytest.approx(-0.5)
    assert quench_epsilon(-5.0, 10.0, 0.5) == pytest.approx(-0.25)
    assert quench_epsilon(0.0, 10.0, 0.5) == pytest.approx(0.0)
    assert quench_epsilon(20.0, 10.0, 0.5) == pytest.approx(0.5)


def test_counting_cumulants_for_symmetric_binary_sample() -> None:
    cumulants = counting_cumulants([-1.0, 1.0])
    assert cumulants["kappa_1"] == pytest.approx(0.0)
    assert cumulants["kappa_2"] == pytest.approx(1.0)
    assert cumulants["kappa_3"] == pytest.approx(0.0)
    assert cumulants["kappa_4"] == pytest.approx(-2.0)


def test_loglog_slope_recovers_power() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert loglog_slope(x, 3.0 * x ** (-0.5)) == pytest.approx(-0.5)


def test_poisson_ks_is_small_for_large_reference_sample() -> None:
    rng = np.random.default_rng(90)
    uniform = rng.random(50_000)
    spacings = np.sqrt(-4.0 * np.log1p(-uniform) / np.pi)
    assert poisson_nn_ks(spacings) < 0.01


def test_paired_bootstrap_preserves_exact_shift() -> None:
    result = paired_bootstrap_mean_difference([1, 2, 4], [3, 4, 6], draws=100, seed=1)
    assert result["observed"] == pytest.approx(2.0)
    assert result["low_95"] == pytest.approx(2.0)
    assert result["high_95"] == pytest.approx(2.0)


def test_paired_bootstrap_drops_only_nonfinite_pairs() -> None:
    result = paired_bootstrap_mean_difference(
        [1.0, np.nan, 3.0], [2.0, 100.0, 4.0], draws=100, seed=1
    )
    assert result["finite_pairs"] == 2
    assert result["observed"] == pytest.approx(1.0)


def test_paired_slope_contrast_recovers_exponent_shift() -> None:
    x = [1.0, 2.0, 4.0]
    reference = [[10.0, 10.0], [5.0, 5.0], [2.5, 2.5]]
    treatment = [[10.0, 10.0], [2.5, 2.5], [0.625, 0.625]]
    result = bootstrap_loglog_slope_difference(x, reference, treatment, draws=100, seed=2)
    assert result["observed"] == pytest.approx(-1.0)
    assert result["low_95"] == pytest.approx(-1.0)
    assert result["high_95"] == pytest.approx(-1.0)


def test_density_spectrum_finds_imposed_radial_mode() -> None:
    size = 64
    mode = 5
    x = np.arange(size)[:, None]
    density = 1.0 + 0.2 * np.cos(2.0 * np.pi * mode * x / size)
    field = np.sqrt(density) * np.ones((1, size), dtype=complex)
    result = density_spectrum_diagnostics(field, dx=1.0)
    assert result["radial_peak_k"] == pytest.approx(2.0 * np.pi * mode / size)
