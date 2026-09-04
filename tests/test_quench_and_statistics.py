from __future__ import annotations

import numpy as np
import pytest

from kdefect.quench import quench_epsilon
from kdefect.statistics import counting_cumulants, loglog_slope, poisson_nn_ks


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
