from __future__ import annotations

import numpy as np
import pytest

from kdefect.lattice import deterministic_force, energy
from kdefect.laws import LAWS


@pytest.mark.parametrize("name", sorted(LAWS))
def test_all_arms_share_canonical_small_gradient_limit(name: str) -> None:
    law = LAWS[name]
    y = np.array([1.0e-10, 2.0e-10])
    assert np.allclose(law.energy_density(y, 3.0) / y, 1.0, rtol=2.0e-9)
    assert np.allclose(law.mobility(y, 3.0), 1.0, rtol=1.0e-9)


def test_log_arm_has_declared_ellipticity_boundary() -> None:
    kappa = 4.0
    law = LAWS["log"]
    assert law.longitudinal_stiffness(0.99 / kappa, kappa) > 0.0
    assert law.longitudinal_stiffness(1.0 / kappa, kappa) == pytest.approx(0.0)
    assert law.longitudinal_stiffness(1.01 / kappa, kappa) < 0.0
    assert np.all(LAWS["sqrt"].longitudinal_stiffness(np.logspace(-8, 8), kappa) > 0.0)


@pytest.mark.parametrize("name", sorted(LAWS))
def test_force_is_negative_discrete_energy_gradient(name: str) -> None:
    rng = np.random.default_rng(81)
    field = 0.12 * (
        rng.standard_normal((9, 11)) + 1j * rng.standard_normal((9, 11))
    )
    direction = rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
    parameters = dict(
        epsilon=0.2,
        kappa=4.0,
        quartic=1.0,
        regulator=0.04,
        dx=0.7,
    )
    force = deterministic_force(field, LAWS[name], **parameters)
    delta = 1.0e-6
    numerical = (
        energy(field + delta * direction, LAWS[name], **parameters)
        - energy(field - delta * direction, LAWS[name], **parameters)
    ) / (2.0 * delta)
    variational = -parameters["dx"] ** 2 * np.real(np.vdot(direction, force))
    assert numerical == pytest.approx(variational, rel=2.0e-8, abs=2.0e-8)


@pytest.mark.parametrize("name", sorted(LAWS))
def test_deterministic_gradient_flow_decreases_energy(name: str) -> None:
    rng = np.random.default_rng(5)
    field = 0.2 * (
        rng.standard_normal((10, 10)) + 1j * rng.standard_normal((10, 10))
    )
    parameters = dict(
        epsilon=0.25,
        kappa=4.0,
        quartic=1.0,
        regulator=0.03,
        dx=1.0,
    )
    before = energy(field, LAWS[name], **parameters)
    force = deterministic_force(field, LAWS[name], **parameters)
    after = energy(field + 1.0e-5 * force, LAWS[name], **parameters)
    assert after < before
