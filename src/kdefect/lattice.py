"""Discrete free energy and its exact lattice gradient.

Forward differences and their negative adjoints are paired deliberately.  As
a result, ``deterministic_force`` is the negative gradient of ``energy`` for
the *discretized* theory, not merely a continuum expression transcribed onto
a grid.  Gate 0 verifies this identity numerically for every arm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .laws import GradientLaw

ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


def forward_gradient(field: ComplexArray, dx: float = 1.0) -> tuple[ComplexArray, ComplexArray]:
    """Periodic forward differences along the last two axes."""

    gx = (np.roll(field, -1, axis=-2) - field) / dx
    gy = (np.roll(field, -1, axis=-1) - field) / dx
    return gx, gy


def flux_divergence(fx: ComplexArray, fy: ComplexArray, dx: float = 1.0) -> ComplexArray:
    """Divergence paired with :func:`forward_gradient` on a periodic grid."""

    return (
        fx
        - np.roll(fx, 1, axis=-2)
        + fy
        - np.roll(fy, 1, axis=-1)
    ) / dx


def laplacian(field: ComplexArray, dx: float = 1.0) -> ComplexArray:
    """Five-point periodic Laplacian, including arbitrary batch axes."""

    gx, gy = forward_gradient(field, dx)
    return flux_divergence(gx, gy, dx)


def gradient_invariant(field: ComplexArray, dx: float = 1.0) -> RealArray:
    """Return Y=|D_x^+ psi|^2+|D_y^+ psi|^2 at each lattice site."""

    gx, gy = forward_gradient(field, dx)
    return np.abs(gx) ** 2 + np.abs(gy) ** 2


def energy(
    field: ComplexArray,
    law: GradientLaw,
    *,
    epsilon: float,
    kappa: float,
    quartic: float,
    regulator: float,
    dx: float = 1.0,
) -> float | RealArray:
    """Evaluate the lattice free energy.

    Leading dimensions are treated as batch dimensions.  The returned scalar
    or array is summed only over the final two spatial axes.
    """

    y = gradient_invariant(field, dx)
    power = np.abs(field) ** 2
    lap = laplacian(field, dx)
    density = (
        0.5 * law.energy_density(y, kappa)
        + 0.5 * regulator * np.abs(lap) ** 2
        - 0.5 * epsilon * power
        + 0.25 * quartic * power**2
    )
    total = dx**2 * np.sum(density, axis=(-2, -1))
    if np.ndim(total) == 0:
        return float(total)
    return np.asarray(total)


def deterministic_force(
    field: ComplexArray,
    law: GradientLaw,
    *,
    epsilon: float,
    kappa: float,
    quartic: float,
    regulator: float,
    dx: float = 1.0,
) -> ComplexArray:
    """Negative lattice-energy gradient for overdamped Model-A dynamics."""

    gx, gy = forward_gradient(field, dx)
    y = np.abs(gx) ** 2 + np.abs(gy) ** 2
    h = law.mobility(y, kappa)
    variational_flux = flux_divergence(h * gx, h * gy, dx)
    power = np.abs(field) ** 2
    biharmonic = laplacian(laplacian(field, dx), dx)
    return (
        variational_flux
        + epsilon * field
        - quartic * power * field
        - regulator * biharmonic
    )


def euler_maruyama_step(
    field: ComplexArray,
    law: GradientLaw,
    rng: np.random.Generator,
    *,
    dt: float,
    temperature: float,
    epsilon: float,
    kappa: float,
    quartic: float,
    regulator: float,
    dx: float = 1.0,
) -> ComplexArray:
    """Advance one stochastic gradient-flow step.

    Each real component has continuum covariance
    ``<eta_a(x,t) eta_b(x',t')> = 2 T delta_ab delta(x-x') delta(t-t')``.
    """

    force = deterministic_force(
        field,
        law,
        epsilon=epsilon,
        kappa=kappa,
        quartic=quartic,
        regulator=regulator,
        dx=dx,
    )
    if temperature <= 0.0:
        return field + dt * force
    noise_scale = np.sqrt(2.0 * temperature * dt / dx**2)
    noise = rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
    return field + dt * force + noise_scale * noise
