"""Gradient-energy laws used by the Kdefect comparison.

Every arm has the same quadratic small-gradient limit, F(Y) = Y + O(Y^2),
where Y = |grad psi|^2.  They therefore share the same linear instability at
the U(1)-breaking transition while differing inside nonlinear defect cores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

RealArray = NDArray[np.floating]


@dataclass(frozen=True)
class GradientLaw:
    """A rotationally invariant gradient energy F(Y).

    ``mobility`` is F'(Y), the coefficient multiplying grad(psi) in the
    variational flux. ``longitudinal_stiffness`` is F'(Y) + 2 Y F''(Y), the
    principal coefficient for perturbations parallel to the background
    gradient.  Positivity of both is the second-order ellipticity test.
    """

    name: str
    label: str
    energy_density: Callable[[ArrayLike, float], RealArray]
    mobility: Callable[[ArrayLike, float], RealArray]
    longitudinal_stiffness: Callable[[ArrayLike, float], RealArray]
    description: str


def _canonical_energy(y: ArrayLike, kappa: float) -> RealArray:
    del kappa
    return np.asarray(y, dtype=float)


def _canonical_mobility(y: ArrayLike, kappa: float) -> RealArray:
    del kappa
    return np.ones_like(np.asarray(y, dtype=float))


def _canonical_stiffness(y: ArrayLike, kappa: float) -> RealArray:
    del kappa
    return np.ones_like(np.asarray(y, dtype=float))


def _log_energy(y: ArrayLike, kappa: float) -> RealArray:
    return np.log1p(kappa * np.asarray(y, dtype=float)) / kappa


def _log_mobility(y: ArrayLike, kappa: float) -> RealArray:
    return 1.0 / (1.0 + kappa * np.asarray(y, dtype=float))


def _log_stiffness(y: ArrayLike, kappa: float) -> RealArray:
    ky = kappa * np.asarray(y, dtype=float)
    return (1.0 - ky) / (1.0 + ky) ** 2


def _sqrt_energy(y: ArrayLike, kappa: float) -> RealArray:
    y_array = np.asarray(y, dtype=float)
    ky = kappa * y_array
    # Rationalized form avoids cancellation when the common linear limit is
    # tested at tiny Y.
    return 2.0 * y_array / (np.sqrt(1.0 + ky) + 1.0)


def _sqrt_mobility(y: ArrayLike, kappa: float) -> RealArray:
    return 1.0 / np.sqrt(1.0 + kappa * np.asarray(y, dtype=float))


def _sqrt_stiffness(y: ArrayLike, kappa: float) -> RealArray:
    return (1.0 + kappa * np.asarray(y, dtype=float)) ** (-1.5)


LAWS: dict[str, GradientLaw] = {
    "canonical": GradientLaw(
        name="canonical",
        label="Canonical",
        energy_density=_canonical_energy,
        mobility=_canonical_mobility,
        longitudinal_stiffness=_canonical_stiffness,
        description="F(Y)=Y; ordinary Model-A U(1) reference arm.",
    ),
    "log": GradientLaw(
        name="log",
        label="Log-concave",
        energy_density=_log_energy,
        mobility=_log_mobility,
        longitudinal_stiffness=_log_stiffness,
        description=(
            "F(Y)=log(1+kappa Y)/kappa; loses longitudinal ellipticity "
            "when kappa Y>1 and therefore requires the explicit regulator."
        ),
    ),
    "sqrt": GradientLaw(
        name="sqrt",
        label="Square-root screened",
        energy_density=_sqrt_energy,
        mobility=_sqrt_mobility,
        longitudinal_stiffness=_sqrt_stiffness,
        description=(
            "F(Y)=2(sqrt(1+kappa Y)-1)/kappa; suppresses large-gradient "
            "flux while remaining elliptic for every Y>=0."
        ),
    ),
}


def get_law(name: str) -> GradientLaw:
    """Return a registered gradient law by its stable command-line name."""

    try:
        return LAWS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(LAWS))
        raise ValueError(f"unknown gradient law {name!r}; choose {choices}") from exc
