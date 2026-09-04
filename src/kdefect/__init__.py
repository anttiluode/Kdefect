"""Action-derived simulations of noncanonical U(1) defects."""

from .lattice import deterministic_force, energy
from .laws import GradientLaw, get_law
from .quench import QuenchConfig, QuenchResult, simulate_quench
from .vortices import Vortex, detect_vortices, plaquette_winding, track_vortices

__all__ = [
    "GradientLaw",
    "QuenchConfig",
    "QuenchResult",
    "Vortex",
    "detect_vortices",
    "deterministic_force",
    "energy",
    "get_law",
    "plaquette_winding",
    "simulate_quench",
    "track_vortices",
]

__version__ = "0.1.0"
