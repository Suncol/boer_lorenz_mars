"""Reference-state utilities for the Mars exact Lorenz/Boer branch."""

from .interpolate_isentropes import (
    ISENTROPIC_DIM,
    ISENTROPIC_LAYER_DIM,
    interpolate_pressure_to_isentropes,
    interpolate_pressure_to_isentropes_metadata,
    isentropic_interfaces,
    isentropic_layer_mass_statistics,
    normalize_isentropic_coordinate,
    potential_temperature,
    pressure_level_edges,
    pressure_at_isentropes,
)
from .finite_volume_solver import FiniteVolumeReferenceState
from .koehler1986_preprocessing import (
    build_theta_levels,
    interpolate_pressure_to_koehler_isentropes,
    koehler_isentropic_layer_mass_statistics,
    resolve_surface_potential_temperature,
)
from .koehler1986_solver import Koehler1986ReferenceState
from .koehler_solver import KoehlerReferenceState
from .solution import ReferenceStateSolution

__all__ = [
    "ISENTROPIC_DIM",
    "ISENTROPIC_LAYER_DIM",
    "pressure_level_edges",
    "normalize_isentropic_coordinate",
    "isentropic_interfaces",
    "potential_temperature",
    "interpolate_pressure_to_isentropes",
    "interpolate_pressure_to_isentropes_metadata",
    "pressure_at_isentropes",
    "isentropic_layer_mass_statistics",
    "ReferenceStateSolution",
    "FiniteVolumeReferenceState",
    "build_theta_levels",
    "resolve_surface_potential_temperature",
    "interpolate_pressure_to_koehler_isentropes",
    "koehler_isentropic_layer_mass_statistics",
    "Koehler1986ReferenceState",
    "KoehlerReferenceState",
]
