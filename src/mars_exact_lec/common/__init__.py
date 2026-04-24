"""Common geometric and integral tools for Mars exact diagnostics."""

from .geopotential import (
    GEOPOTENTIAL_MODES,
    broadcast_surface_field,
    normalize_geopotential_mode,
    reconstruct_hydrostatic_geopotential,
    resolve_geopotential,
)
from .grid_weights import cell_area, infer_grid, latitude_weights, zonal_band_area
from .integrals import (
    MassIntegrator,
    build_mass_integrator,
    delta_p,
    integrate_mass_full,
    integrate_mass_zonal,
    integrate_surface,
)
from .normalization import normalize_dataset_per_area, planetary_area, to_per_area
from .time_derivatives import time_derivative
from .topography_measure import TopographyAwareMeasure
from .zonal_ops import (
    representative_eddy,
    representative_zonal_mean,
    theta_coverage,
    weighted_coverage,
    weighted_representative_eddy,
    weighted_representative_zonal_mean,
    zonal_mean,
)

__all__ = [
    "infer_grid",
    "latitude_weights",
    "cell_area",
    "zonal_band_area",
    "GEOPOTENTIAL_MODES",
    "broadcast_surface_field",
    "normalize_geopotential_mode",
    "reconstruct_hydrostatic_geopotential",
    "resolve_geopotential",
    "delta_p",
    "MassIntegrator",
    "TopographyAwareMeasure",
    "build_mass_integrator",
    "integrate_mass_full",
    "integrate_mass_zonal",
    "integrate_surface",
    "planetary_area",
    "to_per_area",
    "normalize_dataset_per_area",
    "time_derivative",
    "zonal_mean",
    "weighted_coverage",
    "weighted_representative_zonal_mean",
    "weighted_representative_eddy",
    "theta_coverage",
    "representative_zonal_mean",
    "representative_eddy",
]
