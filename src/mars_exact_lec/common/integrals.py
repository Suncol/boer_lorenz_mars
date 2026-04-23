"""Mass integrals for phase-1 Mars exact diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .._validation import (
    normalize_coordinate,
    normalize_field,
    normalize_surface_field,
    normalize_zonal_field,
    require_dataarray,
)
from ..constants_mars import MARS, MarsConstants
from .grid_weights import cell_area, zonal_band_area


def _coordinate_matches(reference: xr.DataArray, current: xr.DataArray, coord_name: str) -> bool:
    reference_values = reference.coords[coord_name].values
    current_values = current.coords[coord_name].values
    return np.allclose(reference_values, current_values)


def _get_explicit_level_bounds(level: xr.DataArray, bounds: xr.DataArray | None = None) -> xr.DataArray | None:
    if bounds is not None:
        bounds = xr.DataArray(bounds)
        if bounds.ndim != 2:
            raise ValueError("Pressure level bounds must be two-dimensional.")
        if bounds.shape[-1] != 2 and bounds.shape[0] == 2:
            bounds = xr.DataArray(
                bounds.values.T,
                dims=("level", "bounds"),
                coords={"level": level.values, "bounds": [0, 1]},
            )
        else:
            if bounds.dims[0] != "level":
                bounds = bounds.rename({bounds.dims[0]: "level"})
            if bounds.dims[1] != "bounds":
                bounds = bounds.rename({bounds.dims[1]: "bounds"})
        return bounds.transpose("level", "bounds")

    bounds_name = level.attrs.get("bounds")
    candidates = [bounds_name] if bounds_name else []
    candidates.extend(["level_bounds", "level_bnds", "plev_bounds", "plev_bnds"])

    for candidate in candidates:
        if candidate and candidate in level.coords:
            bounds = xr.DataArray(level.coords[candidate])
            if bounds.ndim != 2:
                raise ValueError("Pressure level bounds must be two-dimensional.")
            if bounds.shape[-1] != 2 and bounds.shape[0] == 2:
                bounds = xr.DataArray(
                    bounds.values.T,
                    dims=("level", "bounds"),
                    coords={"level": level.values, "bounds": [0, 1]},
                )
            else:
                if bounds.dims[0] != "level":
                    bounds = bounds.rename({bounds.dims[0]: "level"})
                if bounds.dims[1] != "bounds":
                    bounds = bounds.rename({bounds.dims[1]: "bounds"})
            return bounds.transpose("level", "bounds")
    return None


def pressure_level_edges(level: xr.DataArray, *, bounds: xr.DataArray | None = None) -> xr.DataArray:
    """Return strictly descending pressure interfaces for a pressure coordinate."""

    level = normalize_coordinate(level, "level")
    resolved_bounds = _get_explicit_level_bounds(level, bounds=bounds)
    if resolved_bounds is not None:
        values = np.asarray(resolved_bounds.values, dtype=float)
        upper = np.maximum(values[:, 0], values[:, 1])
        lower = np.minimum(values[:, 0], values[:, 1])
        level_values = np.asarray(level.values, dtype=float)
        if np.any((level_values > upper) | (level_values < lower)):
            raise ValueError("Each pressure level must lie inside its explicit pressure bounds.")
        if values.shape[0] > 1 and not np.allclose(lower[:-1], upper[1:]):
            raise ValueError("Pressure level bounds must define contiguous interfaces.")

        edges = np.empty(values.shape[0] + 1, dtype=float)
        edges[0] = upper[0]
        edges[1:] = lower
    else:
        values = np.asarray(level.values, dtype=float)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] + 0.5 * (values[0] - values[1])
        edges[-1] = max(0.0, values[-1] + 0.5 * (values[-1] - values[-2]))

    if np.any(edges <= 0.0):
        raise ValueError("Derived pressure interfaces must remain strictly positive.")
    if not np.all(np.diff(edges) < 0.0):
        raise ValueError("Derived pressure interfaces must be strictly descending.")

    return xr.DataArray(
        edges,
        dims=("level_edge",),
        coords={"level_edge": np.arange(edges.size)},
        name="pressure_level_edges",
        attrs={"units": "Pa"},
    )


def delta_p(level: xr.DataArray, *, bounds: xr.DataArray | None = None) -> xr.DataArray:
    """Return positive pressure thicknesses for descending pressure levels."""

    level = normalize_coordinate(level, "level")
    edges = pressure_level_edges(level, bounds=bounds)
    thickness = np.asarray(edges.values[:-1] - edges.values[1:], dtype=float)
    if np.any(thickness <= 0.0):
        raise ValueError("Derived pressure thickness must be strictly positive.")

    return xr.DataArray(
        thickness,
        dims=("level",),
        coords={"level": level.values},
        name="delta_p",
        attrs={"units": "Pa"},
    )


@dataclass(frozen=True)
class MassIntegrator:
    """Mass integrator for full 3D and already zonal-mean fields."""

    delta_p: xr.DataArray
    cell_area: xr.DataArray
    zonal_band_area: xr.DataArray
    level_bounds: xr.DataArray | None = None
    constants: MarsConstants = MARS

    def __post_init__(self) -> None:
        delta_p = require_dataarray(self.delta_p, "delta_p")
        if delta_p.ndim != 1 or delta_p.dims != ("level",):
            raise ValueError("'delta_p' must be a one-dimensional DataArray with dim 'level'.")
        if np.any(np.asarray(delta_p.values, dtype=float) <= 0.0):
            raise ValueError("'delta_p' must contain strictly positive pressure thicknesses.")
        object.__setattr__(self, "delta_p", delta_p)
        if self.level_bounds is not None:
            level_bounds = require_dataarray(self.level_bounds, "level_bounds")
            if set(level_bounds.dims) != {"level", "bounds"}:
                raise ValueError("'level_bounds' must contain exactly the dims ('level', 'bounds').")
            level_bounds = level_bounds.transpose("level", "bounds")
            if not _coordinate_matches(delta_p, level_bounds, "level"):
                raise ValueError("Level bounds must match the integrator level coordinate.")
            object.__setattr__(self, "level_bounds", level_bounds)
        object.__setattr__(
            self,
            "cell_area",
            self.cell_area.transpose("latitude", "longitude"),
        )
        object.__setattr__(
            self,
            "zonal_band_area",
            self.zonal_band_area.transpose("latitude"),
        )

    @property
    def level_edges(self) -> xr.DataArray:
        return pressure_level_edges(self.delta_p.coords["level"], bounds=self.level_bounds)

    @property
    def full_mass_weights(self) -> xr.DataArray:
        weights = (self.cell_area * self.delta_p) / self.constants.g
        weights.name = "mass_weights_full"
        weights.attrs["units"] = "kg"
        return weights

    @property
    def zonal_mass_weights(self) -> xr.DataArray:
        weights = (self.zonal_band_area * self.delta_p) / self.constants.g
        weights.name = "mass_weights_zonal"
        weights.attrs["units"] = "kg"
        return weights

    @property
    def surface_weights(self) -> xr.DataArray:
        weights = self.cell_area / self.constants.g
        weights.name = "surface_weights"
        weights.attrs["units"] = "m s2"
        return weights

    def _ensure_full_grid_matches(self, field: xr.DataArray) -> None:
        if not _coordinate_matches(self.delta_p, field, "level"):
            raise ValueError("Field level coordinates do not match the integrator grid.")
        if not _coordinate_matches(self.cell_area, field, "latitude"):
            raise ValueError("Field latitude coordinates do not match the integrator grid.")
        if not _coordinate_matches(self.cell_area, field, "longitude"):
            raise ValueError("Field longitude coordinates do not match the integrator grid.")

    def _ensure_zonal_grid_matches(self, field: xr.DataArray) -> None:
        if not _coordinate_matches(self.delta_p, field, "level"):
            raise ValueError("Field level coordinates do not match the integrator grid.")
        if not _coordinate_matches(self.zonal_band_area, field, "latitude"):
            raise ValueError("Field latitude coordinates do not match the integrator grid.")

    def _ensure_surface_grid_matches(self, field: xr.DataArray) -> None:
        if not _coordinate_matches(self.cell_area, field, "latitude"):
            raise ValueError("Field latitude coordinates do not match the integrator grid.")
        if not _coordinate_matches(self.cell_area, field, "longitude"):
            raise ValueError("Field longitude coordinates do not match the integrator grid.")

    def integrate_full(self, field: xr.DataArray) -> xr.DataArray:
        field = normalize_field(field, "field")
        self._ensure_full_grid_matches(field)
        weights = self.full_mass_weights
        return (field * weights).sum(dim=("level", "latitude", "longitude"))

    def integrate_zonal(self, field: xr.DataArray) -> xr.DataArray:
        field = normalize_zonal_field(field, "field")
        self._ensure_zonal_grid_matches(field)
        weights = self.zonal_mass_weights
        return (field * weights).sum(dim=("level", "latitude"))

    def integrate_surface(self, field: xr.DataArray) -> xr.DataArray:
        field = normalize_surface_field(field, "field")
        self._ensure_surface_grid_matches(field)
        weights = self.surface_weights
        return (field * weights).sum(dim=("latitude", "longitude"))


def build_mass_integrator(
    level: xr.DataArray,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    constants: MarsConstants = MARS,
    *,
    level_bounds: xr.DataArray | None = None,
    latitude_cell_bounds: xr.DataArray | None = None,
    longitude_cell_bounds: xr.DataArray | None = None,
) -> MassIntegrator:
    """Build a phase-1 Mars mass integrator from 1D coordinates."""

    level = normalize_coordinate(level, "level")
    latitude = normalize_coordinate(latitude, "latitude")
    longitude = normalize_coordinate(longitude, "longitude")
    resolved_level_bounds = _get_explicit_level_bounds(level, bounds=level_bounds)
    return MassIntegrator(
        delta_p=delta_p(level, bounds=resolved_level_bounds),
        cell_area=cell_area(
            latitude,
            longitude,
            radius=constants.a,
            latitude_cell_bounds=latitude_cell_bounds,
            longitude_cell_bounds=longitude_cell_bounds,
        ),
        zonal_band_area=zonal_band_area(
            latitude,
            radius=constants.a,
            latitude_cell_bounds=latitude_cell_bounds,
        ),
        level_bounds=resolved_level_bounds,
        constants=constants,
    )


def integrate_mass_full(
    field: xr.DataArray,
    integrator: MassIntegrator | None = None,
    *,
    level: xr.DataArray | None = None,
    latitude: xr.DataArray | None = None,
    longitude: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
    level_bounds: xr.DataArray | None = None,
    latitude_cell_bounds: xr.DataArray | None = None,
    longitude_cell_bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Integrate a full 4D field over atmospheric mass."""

    field = normalize_field(field, "field")
    if integrator is None:
        level = level if level is not None else field.coords["level"]
        latitude = latitude if latitude is not None else field.coords["latitude"]
        longitude = longitude if longitude is not None else field.coords["longitude"]
        integrator = build_mass_integrator(
            level,
            latitude,
            longitude,
            constants=constants,
            level_bounds=level_bounds,
            latitude_cell_bounds=latitude_cell_bounds,
            longitude_cell_bounds=longitude_cell_bounds,
        )
    return integrator.integrate_full(field)


def integrate_mass_zonal(
    field: xr.DataArray,
    integrator: MassIntegrator | None = None,
    *,
    level: xr.DataArray | None = None,
    latitude: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
    level_bounds: xr.DataArray | None = None,
    latitude_cell_bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Integrate an already zonal-mean field over atmospheric mass."""

    field = normalize_zonal_field(field, "field")
    if integrator is None:
        level = level if level is not None else field.coords["level"]
        latitude = latitude if latitude is not None else field.coords["latitude"]
        if latitude is None:
            raise ValueError("A latitude coordinate is required to build a zonal mass integrator.")
        integrator = MassIntegrator(
            delta_p=delta_p(level, bounds=level_bounds),
            cell_area=cell_area(
                latitude,
                _default_global_longitude(),
                radius=constants.a,
                latitude_cell_bounds=latitude_cell_bounds,
            ),
            zonal_band_area=zonal_band_area(
                latitude,
                radius=constants.a,
                latitude_cell_bounds=latitude_cell_bounds,
            ),
            constants=constants,
        )
    return integrator.integrate_zonal(field)


def integrate_surface(
    field: xr.DataArray,
    integrator: MassIntegrator | None = None,
    *,
    latitude: xr.DataArray | None = None,
    longitude: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
    latitude_cell_bounds: xr.DataArray | None = None,
    longitude_cell_bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Integrate a surface field using ``dσ / g`` weights."""

    field = normalize_surface_field(field, "field")
    if integrator is None:
        latitude = latitude if latitude is not None else field.coords["latitude"]
        longitude = longitude if longitude is not None else field.coords["longitude"]
        if latitude is None or longitude is None:
            raise ValueError("Latitude and longitude are required to build a surface integrator.")
        weights = (
            cell_area(
                latitude,
                longitude,
                radius=constants.a,
                latitude_cell_bounds=latitude_cell_bounds,
                longitude_cell_bounds=longitude_cell_bounds,
            )
            / constants.g
        )
        return (field * weights).sum(dim=("latitude", "longitude"))
    return integrator.integrate_surface(field)


def _default_global_longitude() -> xr.DataArray:
    """Return a minimal regular full-global longitude ring."""

    return xr.DataArray(
        np.asarray([0.0, 90.0, 180.0, 270.0]),
        dims=("longitude",),
        coords={"longitude": [0.0, 90.0, 180.0, 270.0]},
    )


__all__ = [
    "pressure_level_edges",
    "delta_p",
    "MassIntegrator",
    "build_mass_integrator",
    "integrate_mass_full",
    "integrate_mass_zonal",
    "integrate_surface",
]
