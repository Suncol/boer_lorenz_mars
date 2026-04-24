"""Geometric weights on a full global latitude-longitude grid."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

from .._validation import (
    _require_1d_coord,
    _validate_longitude,
    normalize_coordinate,
    require_dataarray,
)
from ..constants_mars import MARS


_BOUNDS_TOL = 1.0e-6


def _candidate_bounds_names(coord_name: str) -> tuple[str, ...]:
    return (
        f"{coord_name}_bounds",
        f"{coord_name}_bnds",
        f"{coord_name}_bound",
        f"{coord_name}_bnd",
    )


def _equal_area_regular_latitude_bounds(size: int) -> np.ndarray:
    sin_edges = np.linspace(1.0, -1.0, size + 1)
    upper = np.rad2deg(np.arcsin(np.clip(sin_edges[:-1], -1.0, 1.0)))
    lower = np.rad2deg(np.arcsin(np.clip(sin_edges[1:], -1.0, 1.0)))
    return np.column_stack([lower, upper])


def _equal_area_regular_latitude_values(size: int) -> np.ndarray:
    bounds = _equal_area_regular_latitude_bounds(size)
    sin_lower = np.sin(np.deg2rad(bounds[:, 0]))
    sin_upper = np.sin(np.deg2rad(bounds[:, 1]))
    return np.rad2deg(np.arcsin(np.clip(0.5 * (sin_lower + sin_upper), -1.0, 1.0)))


def _coerce_bounds(bounds: xr.DataArray, coord_name: str) -> xr.DataArray:
    bounds = xr.DataArray(bounds)
    if bounds.ndim != 2:
        raise ValueError(f"Bounds for {coord_name!r} must be two-dimensional.")

    if bounds.shape[0] == 2 and bounds.shape[1] != 2:
        bounds = xr.DataArray(
            bounds.values.T,
            dims=(coord_name, "bounds"),
            coords={
                coord_name: bounds.coords.get(coord_name, np.arange(bounds.shape[1])),
                "bounds": [0, 1],
            },
        )
    elif bounds.shape[-1] != 2:
        raise ValueError(f"Bounds for {coord_name!r} must end with a size-2 bounds axis.")

    if bounds.dims[0] != coord_name:
        bounds = bounds.rename({bounds.dims[0]: coord_name})
    if bounds.dims[1] != "bounds":
        bounds = bounds.rename({bounds.dims[1]: "bounds"})

    return bounds.transpose(coord_name, "bounds")


def _get_explicit_bounds(coord: xr.DataArray, bounds: xr.DataArray | None = None) -> xr.DataArray | None:
    if bounds is not None:
        return _coerce_bounds(bounds, coord.name)

    bounds_name = coord.attrs.get("bounds")
    if bounds_name and bounds_name in coord.coords:
        return _coerce_bounds(coord.coords[bounds_name], coord.name)

    for candidate in _candidate_bounds_names(coord.name):
        if candidate in coord.coords:
            return _coerce_bounds(coord.coords[candidate], coord.name)
    return None


def _normalize_longitude_coordinate(longitude: xr.DataArray) -> xr.DataArray:
    longitude = require_dataarray(longitude, "longitude")
    longitude = _require_1d_coord(longitude, "longitude")
    if longitude.name != "longitude":
        longitude = longitude.rename("longitude")
    return longitude


def _unwrap_longitude_bound_pair(
    lower: float,
    upper: float,
    center: float,
) -> tuple[float, float]:
    while upper <= lower:
        upper += 360.0
    while center < lower - _BOUNDS_TOL:
        lower -= 360.0
        upper -= 360.0
    while center > upper + _BOUNDS_TOL:
        lower += 360.0
        upper += 360.0
    return lower, upper


def _validate_explicit_longitude_bounds(
    longitude: xr.DataArray,
    bounds: xr.DataArray,
) -> xr.DataArray:
    values = np.asarray(bounds.values, dtype=float)
    centers = np.asarray(longitude.values, dtype=float)
    if values.shape[0] != centers.size:
        raise ValueError("Explicit longitude bounds must have one row per longitude coordinate.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Explicit longitude bounds must contain only finite degree values.")

    unwrapped = np.empty_like(values, dtype=float)
    for idx, (raw_lower, raw_upper) in enumerate(values):
        lower, upper = _unwrap_longitude_bound_pair(
            float(raw_lower),
            float(raw_upper),
            float(centers[idx]),
        )
        if not (lower - _BOUNDS_TOL <= centers[idx] <= upper + _BOUNDS_TOL):
            raise ValueError(
                "Each longitude coordinate must lie inside its corresponding longitude bounds "
                "modulo 360 degrees."
            )
        unwrapped[idx, 0] = lower
        unwrapped[idx, 1] = upper

    widths = unwrapped[:, 1] - unwrapped[:, 0]
    if np.any(widths <= _BOUNDS_TOL):
        raise ValueError("Explicit longitude bounds must define positive-width cells.")
    if values.shape[0] > 1 and not np.allclose(
        unwrapped[:-1, 1],
        unwrapped[1:, 0],
        atol=_BOUNDS_TOL,
    ):
        raise ValueError(
            "Explicit longitude bounds must be contiguous, non-overlapping positive-width "
            "cells that tile exactly 360 degrees."
        )
    if not np.isclose(unwrapped[-1, 1] - unwrapped[0, 0], 360.0, atol=_BOUNDS_TOL):
        raise ValueError(
            "Explicit longitude bounds must be contiguous, non-overlapping positive-width "
            "cells that tile exactly 360 degrees."
        )

    return xr.DataArray(
        unwrapped,
        dims=("longitude", "bounds"),
        coords={"longitude": longitude.values, "bounds": [0, 1]},
        name="longitude_bounds",
    )


def _derive_latitude_bounds(latitude: xr.DataArray) -> xr.DataArray:
    values = np.asarray(latitude.values, dtype=float)
    try:
        grid = infer_grid(latitude)
    except ValueError:
        grid = None

    if grid == "gaussian":
        _, weights = leggauss(values.size)
        weights = weights[::-1]
        north_edges = np.empty(values.size, dtype=float)
        south_edges = np.empty(values.size, dtype=float)
        edge = 1.0
        for idx, weight in enumerate(weights):
            north_edges[idx] = edge
            edge = edge - weight
            south_edges[idx] = edge

        lower = np.rad2deg(np.arcsin(np.clip(south_edges, -1.0, 1.0)))
        upper = np.rad2deg(np.arcsin(np.clip(north_edges, -1.0, 1.0)))
        return xr.DataArray(
            np.column_stack([lower, upper]),
            dims=("latitude", "bounds"),
            coords={"latitude": latitude.values, "bounds": [0, 1]},
            name="latitude_bounds",
        )

    equal_area_reference = _equal_area_regular_latitude_values(values.size)
    if np.allclose(values, equal_area_reference, atol=5e-8):
        return xr.DataArray(
            _equal_area_regular_latitude_bounds(values.size),
            dims=("latitude", "bounds"),
            coords={"latitude": latitude.values, "bounds": [0, 1]},
            name="latitude_bounds",
        )

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] + 0.5 * (values[0] - values[1])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    edges = np.clip(edges, -90.0, 90.0)

    lower = np.minimum(edges[:-1], edges[1:])
    upper = np.maximum(edges[:-1], edges[1:])
    return xr.DataArray(
        np.column_stack([lower, upper]),
        dims=("latitude", "bounds"),
        coords={"latitude": latitude.values, "bounds": [0, 1]},
        name="latitude_bounds",
    )


def _derive_longitude_bounds(longitude: xr.DataArray) -> xr.DataArray:
    longitude = _validate_longitude(longitude, require_regular_global_ring=True)
    values = np.asarray(longitude.values, dtype=float)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    cyclic_gap = (values[0] + 360.0) - values[-1]
    edges[0] = values[0] - 0.5 * cyclic_gap
    edges[-1] = values[-1] + 0.5 * cyclic_gap

    widths = np.diff(edges)
    if np.any(widths <= 0.0) or not np.isclose(widths.sum(), 360.0, atol=1e-6):
        raise ValueError(
            "Phase-1 Mars exact diagnostics only support full global longitude rings."
        )

    return xr.DataArray(
        np.column_stack([edges[:-1], edges[1:]]),
        dims=("longitude", "bounds"),
        coords={"longitude": longitude.values, "bounds": [0, 1]},
        name="longitude_bounds",
    )


def latitude_bounds(latitude: xr.DataArray, *, bounds: xr.DataArray | None = None) -> xr.DataArray:
    """Return latitude bounds, using explicit bounds when available."""

    latitude = normalize_coordinate(latitude, "latitude")
    resolved_bounds = _get_explicit_bounds(latitude, bounds=bounds)
    if resolved_bounds is not None:
        values = np.asarray(resolved_bounds.values, dtype=float)
        lower = np.clip(np.minimum(values[:, 0], values[:, 1]), -90.0, 90.0)
        upper = np.clip(np.maximum(values[:, 0], values[:, 1]), -90.0, 90.0)
        order = np.argsort(lower, kind="mergesort")
        lower_sorted = lower[order]
        upper_sorted = upper[order]
        if not np.isclose(lower_sorted[0], -90.0, atol=_BOUNDS_TOL) or not np.isclose(
            upper_sorted[-1],
            90.0,
            atol=_BOUNDS_TOL,
        ):
            raise ValueError("Explicit latitude bounds must tile the full sphere from -90 to 90 degrees.")
        if np.any((upper_sorted - lower_sorted) <= 0.0):
            raise ValueError("Explicit latitude bounds must define positive-width latitude bands.")
        if lower_sorted.size > 1 and not np.allclose(upper_sorted[:-1], lower_sorted[1:], atol=_BOUNDS_TOL):
            raise ValueError("Explicit latitude bounds must be contiguous and non-overlapping.")
        return xr.DataArray(
            np.column_stack([lower, upper]),
            dims=("latitude", "bounds"),
            coords={"latitude": latitude.values, "bounds": [0, 1]},
            name="latitude_bounds",
        )
    return _derive_latitude_bounds(latitude)


def longitude_bounds(longitude: xr.DataArray, *, bounds: xr.DataArray | None = None) -> xr.DataArray:
    """Return longitude bounds, using explicit bounds when available."""

    longitude = _normalize_longitude_coordinate(longitude)
    resolved_bounds = _get_explicit_bounds(longitude, bounds=bounds)
    longitude = _validate_longitude(
        longitude,
        require_regular_global_ring=resolved_bounds is None,
    )
    if resolved_bounds is not None:
        return _validate_explicit_longitude_bounds(longitude, resolved_bounds)
    return _derive_longitude_bounds(longitude)


def infer_grid(latitude: xr.DataArray) -> str:
    """Infer whether the latitude grid is regular or Gaussian."""

    latitude = normalize_coordinate(latitude, "latitude")
    values = np.asarray(latitude.values, dtype=float)
    diffs = np.diff(values)
    if np.allclose(diffs, diffs[0], atol=5e-8):
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] + 0.5 * (values[0] - values[1])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        south, north = (edges[-1], edges[0]) if edges[0] > edges[-1] else (edges[0], edges[-1])
        if not np.isclose(south, -90.0, atol=5e-8) or not np.isclose(north, 90.0, atol=5e-8):
            raise ValueError("Latitude grid is evenly spaced but does not tile the full sphere.")
        return "regular"

    equal_area_reference = _equal_area_regular_latitude_values(values.size)
    if np.allclose(values, equal_area_reference, atol=5e-8):
        return "regular"

    gauss_nodes, _ = leggauss(values.size)
    gaussian_reference = np.rad2deg(np.arcsin(gauss_nodes[::-1]))
    if np.allclose(values, gaussian_reference, atol=5e-8):
        return "gaussian"

    raise ValueError("Latitude grid is neither global regular nor Gaussian.")


def latitude_weights(
    latitude: xr.DataArray,
    grid: str | None = None,
    normalize: bool = False,
    *,
    bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return latitude weights derived from geometric band areas."""

    latitude = normalize_coordinate(latitude, "latitude")
    if grid is not None and grid not in {"regular", "gaussian"}:
        raise ValueError("Parameter 'grid' must be 'regular', 'gaussian', or None.")
    if grid is None:
        grid = infer_grid(latitude)

    resolved_bounds = latitude_bounds(latitude, bounds=bounds)
    lat_south = np.deg2rad(resolved_bounds.isel(bounds=0))
    lat_north = np.deg2rad(resolved_bounds.isel(bounds=1))
    weights = xr.DataArray(
        np.sin(lat_north.values) - np.sin(lat_south.values),
        dims=("latitude",),
        coords={"latitude": latitude.values},
        name=f"{grid}_latitude_weights",
    )
    if normalize:
        weights = weights / weights.sum(dim="latitude")
    return weights


def longitude_weights(
    longitude: xr.DataArray,
    normalize: bool = False,
    *,
    bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return longitude weights from cyclic cell widths."""

    resolved_bounds = longitude_bounds(longitude, bounds=bounds)
    longitude_values = resolved_bounds.coords["longitude"].values
    widths = xr.DataArray(
        np.deg2rad(resolved_bounds.isel(bounds=1).values - resolved_bounds.isel(bounds=0).values),
        dims=("longitude",),
        coords={"longitude": longitude_values},
        name="longitude_widths",
    )
    if normalize:
        widths = widths / widths.sum(dim="longitude")
    return widths


def zonal_band_area(
    latitude: xr.DataArray,
    radius: float = MARS.a,
    *,
    latitude_cell_bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return the full-ring area of each latitude band."""

    latitude = normalize_coordinate(latitude, "latitude")
    resolved_bounds = latitude_bounds(latitude, bounds=latitude_cell_bounds)
    lat_south = np.deg2rad(resolved_bounds.isel(bounds=0))
    lat_north = np.deg2rad(resolved_bounds.isel(bounds=1))
    area = 2.0 * np.pi * radius**2 * (np.sin(lat_north.values) - np.sin(lat_south.values))
    return xr.DataArray(
        area,
        dims=("latitude",),
        coords={"latitude": latitude.values},
        name="zonal_band_area",
        attrs={"units": "m2"},
    )


def cell_area(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    radius: float = MARS.a,
    *,
    latitude_cell_bounds: xr.DataArray | None = None,
    longitude_cell_bounds: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return the area of each latitude-longitude grid cell."""

    latitude = normalize_coordinate(latitude, "latitude")
    band_area = zonal_band_area(latitude, radius=radius, latitude_cell_bounds=latitude_cell_bounds)
    lon_widths = longitude_weights(longitude, normalize=True, bounds=longitude_cell_bounds)
    area = band_area * lon_widths
    area = area.transpose("latitude", "longitude")
    area.name = "cell_area"
    area.attrs["units"] = "m2"
    return area


__all__ = [
    "infer_grid",
    "latitude_weights",
    "longitude_weights",
    "latitude_bounds",
    "longitude_bounds",
    "zonal_band_area",
    "cell_area",
]
