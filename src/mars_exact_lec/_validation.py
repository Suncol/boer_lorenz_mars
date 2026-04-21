"""Internal validation helpers shared by the phase-1 Mars exact package."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr


FIELD_DIMS = ("time", "level", "latitude", "longitude")
ZONAL_DIMS = ("time", "level", "latitude")


def require_dataarray(obj: xr.DataArray, name: str) -> xr.DataArray:
    if not isinstance(obj, xr.DataArray):
        raise TypeError(f"{name!r} must be an xarray.DataArray.")
    return obj


def _require_1d_coord(coord: xr.DataArray, name: str) -> xr.DataArray:
    if coord.ndim != 1 or coord.dims != (name,):
        raise ValueError(f"Coordinate {name!r} must be one-dimensional with dim {name!r}.")
    return coord


def _validate_level(level: xr.DataArray) -> xr.DataArray:
    values = np.asarray(level.values, dtype=float)
    if values.size < 2:
        raise ValueError("Coordinate 'level' must contain at least two pressure levels.")
    if not np.all(np.diff(values) < 0.0):
        raise ValueError(
            "Coordinate 'level' must be strictly descending in pressure from surface to top."
        )
    if np.any(values <= 0.0):
        raise ValueError("Coordinate 'level' must contain strictly positive pressures in Pa.")
    return level


def _validate_latitude(latitude: xr.DataArray) -> xr.DataArray:
    values = np.asarray(latitude.values, dtype=float)
    if values.size < 2:
        raise ValueError("Coordinate 'latitude' must contain at least two global latitudes.")
    if np.any(np.abs(values) > 90.0 + 1e-10):
        raise ValueError("Coordinate 'latitude' must be expressed in degrees within [-90, 90].")
    diffs = np.diff(values)
    if not (np.all(diffs < 0.0) or np.all(diffs > 0.0)):
        raise ValueError("Coordinate 'latitude' must be strictly monotonic.")
    return latitude


def _validate_longitude(longitude: xr.DataArray) -> xr.DataArray:
    values = np.asarray(longitude.values, dtype=float)
    if values.size < 4:
        raise ValueError("Coordinate 'longitude' must contain at least four points.")
    diffs = np.diff(values)
    if not np.all(diffs > 0.0):
        raise ValueError("Coordinate 'longitude' must be strictly increasing.")

    cyclic_gap = (values[0] + 360.0) - values[-1]
    if cyclic_gap <= 0.0:
        raise ValueError("Coordinate 'longitude' must define a full global ring.")

    total_span = np.sum(diffs) + cyclic_gap
    if not np.isclose(total_span, 360.0, atol=1e-6):
        raise ValueError(
            "Phase-1 Mars exact diagnostics only support full global longitude rings."
        )
    return longitude


def normalize_field(field: xr.DataArray, name: str) -> xr.DataArray:
    """Validate and transpose a 4D field to the canonical phase-1 dimension order."""

    field = require_dataarray(field, name)
    if set(field.dims) != set(FIELD_DIMS):
        raise ValueError(
            f"{name!r} must contain exactly the dims {FIELD_DIMS}; got {field.dims!r}."
        )

    field = field.transpose(*FIELD_DIMS)
    _validate_level(_require_1d_coord(field.coords["level"], "level"))
    _validate_latitude(_require_1d_coord(field.coords["latitude"], "latitude"))
    _validate_longitude(_require_1d_coord(field.coords["longitude"], "longitude"))
    return field


def normalize_zonal_field(field: xr.DataArray, name: str) -> xr.DataArray:
    """Validate and transpose an already zonal-mean field."""

    field = require_dataarray(field, name)
    if set(field.dims) != set(ZONAL_DIMS):
        raise ValueError(
            f"{name!r} must contain exactly the dims {ZONAL_DIMS}; got {field.dims!r}."
        )

    field = field.transpose(*ZONAL_DIMS)
    _validate_level(_require_1d_coord(field.coords["level"], "level"))
    _validate_latitude(_require_1d_coord(field.coords["latitude"], "latitude"))
    return field


def normalize_coordinate(coord: xr.DataArray, name: str) -> xr.DataArray:
    """Validate a standalone coordinate array."""

    coord = require_dataarray(coord, name)
    coord = _require_1d_coord(coord, name)
    if name == "level":
        return _validate_level(coord)
    if name == "latitude":
        return _validate_latitude(coord)
    if name == "longitude":
        return _validate_longitude(coord)
    raise ValueError(f"Unsupported coordinate name {name!r}.")


def scalar_coord(value: float, dim: str, attrs: dict[str, str] | None = None) -> xr.DataArray:
    """Create a one-dimensional scalar coordinate DataArray."""

    attrs = attrs or {}
    return xr.DataArray(np.asarray([value], dtype=float), dims=(dim,), coords={dim: [value]}, attrs=attrs)


def ensure_matching_coordinates(reference: xr.DataArray, others: Iterable[xr.DataArray]) -> None:
    """Require the canonical coordinates to match exactly across fields."""

    reference = normalize_field(reference, "reference")
    for idx, other in enumerate(others):
        other = normalize_field(other, f"field_{idx}")
        for coord_name in FIELD_DIMS:
            if coord_name == "time":
                equal = np.array_equal(reference[coord_name].values, other[coord_name].values)
            else:
                equal = np.allclose(reference[coord_name].values, other[coord_name].values)
            if not equal:
                raise ValueError(
                    f"Coordinate {coord_name!r} of field_{idx} does not match the reference field."
                )
