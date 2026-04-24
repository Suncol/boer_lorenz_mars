"""Internal validation helpers shared by the phase-1 Mars exact package."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import xarray as xr


FIELD_DIMS = ("time", "level", "latitude", "longitude")
ZONAL_DIMS = ("time", "level", "latitude")
SURFACE_DIMS = ("time", "latitude", "longitude")
SURFACE_ZONAL_DIMS = ("time", "latitude")
_LONGITUDE_RING_ATOL = 1.0e-6
_LONGITUDE_RING_RTOL = 1.0e-8


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


def _validate_longitude(
    longitude: xr.DataArray,
    *,
    require_regular_global_ring: bool = True,
) -> xr.DataArray:
    values = np.asarray(longitude.values, dtype=float)
    if values.size < 4:
        raise ValueError("Coordinate 'longitude' must contain at least four points.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Coordinate 'longitude' must contain only finite degree values.")
    diffs = np.diff(values)
    if not np.all(diffs > 0.0):
        raise ValueError(
            "Coordinate 'longitude' must be strictly increasing and must not include "
            "a duplicate cyclic endpoint."
        )

    cyclic_gap = (values[0] + 360.0) - values[-1]
    if cyclic_gap <= 0.0:
        raise ValueError("Coordinate 'longitude' must define a full global ring.")

    if require_regular_global_ring:
        spacings = np.concatenate([diffs, np.asarray([cyclic_gap], dtype=float)])
        nominal_spacing = 360.0 / float(values.size)
        if not np.allclose(
            spacings,
            nominal_spacing,
            rtol=_LONGITUDE_RING_RTOL,
            atol=_LONGITUDE_RING_ATOL,
        ):
            raise ValueError(
                "Coordinate 'longitude' must be an equally spaced full global ring when "
                "longitude bounds are not provided; pass explicit longitude bounds for "
                "non-uniform grids."
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


def normalize_theta_mask(theta_mask: xr.DataArray, name: str = "theta_mask") -> xr.DataArray:
    """Validate a full-grid above-ground mask/coverage field."""

    theta_mask = normalize_field(theta_mask, name)
    try:
        values = np.asarray(theta_mask.values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must contain numeric or boolean mask values.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name!r} must contain only finite mask values.")
    if np.any(values < -1.0e-12) or np.any(values > 1.0 + 1.0e-12):
        raise ValueError(
            f"{name!r} must contain above-ground mask or coverage values in [0, 1]; "
            "pass physical potential temperature separately as 'potential_temperature_field'."
        )
    result = theta_mask.astype(float)
    result.name = theta_mask.name
    result.attrs.update(theta_mask.attrs)
    return result


def normalize_bool_mask(mask: xr.DataArray, name: str = "valid_mask") -> xr.DataArray:
    """Validate a full-grid boolean mask, accepting exact 0/1 numeric values."""

    mask = normalize_field(mask, name)
    if np.issubdtype(mask.dtype, np.bool_):
        return mask.astype(bool)
    try:
        values = np.asarray(mask.values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must contain boolean or 0/1 mask values.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name!r} must contain only finite mask values.")
    is_zero_or_one = np.isclose(values, 0.0, rtol=0.0, atol=1.0e-12) | np.isclose(
        values,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    if not np.all(is_zero_or_one):
        raise ValueError(f"{name!r} must be boolean or contain only 0/1 mask values.")
    result = xr.where(np.isclose(mask.astype(float), 1.0, rtol=0.0, atol=1.0e-12), True, False)
    result.name = mask.name
    result.attrs.update(mask.attrs)
    return result


def resolve_deprecated_theta_mask(
    theta_mask: xr.DataArray | None,
    theta: xr.DataArray | None,
    *,
    required: bool = True,
) -> xr.DataArray | None:
    """Resolve the new ``theta_mask`` argument and deprecated ``theta`` spelling."""

    if theta is not None:
        warnings.warn(
            "'theta' is deprecated for above-ground masks; use 'theta_mask' instead. "
            "Physical potential temperature should be passed as 'potential_temperature_field'.",
            FutureWarning,
            stacklevel=3,
        )
    if theta_mask is not None and theta is not None:
        raise ValueError("Pass only one of 'theta_mask' or deprecated 'theta'.")
    resolved = theta_mask if theta_mask is not None else theta
    if resolved is None:
        if required:
            raise TypeError("A full-grid above-ground mask is required as 'theta_mask'.")
        return None
    return normalize_theta_mask(resolved, "theta_mask")


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


def normalize_surface_field(field: xr.DataArray, name: str) -> xr.DataArray:
    """Validate and transpose a surface field to ``(time, latitude, longitude)``."""

    field = require_dataarray(field, name)
    if set(field.dims) != set(SURFACE_DIMS):
        raise ValueError(
            f"{name!r} must contain exactly the dims {SURFACE_DIMS}; got {field.dims!r}."
        )

    field = field.transpose(*SURFACE_DIMS)
    _validate_latitude(_require_1d_coord(field.coords["latitude"], "latitude"))
    _validate_longitude(_require_1d_coord(field.coords["longitude"], "longitude"))
    return field


def normalize_surface_zonal_field(field: xr.DataArray, name: str) -> xr.DataArray:
    """Validate and transpose a zonal surface field to ``(time, latitude)``."""

    field = require_dataarray(field, name)
    if set(field.dims) != set(SURFACE_ZONAL_DIMS):
        raise ValueError(
            f"{name!r} must contain exactly the dims {SURFACE_ZONAL_DIMS}; got {field.dims!r}."
        )

    field = field.transpose(*SURFACE_ZONAL_DIMS)
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


def ensure_matching_surface_coordinates(
    reference: xr.DataArray,
    others: Iterable[xr.DataArray],
) -> None:
    """Require the canonical surface coordinates to match exactly across fields."""

    reference = normalize_surface_field(reference, "reference")
    for idx, other in enumerate(others):
        other = normalize_surface_field(other, f"field_{idx}")
        for coord_name in SURFACE_DIMS:
            if coord_name == "time":
                equal = np.array_equal(reference[coord_name].values, other[coord_name].values)
            else:
                equal = np.allclose(reference[coord_name].values, other[coord_name].values)
            if not equal:
                raise ValueError(
                    f"Coordinate {coord_name!r} of field_{idx} does not match the reference surface field."
                )
