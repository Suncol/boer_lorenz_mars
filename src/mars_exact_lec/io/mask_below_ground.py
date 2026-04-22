"""Below-ground masking utilities for pressure-coordinate Mars diagnostics."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._validation import normalize_field, require_dataarray


def _coordinate_matches(reference: xr.DataArray, current: xr.DataArray, coord_name: str) -> bool:
    reference_values = reference.coords[coord_name].values
    current_values = current.coords[coord_name].values
    if coord_name == "time":
        return np.array_equal(reference_values, current_values)
    return np.allclose(reference_values, current_values)


def _broadcast_surface_pressure(ps: xr.DataArray, pressure: xr.DataArray) -> xr.DataArray:
    if set(ps.dims) == {"time", "latitude", "longitude"}:
        ps = ps.transpose("time", "latitude", "longitude")
        for coord_name in ("time", "latitude", "longitude"):
            if not _coordinate_matches(pressure, ps, coord_name):
                raise ValueError(f"Coordinate {coord_name!r} of 'ps' does not match the pressure grid.")
        return ps

    if set(ps.dims) == {"latitude", "longitude"}:
        ps = ps.transpose("latitude", "longitude")
        for coord_name in ("latitude", "longitude"):
            if not _coordinate_matches(pressure, ps, coord_name):
                raise ValueError(f"Coordinate {coord_name!r} of 'ps' does not match the pressure grid.")
        return ps.expand_dims(time=pressure.coords["time"]).transpose("time", "latitude", "longitude")

    raise ValueError("'ps' must have dims ('time', 'latitude', 'longitude') or ('latitude', 'longitude').")


def make_theta(pressure: xr.DataArray, ps: xr.DataArray) -> xr.DataArray:
    """Return the sharp above-ground mask Theta.

    Theta is defined as:

        1 where p < ps
        0 where p >= ps

    Parameters
    ----------
    pressure:
        A 4D pressure field with dims ``("time", "level", "latitude", "longitude")``.
        For isobaric data, this is typically the pressure coordinate broadcast to the full grid.
    ps:
        Surface pressure with dims ``("time", "latitude", "longitude")`` or
        ``("latitude", "longitude")``.
    """

    pressure = normalize_field(pressure, "pressure")
    ps = require_dataarray(ps, "ps")

    ps = _broadcast_surface_pressure(ps, pressure)

    theta = xr.where(pressure < ps.broadcast_like(pressure), 1.0, 0.0)
    theta.name = "Theta"
    theta.attrs.update(
        {
            "long_name": "sharp pressure-coordinate above-ground mask",
            "description": "Theta = 1 where p < ps, 0 where p >= ps",
        }
    )
    return theta


def make_below_ground_mask(pressure: xr.DataArray, ps: xr.DataArray) -> xr.DataArray:
    """Return a boolean mask that is True below ground."""

    theta = make_theta(pressure, ps)
    mask = theta == 0.0
    mask.name = "below_ground_mask"
    mask.attrs["long_name"] = "below-ground pressure-coordinate mask"
    return mask


def apply_below_ground_mask(field: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """Mask all below-ground cells of ``field`` using a precomputed Theta."""

    field = normalize_field(field, "field")
    theta = normalize_field(theta, "theta")
    for coord_name in field.dims:
        if not _coordinate_matches(field, theta, coord_name):
            raise ValueError(f"Coordinate {coord_name!r} of 'theta' does not match 'field'.")
    masked = field.where(theta > 0.0)
    masked.name = field.name
    masked.attrs.update(field.attrs)
    masked.attrs["masked_below_ground"] = True
    return masked
