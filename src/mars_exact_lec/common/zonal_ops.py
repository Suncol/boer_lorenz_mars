"""Representative zonal operators for the exact Boer decomposition."""

from __future__ import annotations

import xarray as xr

from .._validation import normalize_field
from .grid_weights import longitude_weights


def zonal_mean(field: xr.DataArray) -> xr.DataArray:
    """Return the longitude mean using cyclic geometric longitude weights."""

    field = normalize_field(field, "field")
    weights = longitude_weights(field.coords["longitude"], normalize=True)
    return field.weighted(weights).sum(dim="longitude")


def theta_coverage(theta: xr.DataArray) -> xr.DataArray:
    """Return the representative longitude coverage ``[Theta]``."""

    theta = normalize_field(theta, "theta")
    return zonal_mean(theta)


def representative_zonal_mean(field: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """Return the representative zonal mean ``[X]_R``."""

    field = normalize_field(field, "field")
    theta = normalize_field(theta, "theta")
    coverage = theta_coverage(theta)
    weighted_mean = zonal_mean(theta * field)
    fallback = zonal_mean(field)
    return xr.where(coverage > 0.0, weighted_mean / coverage, fallback)


def representative_eddy(field: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """Return the representative eddy component ``X* = X - [X]_R``."""

    field = normalize_field(field, "field")
    mean = representative_zonal_mean(field, theta)
    return field - mean.broadcast_like(field)


__all__ = ["zonal_mean", "theta_coverage", "representative_zonal_mean", "representative_eddy"]
