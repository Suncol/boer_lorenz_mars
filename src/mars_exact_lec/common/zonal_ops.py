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


def weighted_coverage(weight: xr.DataArray) -> xr.DataArray:
    """Return the longitudinal coverage implied by a full-grid weight field."""

    weight = normalize_field(weight, "weight")
    return zonal_mean(weight)


def weighted_representative_zonal_mean(field: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    """Return the weighted representative zonal mean ``[w X] / [w]``."""

    field = normalize_field(field, "field")
    weight = normalize_field(weight, "weight")
    coverage = weighted_coverage(weight)
    weighted_mean = zonal_mean(weight * field)
    fallback = zonal_mean(field)
    return xr.where(coverage > 0.0, weighted_mean / coverage, fallback)


def weighted_representative_eddy(field: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    """Return the weighted representative eddy component."""

    field = normalize_field(field, "field")
    mean = weighted_representative_zonal_mean(field, weight)
    return field - mean.broadcast_like(field)


def theta_coverage(theta: xr.DataArray) -> xr.DataArray:
    """Return the representative longitude coverage ``[Theta]``."""

    theta = normalize_field(theta, "theta")
    return weighted_coverage(theta)


def representative_zonal_mean(field: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """Return the representative zonal mean ``[X]_R``."""

    field = normalize_field(field, "field")
    theta = normalize_field(theta, "theta")
    return weighted_representative_zonal_mean(field, theta)


def representative_eddy(field: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """Return the representative eddy component ``X* = X - [X]_R``."""

    field = normalize_field(field, "field")
    return weighted_representative_eddy(field, theta)


__all__ = [
    "zonal_mean",
    "weighted_coverage",
    "weighted_representative_zonal_mean",
    "weighted_representative_eddy",
    "theta_coverage",
    "representative_zonal_mean",
    "representative_eddy",
]
