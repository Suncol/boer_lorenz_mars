"""Geopotential helpers for the stage-3 exact topographic terms."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._validation import (
    SURFACE_DIMS,
    ensure_matching_coordinates,
    normalize_field,
    normalize_surface_field,
    require_dataarray,
)
from ..constants_mars import MARS, MarsConstants
from ..io.mask_below_ground import make_theta


def broadcast_surface_field(
    field: xr.DataArray,
    template: xr.DataArray,
    name: str,
) -> xr.DataArray:
    """Broadcast a 2D/3D surface field to the canonical surface grid."""

    template = require_dataarray(template, "template")
    if "level" in template.dims:
        template_surface = normalize_field(template, "template").isel(level=0, drop=True)
    else:
        template_surface = normalize_surface_field(template, "template")

    field = require_dataarray(field, name)
    if set(field.dims) == {"latitude", "longitude"}:
        field = field.transpose("latitude", "longitude").expand_dims(
            time=template_surface.coords["time"]
        )
    elif set(field.dims) == set(SURFACE_DIMS):
        field = normalize_surface_field(field, name)
    else:
        raise ValueError(
            f"{name!r} must have dims ('latitude', 'longitude') or {SURFACE_DIMS}; got {field.dims!r}."
        )

    field = field.transpose(*SURFACE_DIMS).astype(float)
    for coord_name in SURFACE_DIMS:
        reference = template_surface.coords[coord_name]
        current = field.coords[coord_name]
        if coord_name == "time":
            equal = np.array_equal(reference.values, current.values)
        else:
            equal = np.allclose(reference.values, current.values)
        if not equal:
            raise ValueError(f"Coordinate {coord_name!r} of {name!r} does not match the template.")
    return field


def _reconstruct_column_geopotential(
    temperature_column: np.ndarray,
    pressure_column: np.ndarray,
    theta_column: np.ndarray,
    surface_pressure: float,
    surface_geopotential: float,
    *,
    gas_constant: float,
) -> np.ndarray:
    geopotential = np.full_like(temperature_column, np.nan, dtype=float)
    valid = (
        np.isfinite(temperature_column)
        & np.isfinite(pressure_column)
        & (np.asarray(theta_column, dtype=float) > 0.0)
    )
    if not np.any(valid) or not np.isfinite(surface_pressure):
        return geopotential

    valid_indices = np.flatnonzero(valid)
    bottom_idx = int(valid_indices[0])
    bottom_pressure = float(pressure_column[bottom_idx])
    bottom_temperature = float(temperature_column[bottom_idx])
    pressure_ratio = max(float(surface_pressure) / bottom_pressure, 1.0)
    geopotential[bottom_idx] = float(surface_geopotential) + gas_constant * bottom_temperature * np.log(
        pressure_ratio
    )

    above_anchor = valid_indices[valid_indices >= bottom_idx]
    for previous_idx, current_idx in zip(above_anchor[:-1], above_anchor[1:]):
        mean_temperature = 0.5 * (
            float(temperature_column[previous_idx]) + float(temperature_column[current_idx])
        )
        log_ratio = np.log(float(pressure_column[previous_idx]) / float(pressure_column[current_idx]))
        geopotential[current_idx] = geopotential[previous_idx] + gas_constant * mean_temperature * log_ratio

    return geopotential


def _fill_missing_geopotential_column(
    geopotential_column: np.ndarray,
    pressure_column: np.ndarray,
) -> np.ndarray:
    values = np.asarray(geopotential_column, dtype=float)
    pressure = np.asarray(pressure_column, dtype=float)
    valid = np.isfinite(values) & np.isfinite(pressure)
    if not np.any(valid):
        return values
    if np.all(valid):
        return values

    x = np.log(pressure[valid])
    y = values[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    target_x = np.log(pressure)
    filled = np.interp(target_x, x, y)
    if x.size > 1:
        left_slope = (y[1] - y[0]) / (x[1] - x[0])
        right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        left = target_x < x[0]
        right = target_x > x[-1]
        filled[left] = y[0] + left_slope * (target_x[left] - x[0])
        filled[right] = y[-1] + right_slope * (target_x[right] - x[-1])
    else:
        filled[:] = y[0]

    result = values.copy()
    result[~valid] = filled[~valid]
    return result


def reconstruct_hydrostatic_geopotential(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    phis: xr.DataArray,
    *,
    ps: xr.DataArray,
    theta: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Reconstruct geopotential from ``T + p + ps + phis`` using hydrostatic balance."""

    temperature = normalize_field(temperature, "temperature")
    pressure = normalize_field(pressure, "pressure")
    ensure_matching_coordinates(temperature, [pressure])

    if theta is None:
        theta = make_theta(pressure, ps)
    else:
        theta = normalize_field(theta, "theta")
        ensure_matching_coordinates(temperature, [theta])

    surface_pressure = broadcast_surface_field(ps, temperature, "ps")
    surface_geopotential = broadcast_surface_field(phis, temperature, "phis")

    geopotential = xr.apply_ufunc(
        _reconstruct_column_geopotential,
        temperature,
        pressure,
        theta,
        surface_pressure,
        surface_geopotential,
        kwargs={"gas_constant": constants.Rd},
        input_core_dims=[["level"], ["level"], ["level"], [], []],
        output_core_dims=[["level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).transpose("time", "level", "latitude", "longitude")
    geopotential.name = "geopotential"
    geopotential.attrs["units"] = "m2 s-2"
    geopotential.attrs["long_name"] = "hydrostatically reconstructed geopotential"
    geopotential.attrs["reconstructed_hydrostatically"] = True
    return geopotential


def resolve_geopotential(
    *,
    geopotential: xr.DataArray | None = None,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    ps: xr.DataArray | None = None,
    theta: xr.DataArray | None = None,
    valid_mask: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return a canonical geopotential field, reconstructing it when needed."""

    if valid_mask is not None:
        valid_mask = normalize_field(valid_mask, "valid_mask").astype(bool)

    if geopotential is not None:
        geopotential = normalize_field(geopotential, "geopotential")
        if temperature is not None:
            ensure_matching_coordinates(normalize_field(temperature, "temperature"), [geopotential])
        if pressure is not None:
            pressure = normalize_field(pressure, "pressure")
            ensure_matching_coordinates(pressure, [geopotential])
        if theta is not None:
            ensure_matching_coordinates(normalize_field(theta, "theta"), [geopotential])
        if valid_mask is not None:
            ensure_matching_coordinates(geopotential, [valid_mask])
            geopotential = geopotential.where(valid_mask)

        if valid_mask is None:
            all_finite = np.all(np.isfinite(geopotential.values))
        else:
            valid_values = np.asarray(valid_mask.values, dtype=bool)
            all_finite = np.all(np.isfinite(np.asarray(geopotential.values, dtype=float)[valid_values]))

        if all_finite:
            return geopotential

        if temperature is not None and pressure is not None and phis is not None and ps is not None:
            reconstructed = reconstruct_hydrostatic_geopotential(
                temperature=temperature,
                pressure=pressure,
                phis=phis,
                ps=ps,
                theta=theta if theta is not None else valid_mask,
                constants=constants,
            )
            filled = geopotential.where(np.isfinite(geopotential), reconstructed)
            if valid_mask is not None:
                filled = filled.where(valid_mask)
            return filled

        if pressure is None:
            raise ValueError(
                "Explicit geopotential contains non-finite values; provide 'pressure' and either "
                "('temperature', 'ps', 'phis') for hydrostatic continuation or a fully finite field."
            )

        filled = xr.apply_ufunc(
            _fill_missing_geopotential_column,
            geopotential,
            pressure,
            input_core_dims=[["level"], ["level"]],
            output_core_dims=[["level"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        ).transpose("time", "level", "latitude", "longitude")
        filled.name = geopotential.name
        filled.attrs = dict(geopotential.attrs)
        filled.attrs["filled_below_ground"] = True
        if valid_mask is not None:
            filled = filled.where(valid_mask)
        return filled

    if temperature is None or pressure is None or phis is None or ps is None:
        raise ValueError(
            "Resolving geopotential requires either an explicit 'geopotential' field or "
            "'temperature', 'pressure', 'ps', and 'phis'."
        )

    return reconstruct_hydrostatic_geopotential(
        temperature=temperature,
        pressure=pressure,
        phis=phis,
        ps=ps,
        theta=theta if theta is not None else valid_mask,
        constants=constants,
    )


__all__ = [
    "broadcast_surface_field",
    "reconstruct_hydrostatic_geopotential",
    "resolve_geopotential",
]
