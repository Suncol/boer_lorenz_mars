"""Observed-state preprocessing for the future Koehler (1986) solver branch.

This module implements only the observed-state side of the fixed-isentrope
workflow:

- fixed-theta surface construction;
- surface-aware pressure-on-isentrope interpolation;
- observed isentropic-layer mass statistics.

Reference-profile geometry and pressure-profile iteration remain out of scope
for this phase and live in :mod:`mars_exact_lec.reference_state.koehler1986_geometry`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .._validation import FIELD_DIMS, normalize_field, normalize_surface_field, require_dataarray
from ..common.geopotential import broadcast_surface_field
from ..common.grid_weights import cell_area as build_cell_area
from ..common.integrals import MassIntegrator, build_mass_integrator
from ..common.topography_measure import TopographyAwareMeasure
from ..constants_mars import MARS, MarsConstants
from ..io.mask_below_ground import make_theta
from .interpolate_isentropes import (
    ISENTROPIC_DIM,
    ISENTROPIC_LAYER_DIM,
    normalize_isentropic_coordinate,
    pressure_level_edges,
)

DEFAULT_MONOTONIC_POLICY = "reject"


@dataclass(frozen=True)
class _KoehlerObservedStateContext:
    """Private preprocessing context retained for later K86 geometry/iteration phases."""

    theta: xr.DataArray
    pressure: xr.DataArray
    surface_pressure: xr.DataArray
    ps_effective: xr.DataArray
    surface_potential_temperature: xr.DataArray
    theta_mask: xr.DataArray
    theta_levels: xr.DataArray
    integrator: MassIntegrator
    measure: TopographyAwareMeasure
    observed_state: xr.Dataset


def build_theta_levels(
    potential_temperature: xr.DataArray,
    surface_potential_temperature: xr.DataArray,
    *,
    theta_levels: xr.DataArray | Sequence[float] | None = None,
    theta_increment: float | None = None,
    theta_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return fixed isentropic surfaces for Koehler (1986) preprocessing."""

    theta = normalize_field(potential_temperature, "potential_temperature")
    surface_theta = normalize_surface_field(
        surface_potential_temperature,
        "surface_potential_temperature",
    )
    _ensure_matching_surface_coordinates(theta.isel(level=0, drop=True), surface_theta, "surface_potential_temperature")

    has_explicit_levels = theta_levels is not None
    has_increment = theta_increment is not None
    if has_explicit_levels == has_increment:
        raise ValueError("Provide exactly one of 'theta_levels' or 'theta_increment'.")

    if has_explicit_levels:
        levels = normalize_isentropic_coordinate(theta_levels, name=ISENTROPIC_DIM)
        levels.attrs.update(
            {
                "long_name": "fixed isentropic surfaces for Koehler (1986) preprocessing",
                "theta_level_semantics": "fixed_isentropic_surfaces",
                "construction": "explicit",
            }
        )
        return levels

    increment = float(theta_increment)
    if not np.isfinite(increment) or increment <= 0.0:
        raise ValueError("'theta_increment' must be a strictly positive finite float.")

    if theta_mask is None:
        valid_theta = theta.where(np.isfinite(theta))
    else:
        mask = _normalize_theta_mask(theta, theta_mask)
        valid_theta = theta.where(mask > 0.0)

    valid_theta_values = np.asarray(valid_theta.values, dtype=float)
    surface_theta_values = np.asarray(surface_theta.values, dtype=float)
    finite_above_ground = valid_theta_values[np.isfinite(valid_theta_values)]
    finite_surface = surface_theta_values[np.isfinite(surface_theta_values)]
    if finite_surface.size == 0:
        raise ValueError("Surface potential temperature must contain at least one finite value.")
    if finite_above_ground.size == 0:
        raise ValueError("No finite above-ground potential temperatures remain to build theta levels.")

    theta_start = np.floor(float(np.min(finite_surface)) / increment) * increment
    theta_stop = np.ceil(
        max(float(np.max(finite_above_ground)), float(np.max(finite_surface))) / increment
    ) * increment

    level_values = np.arange(theta_start, theta_stop + 0.5 * increment, increment, dtype=float)
    if level_values.size < 2:
        raise ValueError(
            "Generated theta levels must contain at least two fixed isentropic surfaces; "
            "provide a different 'theta_increment' or explicit 'theta_levels'."
        )

    levels = normalize_isentropic_coordinate(level_values, name=ISENTROPIC_DIM)
    levels.attrs.update(
        {
            "long_name": "fixed isentropic surfaces for Koehler (1986) preprocessing",
            "theta_level_semantics": "fixed_isentropic_surfaces",
            "construction": "uniform_increment",
            "theta_increment": increment,
        }
    )
    return levels


def resolve_surface_potential_temperature(
    *,
    pressure: xr.DataArray,
    surface_pressure: xr.DataArray,
    surface_potential_temperature: xr.DataArray | None = None,
    surface_temperature: xr.DataArray | None = None,
    template: xr.DataArray,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return surface potential temperature on the canonical surface grid."""

    pressure_field = normalize_field(pressure, "pressure")
    template_surface = _normalize_surface_template(template)
    _ensure_matching_surface_coordinates(pressure_field.isel(level=0, drop=True), template_surface, "template")

    surface_pressure_field = broadcast_surface_field(surface_pressure, template_surface, "surface_pressure")
    if np.any(np.asarray(surface_pressure_field.values, dtype=float) <= 0.0):
        raise ValueError("'surface_pressure' must remain strictly positive.")

    if surface_potential_temperature is not None:
        surface_theta = broadcast_surface_field(
            surface_potential_temperature,
            template_surface,
            "surface_potential_temperature",
        ).astype(float)
    elif surface_temperature is not None:
        surface_temperature_field = broadcast_surface_field(
            surface_temperature,
            template_surface,
            "surface_temperature",
        ).astype(float)
        if np.any(np.asarray(surface_temperature_field.values, dtype=float) <= 0.0):
            raise ValueError("'surface_temperature' must remain strictly positive in K.")
        surface_theta = surface_temperature_field * (
            constants.p00 / surface_pressure_field
        ) ** constants.kappa
        surface_theta.name = "surface_potential_temperature"
        surface_theta.attrs.update(
            {
                "units": "K",
                "long_name": "surface potential temperature derived from surface temperature",
                "derived_from": "surface_temperature",
            }
        )
    else:
        raise ValueError(
            "Koehler1986 preprocessing requires 'surface_potential_temperature' or "
            "'surface_temperature'; lowest-model-level theta fallback is not allowed."
        )

    if np.any(np.asarray(surface_theta.values, dtype=float) <= 0.0):
        raise ValueError("'surface_potential_temperature' must remain strictly positive in K.")

    surface_theta = surface_theta.astype(float)
    surface_theta.name = "surface_potential_temperature"
    surface_theta.attrs.update(
        {
            "units": surface_theta.attrs.get("units", "K"),
            "long_name": "surface potential temperature used by Koehler (1986) preprocessing",
        }
    )
    return surface_theta


def interpolate_pressure_to_koehler_isentropes(
    potential_temperature_field: xr.DataArray,
    pressure: xr.DataArray | None,
    surface_pressure: xr.DataArray,
    surface_potential_temperature: xr.DataArray,
    theta_levels: xr.DataArray | Sequence[float],
    *,
    theta_mask: xr.DataArray | None = None,
    level_bounds: xr.DataArray | None = None,
    interpolation_space: str = "exner",
    monotonic_policy: str = DEFAULT_MONOTONIC_POLICY,
) -> xr.Dataset:
    """Interpolate pressure to fixed Koehler (1986) isentropic surfaces."""

    theta = normalize_field(potential_temperature_field, "potential_temperature_field")
    pressure_field = _broadcast_pressure_field(theta, pressure)
    _ensure_matching_core_coordinates(theta, pressure_field, "pressure")

    mask_field = _normalize_theta_mask(theta, theta_mask) if theta_mask is not None else xr.where(
        np.isfinite(theta),
        1.0,
        0.0,
    )
    surface_pressure_field = broadcast_surface_field(surface_pressure, theta, "surface_pressure")
    surface_theta = broadcast_surface_field(
        surface_potential_temperature,
        theta,
        "surface_potential_temperature",
    )
    targets = normalize_isentropic_coordinate(theta_levels, name=ISENTROPIC_DIM)
    level_edges = pressure_level_edges(theta.coords["level"], bounds=level_bounds)

    interpolation_space = _validate_interpolation_space(interpolation_space)
    monotonic_policy = _validate_monotonic_policy(monotonic_policy)

    (
        pressure_on_theta,
        is_below_surface,
        is_above_model_top,
        is_free_atmosphere,
        column_interpolation_valid,
        theta_min,
        theta_max,
        p_top,
        p_bottom,
        p_top_edge,
        p_bottom_edge,
        valid_count,
        monotonic_violations,
        monotonic_repairs,
    ) = xr.apply_ufunc(
        _interpolate_koehler_column,
        theta,
        pressure_field,
        mask_field,
        surface_pressure_field,
        surface_theta,
        targets,
        level_edges,
        kwargs={
            "interpolation_space": interpolation_space,
            "monotonic_policy": monotonic_policy,
            "p00": float(MARS.p00),
            "kappa": float(MARS.kappa),
        },
        input_core_dims=[["level"], ["level"], ["level"], [], [], [ISENTROPIC_DIM], ["level_edge"]],
        output_core_dims=[
            [ISENTROPIC_DIM],
            [ISENTROPIC_DIM],
            [ISENTROPIC_DIM],
            [ISENTROPIC_DIM],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[
            float,
            bool,
            bool,
            bool,
            bool,
            float,
            float,
            float,
            float,
            float,
            float,
            np.int64,
            np.int64,
            np.int64,
        ],
    )

    pressure_on_theta = pressure_on_theta.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")
    is_below_surface = is_below_surface.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")
    is_above_model_top = is_above_model_top.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")
    is_free_atmosphere = is_free_atmosphere.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")

    data = xr.Dataset(
        data_vars={
            "surface_pressure": surface_pressure_field,
            "surface_potential_temperature": surface_theta,
            "pressure_on_theta": pressure_on_theta,
            "is_below_surface": is_below_surface,
            "is_above_model_top": is_above_model_top,
            "is_free_atmosphere": is_free_atmosphere,
            "column_interpolation_valid": column_interpolation_valid,
            "column_theta_min": theta_min,
            "column_theta_max": theta_max,
            "column_top_pressure": p_top,
            "column_bottom_pressure": p_bottom,
            "column_top_edge_pressure": p_top_edge,
            "column_bottom_edge_pressure": p_bottom_edge,
            "valid_level_count": valid_count,
            "monotonic_violations": monotonic_violations,
            "monotonic_repairs": monotonic_repairs,
        },
        coords={
            "time": theta.coords["time"],
            ISENTROPIC_DIM: targets,
            "latitude": theta.coords["latitude"],
            "longitude": theta.coords["longitude"],
        },
        attrs={
            "theta_level_semantics": "fixed_isentropic_surfaces",
            "interpolation_space": interpolation_space,
            "monotonic_policy": monotonic_policy,
            "surface_pressure_behavior": "surface_aware",
        },
    )
    data["surface_pressure"].attrs.update({"units": "Pa", "long_name": "effective surface pressure used by preprocessing"})
    data["surface_potential_temperature"].attrs.update(
        {"units": "K", "long_name": "surface potential temperature used by preprocessing"}
    )
    data["pressure_on_theta"].attrs.update({"units": "Pa", "long_name": "pressure on fixed isentropic surfaces"})
    data["is_below_surface"].attrs["long_name"] = "fixed isentropic surface lies at or below the local surface"
    data["is_above_model_top"].attrs["long_name"] = "fixed isentropic surface lies above the resolved model-top theta range"
    data["is_free_atmosphere"].attrs["long_name"] = "fixed isentropic surface is interpolated within the resolved free atmosphere"
    data["column_interpolation_valid"].attrs["long_name"] = "column passed Koehler preprocessing interpolation checks"
    data["column_theta_min"].attrs.update({"units": "K", "long_name": "minimum theta represented by the surface-aware interpolation profile"})
    data["column_theta_max"].attrs.update({"units": "K", "long_name": "maximum theta represented by the surface-aware interpolation profile"})
    data["column_top_pressure"].attrs.update({"units": "Pa", "long_name": "top resolved pressure-level center used in the column"})
    data["column_bottom_pressure"].attrs.update({"units": "Pa", "long_name": "bottom resolved pressure-level center used in the column"})
    data["column_top_edge_pressure"].attrs.update({"units": "Pa", "long_name": "upper interface of the top resolved pressure level in the column"})
    data["column_bottom_edge_pressure"].attrs.update({"units": "Pa", "long_name": "lower interface of the bottom resolved pressure level in the column"})
    data["valid_level_count"].attrs["long_name"] = "number of above-ground resolved pressure levels used for interpolation"
    data["monotonic_violations"].attrs["long_name"] = "count of negative theta steps before any monotonic repair"
    data["monotonic_repairs"].attrs["long_name"] = "count of theta values raised by monotonic repair"
    return data


def koehler_isentropic_layer_mass_statistics(
    interpolation: xr.Dataset,
    *,
    integrator: MassIntegrator | None = None,
    area: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> xr.Dataset:
    """Return observed isentropic-layer mass statistics for Koehler preprocessing."""

    interpolation = _normalize_koehler_interpolation_dataset(interpolation)
    interfaces = normalize_isentropic_coordinate(interpolation.coords[ISENTROPIC_DIM], name=ISENTROPIC_DIM)
    if interfaces.size < 2:
        raise ValueError("At least two fixed isentropic surfaces are required for layer statistics.")

    cell_area = _resolve_cell_area(interpolation, integrator=integrator, area=area)
    total_area = float(cell_area.sum())

    pressure_on_theta = interpolation["pressure_on_theta"]
    valid_columns = interpolation["column_interpolation_valid"].broadcast_like(pressure_on_theta)
    interface_pressure = xr.where(
        interpolation["is_below_surface"],
        interpolation["surface_pressure"].broadcast_like(pressure_on_theta),
        xr.where(
            interpolation["is_above_model_top"],
            interpolation["column_top_edge_pressure"].broadcast_like(pressure_on_theta),
            pressure_on_theta,
        ),
    ).where(valid_columns)
    interface_pressure = interface_pressure.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")
    interface_pressure.name = "interface_pressure"
    interface_pressure.attrs.update(
        {
            "units": "Pa",
            "long_name": "pressure at fixed isentropic surfaces after surface-aware/top-edge closure",
        }
    )

    lower = interface_pressure.isel({ISENTROPIC_DIM: slice(None, -1)}).rename({ISENTROPIC_DIM: ISENTROPIC_LAYER_DIM})
    upper = interface_pressure.isel({ISENTROPIC_DIM: slice(1, None)}).rename({ISENTROPIC_DIM: ISENTROPIC_LAYER_DIM})
    lower_theta = np.asarray(interfaces.values[:-1], dtype=float)
    upper_theta = np.asarray(interfaces.values[1:], dtype=float)
    layer_values = 0.5 * (lower_theta + upper_theta)
    layer_coord = xr.DataArray(
        layer_values,
        dims=(ISENTROPIC_LAYER_DIM,),
        coords={ISENTROPIC_LAYER_DIM: layer_values},
        name=ISENTROPIC_LAYER_DIM,
        attrs={"units": interfaces.attrs.get("units", "K")},
    )
    lower = lower.assign_coords({ISENTROPIC_LAYER_DIM: layer_coord})
    upper = upper.assign_coords({ISENTROPIC_LAYER_DIM: layer_coord})

    layer_pressure_thickness = (lower - upper).clip(min=0.0)
    layer_pressure_thickness.name = "layer_pressure_thickness"
    layer_pressure_thickness.attrs.update({"units": "Pa", "long_name": "pressure thickness of each fixed isentropic layer"})

    layer_mass_per_area = layer_pressure_thickness / constants.g
    layer_mass_per_area.name = "layer_mass_per_area"
    layer_mass_per_area.attrs.update({"units": "kg m-2", "long_name": "mass per unit area of each fixed isentropic layer"})

    layer_mass = (layer_mass_per_area * cell_area).sum(dim=("latitude", "longitude"))
    layer_mass.name = "layer_mass"
    layer_mass.attrs.update({"units": "kg", "long_name": "global mass of each fixed isentropic layer"})

    area_broadcast = cell_area.broadcast_like(interface_pressure)
    valid_area = xr.where(interface_pressure.notnull(), area_broadcast, 0.0)
    mean_pressure_on_theta = xr.where(
        valid_area.sum(dim=("latitude", "longitude")) > 0.0,
        (interface_pressure * area_broadcast).sum(dim=("latitude", "longitude"))
        / valid_area.sum(dim=("latitude", "longitude")),
        np.nan,
    )
    mean_pressure_on_theta.name = "mean_pressure_on_theta"
    mean_pressure_on_theta.attrs.update({"units": "Pa", "long_name": "area-weighted mean pressure on each fixed isentropic surface"})

    below_surface_area_fraction = (
        xr.where(interpolation["is_below_surface"] & valid_columns, area_broadcast, 0.0).sum(dim=("latitude", "longitude"))
        / total_area
    )
    below_surface_area_fraction.name = "below_surface_area_fraction"
    below_surface_area_fraction.attrs.update({"units": "1", "long_name": "surface area fraction where each fixed isentropic surface lies at or below ground"})

    above_model_top_area_fraction = (
        xr.where(interpolation["is_above_model_top"] & valid_columns, area_broadcast, 0.0).sum(dim=("latitude", "longitude"))
        / total_area
    )
    above_model_top_area_fraction.name = "above_model_top_area_fraction"
    above_model_top_area_fraction.attrs.update({"units": "1", "long_name": "surface area fraction where each fixed isentropic surface lies above the resolved model-top theta range"})

    free_atmosphere_area_fraction = (
        xr.where(interpolation["is_free_atmosphere"] & valid_columns, area_broadcast, 0.0).sum(dim=("latitude", "longitude"))
        / total_area
    )
    free_atmosphere_area_fraction.name = "free_atmosphere_area_fraction"
    free_atmosphere_area_fraction.attrs.update({"units": "1", "long_name": "surface area fraction where each fixed isentropic surface is interpolated in free atmosphere"})

    result = xr.Dataset(
        data_vars={
            "interface_pressure": interface_pressure,
            "layer_pressure_thickness": layer_pressure_thickness,
            "layer_mass_per_area": layer_mass_per_area,
            "layer_mass": layer_mass,
            "mean_pressure_on_theta": mean_pressure_on_theta,
            "below_surface_area_fraction": below_surface_area_fraction,
            "above_model_top_area_fraction": above_model_top_area_fraction,
            "free_atmosphere_area_fraction": free_atmosphere_area_fraction,
        },
        coords={
            "time": interpolation.coords["time"],
            ISENTROPIC_DIM: interfaces,
            ISENTROPIC_LAYER_DIM: layer_coord,
            "latitude": interpolation.coords["latitude"],
            "longitude": interpolation.coords["longitude"],
            "lower_theta": xr.DataArray(
                lower_theta,
                dims=(ISENTROPIC_LAYER_DIM,),
                coords={ISENTROPIC_LAYER_DIM: layer_coord.values},
                attrs={"units": interfaces.attrs.get("units", "K")},
            ),
            "upper_theta": xr.DataArray(
                upper_theta,
                dims=(ISENTROPIC_LAYER_DIM,),
                coords={ISENTROPIC_LAYER_DIM: layer_coord.values},
                attrs={"units": interfaces.attrs.get("units", "K")},
            ),
        },
        attrs={
            "theta_level_semantics": "fixed_isentropic_surfaces",
            "surface_pressure_behavior": "surface_aware",
            "mass_mode": "koehler1986_observed_state",
        },
    )
    result = xr.merge([interpolation, result], compat="override")
    result.attrs.update(
        {
            "theta_level_semantics": "fixed_isentropic_surfaces",
            "surface_pressure_behavior": "surface_aware",
            "mass_mode": "koehler1986_observed_state",
        }
    )
    return result


def _prepare_koehler1986_observed_state(
    potential_temperature: xr.DataArray,
    pressure: xr.DataArray,
    ps: xr.DataArray,
    *,
    surface_potential_temperature: xr.DataArray | None = None,
    surface_temperature: xr.DataArray | None = None,
    theta_levels: xr.DataArray | Sequence[float] | None = None,
    theta_increment: float | None = None,
    level_bounds: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
    surface_pressure_policy: str = "raise",
    pressure_tolerance: float = 1.0e-3,
    interpolation_space: str = "exner",
    monotonic_policy: str = DEFAULT_MONOTONIC_POLICY,
) -> xr.Dataset:
    """Build the Koehler (1986) observed-state preprocessing dataset."""

    return _prepare_koehler1986_observed_context(
        potential_temperature,
        pressure,
        ps,
        surface_potential_temperature=surface_potential_temperature,
        surface_temperature=surface_temperature,
        theta_levels=theta_levels,
        theta_increment=theta_increment,
        level_bounds=level_bounds,
        constants=constants,
        surface_pressure_policy=surface_pressure_policy,
        pressure_tolerance=pressure_tolerance,
        interpolation_space=interpolation_space,
        monotonic_policy=monotonic_policy,
    ).observed_state


def _prepare_koehler1986_observed_context(
    potential_temperature: xr.DataArray,
    pressure: xr.DataArray,
    ps: xr.DataArray,
    *,
    surface_potential_temperature: xr.DataArray | None = None,
    surface_temperature: xr.DataArray | None = None,
    theta_levels: xr.DataArray | Sequence[float] | None = None,
    theta_increment: float | None = None,
    level_bounds: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
    surface_pressure_policy: str = "raise",
    pressure_tolerance: float = 1.0e-3,
    interpolation_space: str = "exner",
    monotonic_policy: str = DEFAULT_MONOTONIC_POLICY,
) -> _KoehlerObservedStateContext:
    """Build the private preprocessing context used by later K86 phases."""

    theta = normalize_field(potential_temperature, "potential_temperature")
    pressure_field = normalize_field(pressure, "pressure")
    _ensure_matching_core_coordinates(theta, pressure_field, "pressure")

    surface_pressure = broadcast_surface_field(ps, theta, "ps")
    integrator = build_mass_integrator(
        theta.coords["level"],
        theta.coords["latitude"],
        theta.coords["longitude"],
        constants=constants,
        level_bounds=level_bounds,
    )
    measure = TopographyAwareMeasure.from_surface_pressure(
        theta.coords["level"],
        surface_pressure,
        integrator,
        level_bounds=level_bounds,
        pressure_tolerance=pressure_tolerance,
        surface_pressure_policy=surface_pressure_policy,
    )
    ps_effective = measure.effective_surface_pressure

    surface_theta = resolve_surface_potential_temperature(
        pressure=pressure_field,
        surface_pressure=ps_effective,
        surface_potential_temperature=surface_potential_temperature,
        surface_temperature=surface_temperature,
        template=surface_pressure,
        constants=constants,
    )
    theta_mask = make_theta(pressure_field, ps_effective)
    resolved_theta_levels = build_theta_levels(
        theta,
        surface_theta,
        theta_levels=theta_levels,
        theta_increment=theta_increment,
        theta_mask=theta_mask,
    )
    interpolation = interpolate_pressure_to_koehler_isentropes(
        theta,
        pressure_field,
        ps_effective,
        surface_theta,
        resolved_theta_levels,
        theta_mask=theta_mask,
        level_bounds=level_bounds,
        interpolation_space=interpolation_space,
        monotonic_policy=monotonic_policy,
    )
    statistics = koehler_isentropic_layer_mass_statistics(
        interpolation,
        integrator=integrator,
        constants=constants,
    )

    observed_state = xr.merge(
        [
            statistics,
            ps_effective.rename("ps_effective"),
            theta_mask.rename("theta_mask"),
        ],
        compat="override",
    )
    observed_state.attrs.update(measure.domain_metadata)
    observed_state.attrs.update(
        {
            "preprocessing_stage": "koehler1986_observed_state",
            "theta_level_semantics": "fixed_isentropic_surfaces",
            "interpolation_space": interpolation_space,
            "monotonic_policy": monotonic_policy,
        }
    )
    observed_state["ps_effective"].attrs.update(
        {
            "units": "Pa",
            "long_name": "effective surface pressure used by Koehler (1986) preprocessing",
        }
    )
    observed_state["theta_mask"].attrs.update(theta_mask.attrs)
    return _KoehlerObservedStateContext(
        theta=theta,
        pressure=pressure_field,
        surface_pressure=surface_pressure,
        ps_effective=ps_effective,
        surface_potential_temperature=surface_theta,
        theta_mask=theta_mask,
        theta_levels=resolved_theta_levels,
        integrator=integrator,
        measure=measure,
        observed_state=observed_state,
    )


def _normalize_theta_mask(template: xr.DataArray, theta_mask: xr.DataArray) -> xr.DataArray:
    mask = normalize_field(theta_mask, "theta_mask")
    _ensure_matching_core_coordinates(template, mask, "theta_mask")
    return xr.where(mask > 0.0, 1.0, 0.0).transpose(*FIELD_DIMS)


def _normalize_surface_template(template: xr.DataArray) -> xr.DataArray:
    template = require_dataarray(template, "template")
    if "level" in template.dims:
        return normalize_field(template, "template").isel(level=0, drop=True)
    return normalize_surface_field(template, "template")


def _broadcast_pressure_field(template: xr.DataArray, pressure: xr.DataArray | None) -> xr.DataArray:
    if pressure is None:
        level = require_dataarray(template.coords["level"], "level")
        pressure_field = xr.broadcast(template, level)[1]
    else:
        pressure_da = require_dataarray(pressure, "pressure")
        if pressure_da.dims == ("level",):
            level = require_dataarray(pressure_da, "pressure")
            pressure_field = xr.broadcast(template, level)[1]
        else:
            pressure_field = normalize_field(pressure_da, "pressure")
    pressure_field = pressure_field.astype(float).transpose(*FIELD_DIMS)
    if np.any(np.asarray(pressure_field.values, dtype=float) <= 0.0):
        raise ValueError("'pressure' must contain strictly positive pressures in Pa.")
    return pressure_field


def _resolve_cell_area(
    interpolation: xr.Dataset,
    *,
    integrator: MassIntegrator | None,
    area: xr.DataArray | None,
) -> xr.DataArray:
    if integrator is not None:
        resolved = integrator.cell_area
    elif area is not None:
        resolved = require_dataarray(area, "area")
    else:
        resolved = build_cell_area(interpolation.coords["latitude"], interpolation.coords["longitude"])

    if set(resolved.dims) != {"latitude", "longitude"}:
        raise ValueError("'area' must have exactly the dims ('latitude', 'longitude').")
    resolved = resolved.transpose("latitude", "longitude")
    lat_match = np.allclose(resolved.coords["latitude"].values, interpolation.coords["latitude"].values)
    lon_match = np.allclose(resolved.coords["longitude"].values, interpolation.coords["longitude"].values)
    if not lat_match or not lon_match:
        raise ValueError("Cell areas must share the interpolation latitude/longitude coordinates.")
    return resolved


def _validate_interpolation_space(interpolation_space: str) -> str:
    normalized = str(interpolation_space).strip().lower()
    if normalized not in {"exner", "pressure"}:
        raise ValueError("'interpolation_space' must be either 'exner' or 'pressure'.")
    return normalized


def _validate_monotonic_policy(monotonic_policy: str) -> str:
    normalized = str(monotonic_policy).strip().lower()
    if normalized not in {"repair", "reject"}:
        raise ValueError("'monotonic_policy' must be either 'repair' or 'reject'.")
    return normalized


def _normalize_koehler_interpolation_dataset(interpolation: xr.Dataset) -> xr.Dataset:
    if not isinstance(interpolation, xr.Dataset):
        raise TypeError("'interpolation' must be an xarray.Dataset.")

    required_vars = {
        "surface_pressure",
        "surface_potential_temperature",
        "pressure_on_theta",
        "is_below_surface",
        "is_above_model_top",
        "is_free_atmosphere",
        "column_interpolation_valid",
        "column_theta_min",
        "column_theta_max",
        "column_top_pressure",
        "column_bottom_pressure",
        "column_top_edge_pressure",
        "valid_level_count",
        "monotonic_violations",
        "monotonic_repairs",
    }
    missing = required_vars.difference(interpolation.data_vars)
    if missing:
        raise ValueError(f"'interpolation' is missing required variables: {sorted(missing)!r}.")

    expected_dims = ("time", ISENTROPIC_DIM, "latitude", "longitude")
    for name in ("pressure_on_theta", "is_below_surface", "is_above_model_top", "is_free_atmosphere"):
        if interpolation[name].dims != expected_dims:
            raise ValueError(f"{name!r} must have dims {expected_dims}; got {interpolation[name].dims!r}.")
    return interpolation


def _ensure_matching_core_coordinates(reference: xr.DataArray, other: xr.DataArray, name: str) -> None:
    for coord_name in FIELD_DIMS:
        ref_coord = reference.coords[coord_name]
        other_coord = other.coords[coord_name]
        if coord_name == "time":
            equal = np.array_equal(ref_coord.values, other_coord.values)
        else:
            equal = np.allclose(ref_coord.values, other_coord.values)
        if not equal:
            raise ValueError(f"Coordinate {coord_name!r} of {name!r} does not match the reference field.")


def _ensure_matching_surface_coordinates(reference: xr.DataArray, other: xr.DataArray, name: str) -> None:
    for coord_name in ("time", "latitude", "longitude"):
        ref_coord = reference.coords[coord_name]
        other_coord = other.coords[coord_name]
        if coord_name == "time":
            equal = np.array_equal(ref_coord.values, other_coord.values)
        else:
            equal = np.allclose(ref_coord.values, other_coord.values)
        if not equal:
            raise ValueError(f"Coordinate {coord_name!r} of {name!r} does not match the reference surface grid.")


def _interpolate_koehler_column(
    theta_column: np.ndarray,
    pressure_column: np.ndarray,
    mask_column: np.ndarray,
    surface_pressure: float,
    surface_theta: float,
    targets: np.ndarray,
    level_edges: np.ndarray,
    *,
    interpolation_space: str,
    monotonic_policy: str,
    p00: float,
    kappa: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    int,
]:
    n_targets = targets.size
    pressure_out = np.full(n_targets, np.nan, dtype=float)
    below_surface = np.zeros(n_targets, dtype=bool)
    above_model_top = np.zeros(n_targets, dtype=bool)
    free_atmosphere = np.zeros(n_targets, dtype=bool)

    if not np.isfinite(surface_pressure) or not np.isfinite(surface_theta):
        return (
            pressure_out,
            below_surface,
            above_model_top,
            free_atmosphere,
            False,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            0,
            0,
        )

    valid = (
        np.isfinite(theta_column)
        & np.isfinite(pressure_column)
        & np.isfinite(mask_column)
        & (mask_column > 0.0)
    )
    if not np.any(valid):
        below_surface = targets <= surface_theta
        above_model_top = targets > surface_theta
        return (
            pressure_out,
            below_surface,
            above_model_top,
            free_atmosphere,
            False,
            float(surface_theta),
            float(surface_theta),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            0,
            0,
        )

    valid_indices = np.flatnonzero(valid)
    theta_valid = np.asarray(theta_column[valid], dtype=float)
    pressure_valid = np.asarray(pressure_column[valid], dtype=float)
    valid_count = int(theta_valid.size)
    p_bottom = float(pressure_valid[0])
    p_top = float(pressure_valid[-1])
    p_bottom_edge = float(level_edges[valid_indices[0]])
    p_top_edge = float(level_edges[valid_indices[-1] + 1])

    theta_profile_raw = np.concatenate([[float(surface_theta)], theta_valid])
    pressure_profile = np.concatenate([[float(surface_pressure)], pressure_valid])
    theta_steps = np.diff(theta_profile_raw)
    monotonic_violations = int(np.count_nonzero(theta_steps < 0.0))
    if monotonic_policy == "reject" and monotonic_violations > 0:
        theta_min = float(theta_profile_raw[0])
        theta_max = float(np.nanmax(theta_profile_raw))
        below_surface = targets <= surface_theta
        above_model_top = targets > theta_max
        free_atmosphere = (~below_surface) & (~above_model_top)
        return (
            pressure_out,
            below_surface,
            above_model_top,
            free_atmosphere,
            False,
            theta_min,
            theta_max,
            p_top,
            p_bottom,
            p_top_edge,
            p_bottom_edge,
            valid_count,
            monotonic_violations,
            0,
        )

    theta_profile = np.maximum.accumulate(theta_profile_raw)
    monotonic_repairs = int(
        np.count_nonzero(~np.isclose(theta_profile, theta_profile_raw, rtol=1.0e-10, atol=1.0e-12))
    )
    theta_unique, pressure_unique = _collapse_duplicate_isentropes(theta_profile, pressure_profile)
    theta_min = float(theta_unique[0])
    theta_max = float(theta_unique[-1])

    below_surface = targets <= surface_theta
    above_model_top = targets > theta_max
    free_atmosphere = (~below_surface) & (~above_model_top)

    if np.any(free_atmosphere):
        if interpolation_space == "pressure":
            interpolant = pressure_unique
            interpolated = np.interp(targets[free_atmosphere], theta_unique, interpolant)
        else:
            exner_unique = np.power(pressure_unique / p00, kappa)
            exner_interp = np.interp(targets[free_atmosphere], theta_unique, exner_unique)
            interpolated = p00 * np.power(np.maximum(exner_interp, 0.0), 1.0 / kappa)
        pressure_out[free_atmosphere] = interpolated
    pressure_out[below_surface] = surface_pressure
    return (
        pressure_out,
        below_surface,
        above_model_top,
        free_atmosphere,
        True,
        theta_min,
        theta_max,
        p_top,
        p_bottom,
        p_top_edge,
        p_bottom_edge,
        valid_count,
        monotonic_violations,
        monotonic_repairs,
    )


def _collapse_duplicate_isentropes(
    theta_profile: np.ndarray,
    pressure_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta_unique = [float(theta_profile[0])]
    pressure_unique = [float(pressure_profile[0])]

    for theta_value, pressure_value in zip(theta_profile[1:], pressure_profile[1:]):
        if np.isclose(theta_value, theta_unique[-1], rtol=1.0e-10, atol=1.0e-12):
            pressure_unique[-1] = float(pressure_value)
        else:
            theta_unique.append(float(theta_value))
            pressure_unique.append(float(pressure_value))

    return np.asarray(theta_unique, dtype=float), np.asarray(pressure_unique, dtype=float)


__all__ = [
    "build_theta_levels",
    "resolve_surface_potential_temperature",
    "interpolate_pressure_to_koehler_isentropes",
    "koehler_isentropic_layer_mass_statistics",
]
