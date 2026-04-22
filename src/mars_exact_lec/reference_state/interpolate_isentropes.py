"""Isentropic interpolation helpers for the phase-2 reference-state solver."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .._validation import FIELD_DIMS, normalize_coordinate, normalize_field, require_dataarray
from ..common.grid_weights import cell_area as build_cell_area
from ..common.integrals import MassIntegrator, pressure_level_edges as _pressure_level_edges
from ..constants_mars import MARS, MarsConstants

SURFACE_DIMS = ("time", "latitude", "longitude")
ISENTROPIC_DIM = "isentropic_level"
ISENTROPIC_LAYER_DIM = "isentropic_layer"


def pressure_level_edges(level: xr.DataArray) -> xr.DataArray:
    """Return pressure interfaces using the shared mass-integrator convention."""

    return _pressure_level_edges(level)


def normalize_isentropic_coordinate(
    theta_levels: xr.DataArray | Sequence[float],
    *,
    name: str = ISENTROPIC_DIM,
) -> xr.DataArray:
    """Return a strictly increasing one-dimensional isentropic coordinate."""

    if isinstance(theta_levels, xr.DataArray):
        coord = require_dataarray(theta_levels, name)
        if coord.ndim != 1:
            raise ValueError(f"{name!r} must be one-dimensional.")
        if coord.dims != (name,):
            coord = coord.rename({coord.dims[0]: name}).transpose(name)
    else:
        values = np.asarray(theta_levels, dtype=float)
        coord = xr.DataArray(values, dims=(name,), coords={name: values}, name=name)

    values = np.asarray(coord.values, dtype=float)
    if values.size == 0:
        raise ValueError(f"{name!r} must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name!r} must contain only finite values.")
    if np.any(values <= 0.0):
        raise ValueError(f"{name!r} must contain strictly positive potential temperatures.")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name!r} must be strictly increasing.")

    return xr.DataArray(
        values,
        dims=(name,),
        coords={name: values},
        name=coord.name or name,
        attrs={"units": coord.attrs.get("units", "K"), **dict(coord.attrs)},
    )


def isentropic_interfaces(theta_levels: xr.DataArray | Sequence[float]) -> xr.DataArray:
    """Return interface values derived from isentropic layer centers."""

    levels = normalize_isentropic_coordinate(theta_levels, name=ISENTROPIC_DIM)
    values = np.asarray(levels.values, dtype=float)
    if values.size < 2:
        raise ValueError("At least two isentropic centers are required to derive interfaces.")

    interfaces = np.empty(values.size + 1, dtype=float)
    interfaces[1:-1] = 0.5 * (values[:-1] + values[1:])
    interfaces[0] = values[0] - 0.5 * (values[1] - values[0])
    interfaces[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if np.any(interfaces <= 0.0):
        raise ValueError("Derived isentropic interfaces must remain strictly positive.")

    return xr.DataArray(
        interfaces,
        dims=(ISENTROPIC_DIM,),
        coords={ISENTROPIC_DIM: interfaces},
        name="isentropic_interfaces",
        attrs={"units": levels.attrs.get("units", "K")},
    )


def potential_temperature(
    temperature: xr.DataArray,
    pressure: xr.DataArray | None = None,
    *,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return dry potential temperature ``theta = T (p00 / p)^kappa``."""

    temperature = normalize_field(temperature, "temperature")
    pressure_field = _broadcast_pressure_field(temperature, pressure)
    theta = temperature * (constants.p00 / pressure_field) ** constants.kappa
    theta.name = "potential_temperature"
    theta.attrs["units"] = "K"
    theta.attrs["long_name"] = "dry potential temperature"
    return theta


def interpolate_pressure_to_isentropes(
    potential_temperature_field: xr.DataArray,
    pressure: xr.DataArray | None,
    theta_levels: xr.DataArray | Sequence[float],
    theta_mask: xr.DataArray | None = None,
    *,
    repair_monotonic: bool = True,
) -> xr.DataArray:
    """Interpolate pressure to target isentropes column by column.

    The interpolation is conservative by construction:

    - only above-ground points contribute when ``theta_mask`` is supplied;
    - no extrapolation is performed outside the resolved local theta range;
    - non-monotone columns are either repaired with a non-decreasing envelope or
      rejected entirely when ``repair_monotonic=False``.
    """

    metadata = interpolate_pressure_to_isentropes_metadata(
        potential_temperature_field,
        pressure,
        theta_levels,
        theta_mask=theta_mask,
        repair_monotonic=repair_monotonic,
    )
    return metadata["pressure_on_isentrope"]


def interpolate_pressure_to_isentropes_metadata(
    potential_temperature_field: xr.DataArray,
    pressure: xr.DataArray | None,
    theta_levels: xr.DataArray | Sequence[float],
    theta_mask: xr.DataArray | None = None,
    *,
    repair_monotonic: bool = True,
) -> xr.Dataset:
    """Return interpolated isentropic pressures together with column diagnostics.

    The returned metadata distinguishes pressure-level centers from pressure
    interfaces:

    - ``column_top_pressure`` / ``column_bottom_pressure`` describe the topmost
      and bottommost resolved pressure-level centers used by the interpolation;
    - ``column_top_edge_pressure`` / ``column_bottom_edge_pressure`` describe
      the corresponding pressure-level interfaces and are the authoritative
      bounds for phase-2 discrete mass statistics.
    """

    theta = normalize_field(potential_temperature_field, "potential_temperature_field")
    pressure_field = _broadcast_pressure_field(theta, pressure)
    mask_field = _broadcast_mask_field(theta, theta_mask)
    targets = normalize_isentropic_coordinate(theta_levels, name=ISENTROPIC_DIM)
    level_edges = pressure_level_edges(theta.coords["level"])

    (
        pressure_on_isentrope,
        in_range,
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
        _interpolate_column_to_isentropes,
        theta,
        pressure_field,
        mask_field,
        targets,
        level_edges,
        kwargs={"repair_monotonic": repair_monotonic},
        input_core_dims=[["level"], ["level"], ["level"], [ISENTROPIC_DIM], ["level_edge"]],
        output_core_dims=[
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
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, bool, float, float, float, float, float, float, np.int64, np.int64, np.int64],
    )

    pressure_on_isentrope = pressure_on_isentrope.transpose(
        "time", ISENTROPIC_DIM, "latitude", "longitude"
    )
    in_range = in_range.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")

    data = xr.Dataset(
        data_vars={
            "pressure_on_isentrope": pressure_on_isentrope,
            "isentrope_in_range": in_range,
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
            "monotonic_policy": "repair_envelope" if repair_monotonic else "strict_reject",
            "extrapolation": "disabled",
        },
    )
    data["pressure_on_isentrope"].attrs.update({"units": "Pa", "long_name": "pressure on target isentrope"})
    data["isentrope_in_range"].attrs["long_name"] = "target isentrope lies within resolved column theta range"
    data["column_theta_min"].attrs.update({"units": "K", "long_name": "minimum resolved potential temperature in column"})
    data["column_theta_max"].attrs.update({"units": "K", "long_name": "maximum resolved potential temperature in column"})
    data["column_top_pressure"].attrs.update({"units": "Pa", "long_name": "top resolved pressure-level center used in the column"})
    data["column_bottom_pressure"].attrs.update({"units": "Pa", "long_name": "bottom resolved pressure-level center used in the column"})
    data["column_top_edge_pressure"].attrs.update({"units": "Pa", "long_name": "upper interface of the top resolved pressure level in the column"})
    data["column_bottom_edge_pressure"].attrs.update({"units": "Pa", "long_name": "lower interface of the bottom resolved pressure level in the column"})
    data["valid_level_count"].attrs["long_name"] = "number of above-ground resolved pressure levels used for interpolation"
    data["monotonic_violations"].attrs["long_name"] = "count of negative theta steps before any monotonic repair"
    data["monotonic_repairs"].attrs["long_name"] = "count of theta values raised by the monotonic envelope"
    return data


def isentropic_layer_mass_statistics(
    interpolation: xr.Dataset,
    *,
    surface_pressure: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    area: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> xr.Dataset:
    """Return per-column and global mass statistics for isentropic layers.

    ``interpolation`` must come from :func:`interpolate_pressure_to_isentropes_metadata`
    and its isentropic coordinate is interpreted as ordered layer interfaces.
    Phase 2 always returns discrete whole-cell mass statistics consistent with
    the shared ``Theta + delta_p`` convention. ``surface_pressure`` is accepted
    for interface compatibility only and does not affect the main outputs.
    """

    del surface_pressure

    interpolation = _normalize_interpolation_dataset(interpolation)
    interfaces = normalize_isentropic_coordinate(interpolation.coords[ISENTROPIC_DIM], name=ISENTROPIC_DIM)
    if interfaces.size < 2:
        raise ValueError("At least two isentropic interfaces are required for layer statistics.")

    cell_area = _resolve_cell_area(interpolation, integrator=integrator, area=area)
    top_pressure = interpolation["column_top_edge_pressure"]
    bottom_pressure = interpolation["column_bottom_edge_pressure"]

    pressure_on_isentrope = interpolation["pressure_on_isentrope"]
    theta_min = interpolation["column_theta_min"]
    theta_max = interpolation["column_theta_max"]
    valid_columns = interpolation["valid_level_count"] > 0

    interface_pressure = xr.where(
        interfaces <= theta_min,
        bottom_pressure,
        xr.where(interfaces >= theta_max, top_pressure, pressure_on_isentrope),
    )
    interface_pressure = interface_pressure.where(valid_columns)
    interface_pressure = interface_pressure.transpose("time", ISENTROPIC_DIM, "latitude", "longitude")
    interface_pressure.name = "interface_pressure"
    interface_pressure.attrs.update(
        {
            "units": "Pa",
            "long_name": "pressure at isentropic interfaces after discrete whole-cell clipping",
        }
    )

    lower = interface_pressure.isel({ISENTROPIC_DIM: slice(None, -1)}).rename({ISENTROPIC_DIM: ISENTROPIC_LAYER_DIM})
    upper = interface_pressure.isel({ISENTROPIC_DIM: slice(1, None)}).rename({ISENTROPIC_DIM: ISENTROPIC_LAYER_DIM})
    layer_values = 0.5 * (interfaces.values[:-1] + interfaces.values[1:])
    layer_coord = xr.DataArray(
        np.asarray(layer_values, dtype=float),
        dims=(ISENTROPIC_LAYER_DIM,),
        coords={ISENTROPIC_LAYER_DIM: np.asarray(layer_values, dtype=float)},
        name=ISENTROPIC_LAYER_DIM,
        attrs={"units": interfaces.attrs.get("units", "K")},
    )
    lower = lower.assign_coords({ISENTROPIC_LAYER_DIM: layer_coord})
    upper = upper.assign_coords({ISENTROPIC_LAYER_DIM: layer_coord})

    layer_pressure_thickness = (lower - upper).clip(min=0.0)
    layer_pressure_thickness.name = "layer_pressure_thickness"
    layer_pressure_thickness.attrs.update({"units": "Pa", "long_name": "pressure thickness of each isentropic layer"})

    layer_mass_per_area = layer_pressure_thickness / constants.g
    layer_mass_per_area.name = "layer_mass_per_area"
    layer_mass_per_area.attrs.update({"units": "kg m-2", "long_name": "discrete whole-cell mass per unit area of each isentropic layer"})

    layer_mass = (layer_mass_per_area * cell_area).sum(dim=("latitude", "longitude"))
    layer_mass.name = "layer_mass"
    layer_mass.attrs.update({"units": "kg", "long_name": "global discrete whole-cell mass of each isentropic layer"})

    column_mass_per_area = ((bottom_pressure - top_pressure).clip(min=0.0) / constants.g).where(valid_columns)
    column_mass_per_area.name = "column_mass_per_area"
    column_mass_per_area.attrs.update({"units": "kg m-2", "long_name": "resolved discrete whole-cell column mass per unit area"})

    column_mass = (column_mass_per_area * cell_area).sum(dim=("latitude", "longitude"))
    column_mass.name = "column_mass"
    column_mass.attrs.update({"units": "kg", "long_name": "resolved total discrete whole-cell atmospheric mass"})

    cumulative_mass_above_per_area = ((interface_pressure - top_pressure).clip(min=0.0) / constants.g).where(valid_columns)
    cumulative_mass_above_per_area.name = "cumulative_mass_above_per_area"
    cumulative_mass_above_per_area.attrs.update(
        {"units": "kg m-2", "long_name": "discrete whole-cell mass per unit area above each isentropic interface"}
    )

    cumulative_mass_above = (cumulative_mass_above_per_area * cell_area).sum(dim=("latitude", "longitude"))
    cumulative_mass_above.name = "cumulative_mass_above"
    cumulative_mass_above.attrs.update({"units": "kg", "long_name": "global discrete whole-cell mass above each isentropic interface"})

    result = xr.Dataset(
        data_vars={
            "interface_pressure": interface_pressure,
            "layer_pressure_thickness": layer_pressure_thickness,
            "layer_mass_per_area": layer_mass_per_area,
            "layer_mass": layer_mass,
            "column_mass_per_area": column_mass_per_area,
            "column_mass": column_mass,
            "cumulative_mass_above_per_area": cumulative_mass_above_per_area,
            "cumulative_mass_above": cumulative_mass_above,
        },
        coords={
            "time": interpolation.coords["time"],
            ISENTROPIC_DIM: interfaces,
            ISENTROPIC_LAYER_DIM: layer_coord,
            "latitude": interpolation.coords["latitude"],
            "longitude": interpolation.coords["longitude"],
        },
        attrs={
            "clipping": "level-edge-limited",
            "extrapolation": "disabled",
            "mass_mode": "discrete_whole_cell_phase2",
            "surface_pressure_behavior": "ignored_for_main_outputs",
        },
    )
    result = xr.merge([interpolation, result], compat="override")
    result.attrs.update(
        {
            "clipping": "level-edge-limited",
            "extrapolation": "disabled",
            "mass_mode": "discrete_whole_cell_phase2",
            "surface_pressure_behavior": "ignored_for_main_outputs",
        }
    )
    result = result.assign_coords(
        {
            "lower_isentrope": xr.DataArray(
                np.asarray(interfaces.values[:-1], dtype=float),
                dims=(ISENTROPIC_LAYER_DIM,),
                coords={ISENTROPIC_LAYER_DIM: layer_coord.values},
                attrs={"units": interfaces.attrs.get("units", "K")},
            ),
            "upper_isentrope": xr.DataArray(
                np.asarray(interfaces.values[1:], dtype=float),
                dims=(ISENTROPIC_LAYER_DIM,),
                coords={ISENTROPIC_LAYER_DIM: layer_coord.values},
                attrs={"units": interfaces.attrs.get("units", "K")},
            ),
        }
    )
    return result


def _broadcast_pressure_field(template: xr.DataArray, pressure: xr.DataArray | None) -> xr.DataArray:
    if pressure is None:
        level = normalize_coordinate(template.coords["level"], "level")
        pressure_field = xr.broadcast(template, level)[1]
    else:
        pressure_da = require_dataarray(pressure, "pressure")
        if pressure_da.dims == ("level",):
            level = normalize_coordinate(pressure_da, "level")
            pressure_field = xr.broadcast(template, level)[1]
        else:
            pressure_field = normalize_field(pressure_da, "pressure")
            _ensure_matching_core_coordinates(template, pressure_field, "pressure")

    pressure_field = pressure_field.astype(float).transpose(*FIELD_DIMS)
    if np.any(np.asarray(pressure_field.values, dtype=float) <= 0.0):
        raise ValueError("Pressure used for isentropic interpolation must remain strictly positive.")
    pressure_field.name = pressure_field.name or "pressure"
    return pressure_field


def _broadcast_mask_field(template: xr.DataArray, theta_mask: xr.DataArray | None) -> xr.DataArray:
    if theta_mask is None:
        mask = xr.where(np.isfinite(template), 1.0, 0.0)
    else:
        mask = normalize_field(theta_mask, "theta_mask")
        _ensure_matching_core_coordinates(template, mask, "theta_mask")
        mask = xr.where(mask > 0.0, 1.0, 0.0)

    mask = mask.transpose(*FIELD_DIMS)
    mask.name = "theta_mask"
    return mask


def _normalize_surface_pressure(surface_pressure: xr.DataArray, template: xr.Dataset) -> xr.DataArray:
    ps = require_dataarray(surface_pressure, "surface_pressure")
    if set(ps.dims) != set(SURFACE_DIMS):
        raise ValueError(
            f"'surface_pressure' must contain exactly the dims {SURFACE_DIMS}; got {ps.dims!r}."
        )
    ps = ps.transpose(*SURFACE_DIMS)
    for coord_name in SURFACE_DIMS:
        reference = template.coords[coord_name]
        current = ps.coords[coord_name]
        if coord_name == "time":
            equal = np.array_equal(reference.values, current.values)
        else:
            equal = np.allclose(reference.values, current.values)
        if not equal:
            raise ValueError(f"Coordinate {coord_name!r} of 'surface_pressure' does not match interpolation.")
    if np.any(np.asarray(ps.values, dtype=float) <= 0.0):
        raise ValueError("'surface_pressure' must contain strictly positive pressures in Pa.")
    return ps.astype(float)


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


def _normalize_interpolation_dataset(interpolation: xr.Dataset) -> xr.Dataset:
    if not isinstance(interpolation, xr.Dataset):
        raise TypeError("'interpolation' must be an xarray.Dataset.")

    required_vars = {
        "pressure_on_isentrope",
        "column_theta_min",
        "column_theta_max",
        "column_top_pressure",
        "column_bottom_pressure",
        "column_top_edge_pressure",
        "column_bottom_edge_pressure",
        "valid_level_count",
    }
    missing = required_vars.difference(interpolation.data_vars)
    if missing:
        raise ValueError(f"'interpolation' is missing required variables: {sorted(missing)!r}.")

    pressure_on_isentrope = interpolation["pressure_on_isentrope"]
    expected_dims = ("time", ISENTROPIC_DIM, "latitude", "longitude")
    if pressure_on_isentrope.dims != expected_dims:
        raise ValueError(
            f"'pressure_on_isentrope' must have dims {expected_dims}; got {pressure_on_isentrope.dims!r}."
        )
    for coord_name in ("time", "latitude", "longitude"):
        if coord_name not in interpolation.coords:
            raise ValueError(f"'interpolation' is missing coordinate {coord_name!r}.")
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


def _interpolate_column_to_isentropes(
    theta_column: np.ndarray,
    pressure_column: np.ndarray,
    mask_column: np.ndarray,
    targets: np.ndarray,
    level_edges: np.ndarray,
    *,
    repair_monotonic: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float, float, int, int, int]:
    valid = (
        np.isfinite(theta_column)
        & np.isfinite(pressure_column)
        & np.isfinite(mask_column)
        & (mask_column > 0.0)
    )
    n_targets = targets.size
    pressure_out = np.full(n_targets, np.nan, dtype=float)
    in_range = np.zeros(n_targets, dtype=bool)

    if not np.any(valid):
        return pressure_out, in_range, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0

    theta_valid = np.asarray(theta_column[valid], dtype=float)
    pressure_valid = np.asarray(pressure_column[valid], dtype=float)
    valid_indices = np.flatnonzero(valid)
    valid_count = int(theta_valid.size)

    theta_steps = np.diff(theta_valid)
    monotonic_violations = int(np.count_nonzero(theta_steps < 0.0))
    if repair_monotonic:
        repaired_theta = np.maximum.accumulate(theta_valid)
    else:
        if monotonic_violations > 0:
            return (
                pressure_out,
                in_range,
                np.nan,
                np.nan,
                float(pressure_valid[-1]),
                float(pressure_valid[0]),
                float(level_edges[valid_indices[-1] + 1]),
                float(level_edges[valid_indices[0]]),
                valid_count,
                monotonic_violations,
                0,
            )
        repaired_theta = theta_valid

    monotonic_repairs = int(
        np.count_nonzero(~np.isclose(repaired_theta, theta_valid, rtol=1e-10, atol=1e-12))
    )
    unique_theta, unique_pressure = _collapse_duplicate_isentropes(repaired_theta, pressure_valid)
    theta_min = float(unique_theta[0])
    theta_max = float(unique_theta[-1])
    p_bottom = float(pressure_valid[0])
    p_top = float(pressure_valid[-1])
    p_bottom_edge = float(level_edges[valid_indices[0]])
    p_top_edge = float(level_edges[valid_indices[-1] + 1])

    if unique_theta.size == 1:
        matches = np.isclose(targets, unique_theta[0], rtol=1e-10, atol=1e-12)
        pressure_out[matches] = unique_pressure[0]
        in_range[matches] = True
        return (
            pressure_out,
            in_range,
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

    bracketed = (targets >= theta_min) & (targets <= theta_max)
    if np.any(bracketed):
        pressure_out[bracketed] = np.interp(targets[bracketed], unique_theta, unique_pressure)
        in_range[bracketed] = True

    return (
        pressure_out,
        in_range,
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
        if np.isclose(theta_value, theta_unique[-1], rtol=1e-10, atol=1e-12):
            pressure_unique[-1] = float(pressure_value)
        else:
            theta_unique.append(float(theta_value))
            pressure_unique.append(float(pressure_value))

    return np.asarray(theta_unique, dtype=float), np.asarray(pressure_unique, dtype=float)


pressure_at_isentropes = interpolate_pressure_to_isentropes


__all__ = [
    "ISENTROPIC_DIM",
    "ISENTROPIC_LAYER_DIM",
    "pressure_level_edges",
    "normalize_isentropic_coordinate",
    "isentropic_interfaces",
    "potential_temperature",
    "interpolate_pressure_to_isentropes",
    "interpolate_pressure_to_isentropes_metadata",
    "pressure_at_isentropes",
    "isentropic_layer_mass_statistics",
]
