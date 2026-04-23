"""Private geometry and reference-mass helpers for the Koehler (1986) branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .._validation import normalize_surface_field, require_dataarray
from ..common.geopotential import broadcast_surface_field
from ..common.grid_weights import cell_area as build_cell_area
from ..constants_mars import MARS, MarsConstants
from .interpolate_isentropes import ISENTROPIC_DIM, ISENTROPIC_LAYER_DIM, normalize_isentropic_coordinate


def _pressure_to_exner(pressure: np.ndarray | float, constants: MarsConstants = MARS) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=float)
    return np.power(pressure / constants.p00, constants.kappa)


def _exner_to_pressure(exner: np.ndarray | float, constants: MarsConstants = MARS) -> np.ndarray:
    exner = np.asarray(exner, dtype=float)
    return constants.p00 * np.power(np.maximum(exner, 0.0), 1.0 / constants.kappa)


@dataclass(frozen=True)
class _KoehlerProfileGeometry:
    theta_levels: xr.DataArray
    pi_levels: xr.DataArray
    phi_levels: xr.DataArray
    theta_anchor: xr.DataArray
    anchor_layer_index: xr.DataArray
    anchor_fraction: xr.DataArray
    anchor_mode: xr.DataArray
    profile_bottom_pressure: xr.DataArray
    profile_top_pressure: xr.DataArray


@dataclass(frozen=True)
class _KoehlerSurfaceReconstruction:
    pi_s: xr.DataArray
    theta_s_ref: xr.DataArray
    surface_layer_index: xr.DataArray
    surface_valid_mask: xr.DataArray
    phis_relative: xr.DataArray
    reference_surface_pressure: xr.DataArray
    deepest_surface_pressure: xr.DataArray


@dataclass(frozen=True)
class _KoehlerReferenceMass:
    interface_pressure: xr.DataArray
    mean_pressure_on_theta: xr.DataArray
    layer_pressure_thickness: xr.DataArray
    layer_mass_per_area: xr.DataArray
    layer_mass: xr.DataArray


@dataclass(frozen=True)
class _KoehlerGeometryFamily:
    family_name: str
    profile: _KoehlerProfileGeometry
    surface: _KoehlerSurfaceReconstruction
    reference_mass: _KoehlerReferenceMass


def koehler_layer_geopotential_drop(
    theta_lower: np.ndarray | float,
    theta_upper: np.ndarray | float,
    p_lower: np.ndarray | float,
    p_upper: np.ndarray | float,
    *,
    constants: MarsConstants = MARS,
) -> np.ndarray:
    """Return the layer geopotential drop for linear-theta-in-Exner Koehler layers."""

    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)
    p_lower = np.asarray(p_lower, dtype=float)
    p_upper = np.asarray(p_upper, dtype=float)
    if np.any(theta_lower <= 0.0) or np.any(theta_upper <= 0.0):
        raise ValueError("Koehler layer theta values must remain strictly positive.")
    if np.any(p_lower <= 0.0) or np.any(p_upper <= 0.0):
        raise ValueError("Koehler layer pressures must remain strictly positive.")

    exner_lower = _pressure_to_exner(p_lower, constants)
    exner_upper = _pressure_to_exner(p_upper, constants)
    return constants.cp * 0.5 * (theta_lower + theta_upper) * (exner_lower - exner_upper)


def _reference_geopotential_with_anchor_details(
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
    *,
    theta_anchor: float,
    phi_anchor: float = 0.0,
    constants: MarsConstants = MARS,
) -> tuple[np.ndarray, int, float, str]:
    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    if theta_levels.ndim != 1 or pi_levels.ndim != 1:
        raise ValueError("'theta_levels' and 'pi_levels' must be one-dimensional.")
    if theta_levels.size != pi_levels.size:
        raise ValueError("'theta_levels' and 'pi_levels' must have the same length.")
    if theta_levels.size < 2:
        raise ValueError("At least two fixed isentropic levels are required.")
    if np.any(theta_levels <= 0.0) or np.any(pi_levels <= 0.0):
        raise ValueError("Koehler geometry inputs must remain strictly positive.")
    if not np.all(np.diff(theta_levels) > 0.0):
        raise ValueError("'theta_levels' must be strictly increasing.")
    if not np.all(np.diff(pi_levels) < 0.0):
        raise ValueError("'pi_levels' must be strictly decreasing with theta.")
    if not (theta_levels[0] <= theta_anchor <= theta_levels[-1]):
        raise ValueError("'theta_anchor' must lie inside the fixed isentropic-level range.")

    phi_raw = np.zeros(theta_levels.size, dtype=float)
    delta_phi = koehler_layer_geopotential_drop(
        theta_levels[:-1],
        theta_levels[1:],
        pi_levels[:-1],
        pi_levels[1:],
        constants=constants,
    )
    phi_raw[1:] = np.cumsum(delta_phi, dtype=float)

    matching = np.where(np.isclose(theta_levels, theta_anchor, rtol=0.0, atol=1.0e-12))[0]
    if matching.size:
        anchor_index = int(matching[0])
        anchor_fraction = 0.0
        anchor_mode = "exact_level"
        phi_anchor_raw = float(phi_raw[anchor_index])
    else:
        upper_index = int(np.searchsorted(theta_levels, theta_anchor, side="right"))
        upper_index = max(1, min(upper_index, theta_levels.size - 1))
        lower_index = upper_index - 1
        theta_lower = float(theta_levels[lower_index])
        theta_upper = float(theta_levels[upper_index])
        anchor_fraction = float(
            (theta_anchor**2 - theta_lower**2) / (theta_upper**2 - theta_lower**2)
        )
        anchor_fraction = float(np.clip(anchor_fraction, 0.0, 1.0))
        anchor_mode = "inside_layer"
        anchor_index = lower_index
        phi_anchor_raw = float(phi_raw[lower_index] + anchor_fraction * delta_phi[lower_index])

    phi_levels = phi_raw - phi_anchor_raw + float(phi_anchor)
    return phi_levels, anchor_index, anchor_fraction, anchor_mode


def reference_geopotential_from_pressure_profile(
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
    *,
    theta_anchor: float,
    phi_anchor: float = 0.0,
    constants: MarsConstants = MARS,
) -> np.ndarray:
    """Return the reference geopotential on fixed isentropic levels."""

    phi_levels, _, _, _ = _reference_geopotential_with_anchor_details(
        theta_levels,
        pi_levels,
        theta_anchor=theta_anchor,
        phi_anchor=phi_anchor,
        constants=constants,
    )
    return phi_levels


def pressure_and_theta_at_geopotential_in_layer(
    phi_target: np.ndarray | float,
    phi_lower: np.ndarray | float,
    phi_upper: np.ndarray | float,
    p_lower: np.ndarray | float,
    p_upper: np.ndarray | float,
    theta_lower: np.ndarray | float,
    theta_upper: np.ndarray | float,
    *,
    constants: MarsConstants = MARS,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a Koehler layer to pressure and theta at target geopotential."""

    phi_target = np.asarray(phi_target, dtype=float)
    phi_lower = np.asarray(phi_lower, dtype=float)
    phi_upper = np.asarray(phi_upper, dtype=float)
    p_lower = np.asarray(p_lower, dtype=float)
    p_upper = np.asarray(p_upper, dtype=float)
    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)

    if np.any(phi_upper <= phi_lower):
        raise ValueError("Koehler layer geopotential must increase upward.")
    if np.any(phi_target < phi_lower - 1.0e-10) or np.any(phi_target > phi_upper + 1.0e-10):
        raise ValueError("'phi_target' must lie inside the Koehler layer bounds.")

    exner_lower = _pressure_to_exner(p_lower, constants)
    exner_upper = _pressure_to_exner(p_upper, constants)
    same_theta = np.isclose(theta_lower, theta_upper, rtol=0.0, atol=1.0e-12)
    phi_fraction = np.clip((phi_target - phi_lower) / (phi_upper - phi_lower), 0.0, 1.0)

    theta_target = np.where(
        same_theta,
        theta_lower,
        np.sqrt(
            np.maximum(
                theta_lower**2 + phi_fraction * (theta_upper**2 - theta_lower**2),
                0.0,
            )
        ),
    )

    exner_target = np.where(
        same_theta,
        exner_lower - (phi_target - phi_lower) / (constants.cp * theta_lower),
        exner_lower
        + ((theta_target - theta_lower) / (theta_upper - theta_lower)) * (exner_upper - exner_lower),
    )
    pressure_target = _exner_to_pressure(exner_target, constants)
    return pressure_target, theta_target


def reference_surface_from_profile(
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
    phi_levels: np.ndarray,
    phis: np.ndarray,
    *,
    constants: MarsConstants = MARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return surface pressure/theta reconstruction on actual topography."""

    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    phi_levels = np.asarray(phi_levels, dtype=float)
    phis = np.asarray(phis, dtype=float)
    if phis.ndim != 2:
        raise ValueError("'phis' must be two-dimensional on the surface grid.")
    if theta_levels.size != pi_levels.size or theta_levels.size != phi_levels.size:
        raise ValueError("Profile arrays must share the same length.")
    if not np.all(np.diff(phi_levels) > 0.0):
        raise ValueError("'phi_levels' must be strictly increasing.")

    phis_relative = phis - float(np.nanmin(phis))
    if np.any(phis_relative > phi_levels[-1] + 1.0e-10):
        raise ValueError("Actual topography rises above the reconstructed reference-profile top.")

    surface_layer_index = np.searchsorted(phi_levels, phis_relative, side="right") - 1
    surface_layer_index = np.clip(surface_layer_index, 0, theta_levels.size - 2)
    top_hits = np.isclose(phis_relative, phi_levels[-1], rtol=0.0, atol=1.0e-10)
    surface_layer_index[top_hits] = theta_levels.size - 2

    flat_index = surface_layer_index.reshape(-1)
    flat_phi = phis_relative.reshape(-1)
    pressure_flat = np.empty(flat_phi.size, dtype=float)
    theta_flat = np.empty(flat_phi.size, dtype=float)

    for point_index, layer_index in enumerate(flat_index):
        pressure_flat[point_index], theta_flat[point_index] = pressure_and_theta_at_geopotential_in_layer(
            flat_phi[point_index],
            float(phi_levels[layer_index]),
            float(phi_levels[layer_index + 1]),
            float(pi_levels[layer_index]),
            float(pi_levels[layer_index + 1]),
            float(theta_levels[layer_index]),
            float(theta_levels[layer_index + 1]),
            constants=constants,
        )

    return (
        pressure_flat.reshape(phis.shape),
        theta_flat.reshape(phis.shape),
        surface_layer_index.astype(np.int64),
    )


def _build_reference_interface_pressure_field(
    pi_levels: np.ndarray,
    phi_levels: np.ndarray,
    pi_s: np.ndarray,
    phis_relative: np.ndarray,
) -> np.ndarray:
    pi_levels = np.asarray(pi_levels, dtype=float)
    phi_levels = np.asarray(phi_levels, dtype=float)
    pi_s = np.asarray(pi_s, dtype=float)
    phis_relative = np.asarray(phis_relative, dtype=float)
    return np.where(
        phi_levels[:, None, None] <= phis_relative[None, :, :],
        pi_s[None, :, :],
        pi_levels[:, None, None],
    )


def _reference_mass_from_profile_numpy(
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
    phi_levels: np.ndarray,
    phis: np.ndarray,
    area: np.ndarray,
    *,
    constants: MarsConstants = MARS,
) -> dict[str, np.ndarray | float]:
    phis = np.asarray(phis, dtype=float)
    area = np.asarray(area, dtype=float)
    if phis.ndim != 2 or area.shape != phis.shape:
        raise ValueError("'phis' and 'area' must be two-dimensional and share the same shape.")

    phis_relative = phis - float(np.nanmin(phis))
    pi_s, theta_s_ref, surface_layer_index = reference_surface_from_profile(
        theta_levels,
        pi_levels,
        phi_levels,
        phis,
        constants=constants,
    )
    interface_pressure = _build_reference_interface_pressure_field(
        pi_levels,
        phi_levels,
        pi_s,
        phis_relative,
    )
    layer_pressure_thickness = np.clip(interface_pressure[:-1] - interface_pressure[1:], 0.0, None)
    layer_mass_per_area = layer_pressure_thickness / constants.g
    layer_mass = np.sum(layer_mass_per_area * area[None, :, :], axis=(1, 2))

    total_area = float(np.sum(area))
    mean_pressure_on_theta = np.sum(interface_pressure * area[None, :, :], axis=(1, 2)) / total_area
    reference_surface_pressure = float(np.sum(pi_s * area) / total_area)
    deepest_surface_pressure = float(np.max(pi_s))
    return {
        "phis_relative": phis_relative,
        "pi_s": pi_s,
        "theta_s_ref": theta_s_ref,
        "surface_layer_index": surface_layer_index,
        "interface_pressure": interface_pressure,
        "mean_pressure_on_theta": mean_pressure_on_theta,
        "layer_pressure_thickness": layer_pressure_thickness,
        "layer_mass_per_area": layer_mass_per_area,
        "layer_mass": layer_mass,
        "reference_surface_pressure": reference_surface_pressure,
        "deepest_surface_pressure": deepest_surface_pressure,
    }


def _normalize_geometry_family_inputs(
    theta_levels: xr.DataArray,
    pi_levels: xr.DataArray,
    surface_potential_temperature: xr.DataArray,
    phis: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    resolved_theta_levels = normalize_isentropic_coordinate(theta_levels, name=ISENTROPIC_DIM)
    surface_theta = normalize_surface_field(surface_potential_temperature, "surface_potential_temperature")
    phis_surface = broadcast_surface_field(phis, surface_theta, "phis")

    pi_levels = require_dataarray(pi_levels, "pi_levels")
    if set(pi_levels.dims) == {ISENTROPIC_DIM}:
        pi_levels = pi_levels.transpose(ISENTROPIC_DIM).expand_dims(time=surface_theta.coords["time"])
    elif set(pi_levels.dims) != {"time", ISENTROPIC_DIM}:
        raise ValueError(
            f"'pi_levels' must have dims ('time', '{ISENTROPIC_DIM}') or ('{ISENTROPIC_DIM}',); got {pi_levels.dims!r}."
        )
    pi_levels = pi_levels.transpose("time", ISENTROPIC_DIM).astype(float)

    if not np.array_equal(surface_theta.coords["time"].values, pi_levels.coords["time"].values):
        raise ValueError("Coordinate 'time' of 'pi_levels' does not match the surface grid.")
    if not np.allclose(
        resolved_theta_levels.values,
        pi_levels.coords[ISENTROPIC_DIM].values,
    ):
        raise ValueError("'pi_levels' must use the same fixed isentropic coordinate as 'theta_levels'.")
    return resolved_theta_levels, pi_levels, surface_theta, phis_surface


def _resolve_geometry_area(
    surface_potential_temperature: xr.DataArray,
    area: xr.DataArray | None = None,
) -> xr.DataArray:
    if area is None:
        resolved = build_cell_area(
            surface_potential_temperature.coords["latitude"],
            surface_potential_temperature.coords["longitude"],
        )
    else:
        resolved = require_dataarray(area, "area")
        if set(resolved.dims) != {"latitude", "longitude"}:
            raise ValueError("'area' must have dims ('latitude', 'longitude').")
        resolved = resolved.transpose("latitude", "longitude")
    if not np.allclose(
        resolved.coords["latitude"].values,
        surface_potential_temperature.coords["latitude"].values,
    ) or not np.allclose(
        resolved.coords["longitude"].values,
        surface_potential_temperature.coords["longitude"].values,
    ):
        raise ValueError("Cell area must share the surface-grid latitude/longitude coordinates.")
    return resolved.astype(float)


def _build_koehler_geometry_family_xr(
    family_name: str,
    theta_levels: xr.DataArray,
    pi_levels: xr.DataArray,
    surface_potential_temperature: xr.DataArray,
    phis: xr.DataArray,
    *,
    area: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> _KoehlerGeometryFamily:
    theta_levels, pi_levels, surface_theta, phis_surface = _normalize_geometry_family_inputs(
        theta_levels,
        pi_levels,
        surface_potential_temperature,
        phis,
    )
    cell_area = _resolve_geometry_area(surface_theta, area)
    time_values = pi_levels.coords["time"].values
    theta_values = np.asarray(theta_levels.values, dtype=float)
    layer_values = 0.5 * (theta_values[:-1] + theta_values[1:])

    phi_levels_values = np.empty(pi_levels.shape, dtype=float)
    anchor_values = np.empty(time_values.size, dtype=float)
    anchor_layer_index_values = np.empty(time_values.size, dtype=np.int64)
    anchor_fraction_values = np.empty(time_values.size, dtype=float)
    anchor_mode_values = np.empty(time_values.size, dtype=object)
    pi_s_values = np.empty(surface_theta.shape, dtype=float)
    theta_s_ref_values = np.empty(surface_theta.shape, dtype=float)
    surface_layer_index_values = np.empty(surface_theta.shape, dtype=np.int64)
    phis_relative_values = np.empty(surface_theta.shape, dtype=float)
    reference_surface_pressure_values = np.empty(time_values.size, dtype=float)
    deepest_surface_pressure_values = np.empty(time_values.size, dtype=float)
    interface_pressure_values = np.empty((time_values.size, theta_values.size, surface_theta.sizes["latitude"], surface_theta.sizes["longitude"]), dtype=float)
    mean_pressure_on_theta_values = np.empty((time_values.size, theta_values.size), dtype=float)
    layer_pressure_thickness_values = np.empty((time_values.size, layer_values.size, surface_theta.sizes["latitude"], surface_theta.sizes["longitude"]), dtype=float)
    layer_mass_per_area_values = np.empty_like(layer_pressure_thickness_values)
    layer_mass_values = np.empty((time_values.size, layer_values.size), dtype=float)

    for time_index in range(time_values.size):
        pi_levels_t = np.asarray(pi_levels.isel(time=time_index).values, dtype=float)
        theta_anchor_t = float(np.nanmin(surface_theta.isel(time=time_index).values))
        anchor_values[time_index] = theta_anchor_t
        (
            phi_levels_t,
            anchor_layer_index_t,
            anchor_fraction_t,
            anchor_mode_t,
        ) = _reference_geopotential_with_anchor_details(
            theta_values,
            pi_levels_t,
            theta_anchor=theta_anchor_t,
            phi_anchor=0.0,
            constants=constants,
        )
        reference_t = _reference_mass_from_profile_numpy(
            theta_values,
            pi_levels_t,
            phi_levels_t,
            np.asarray(phis_surface.isel(time=time_index).values, dtype=float),
            np.asarray(cell_area.values, dtype=float),
            constants=constants,
        )

        phi_levels_values[time_index] = phi_levels_t
        anchor_layer_index_values[time_index] = anchor_layer_index_t
        anchor_fraction_values[time_index] = anchor_fraction_t
        anchor_mode_values[time_index] = anchor_mode_t
        pi_s_values[time_index] = reference_t["pi_s"]
        theta_s_ref_values[time_index] = reference_t["theta_s_ref"]
        surface_layer_index_values[time_index] = reference_t["surface_layer_index"]
        phis_relative_values[time_index] = reference_t["phis_relative"]
        reference_surface_pressure_values[time_index] = reference_t["reference_surface_pressure"]
        deepest_surface_pressure_values[time_index] = reference_t["deepest_surface_pressure"]
        interface_pressure_values[time_index] = reference_t["interface_pressure"]
        mean_pressure_on_theta_values[time_index] = reference_t["mean_pressure_on_theta"]
        layer_pressure_thickness_values[time_index] = reference_t["layer_pressure_thickness"]
        layer_mass_per_area_values[time_index] = reference_t["layer_mass_per_area"]
        layer_mass_values[time_index] = reference_t["layer_mass"]

    profile = _KoehlerProfileGeometry(
        theta_levels=theta_levels,
        pi_levels=xr.DataArray(
            pi_levels.values,
            dims=("time", ISENTROPIC_DIM),
            coords={"time": time_values, ISENTROPIC_DIM: theta_values},
            name=f"{family_name}_pi_levels",
            attrs={
                "units": "Pa",
                "reference_pressure_sampling": "fixed_isentropic_level_pressure",
                "reference_curve_interpolation_space": "exner",
            },
        ),
        phi_levels=xr.DataArray(
            phi_levels_values,
            dims=("time", ISENTROPIC_DIM),
            coords={"time": time_values, ISENTROPIC_DIM: theta_values},
            name=f"{family_name}_phi_levels",
            attrs={
                "units": "m2 s-2",
                "reference_geopotential_datum": "relative_to_minimum_surface_geopotential_with_zero_at_theta_anchor",
            },
        ),
        theta_anchor=xr.DataArray(
            anchor_values,
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_theta_anchor",
            attrs={"units": "K"},
        ),
        anchor_layer_index=xr.DataArray(
            anchor_layer_index_values,
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_anchor_layer_index",
        ),
        anchor_fraction=xr.DataArray(
            anchor_fraction_values,
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_anchor_fraction",
        ),
        anchor_mode=xr.DataArray(
            anchor_mode_values,
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_anchor_mode",
        ),
        profile_bottom_pressure=xr.DataArray(
            np.asarray(pi_levels.values[:, 0], dtype=float),
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_profile_bottom_pressure",
            attrs={"units": "Pa"},
        ),
        profile_top_pressure=xr.DataArray(
            np.asarray(pi_levels.values[:, -1], dtype=float),
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_profile_top_pressure",
            attrs={"units": "Pa"},
        ),
    )
    surface = _KoehlerSurfaceReconstruction(
        pi_s=xr.DataArray(
            pi_s_values,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name="pi_s" if family_name == "full" else "pi_sZ",
            attrs={
                "units": "Pa",
                "long_name": (
                    "reference-state surface pressure"
                    if family_name == "full"
                    else "zonal-thermodynamic reference-state surface pressure on actual topography"
                ),
            },
        ),
        theta_s_ref=xr.DataArray(
            theta_s_ref_values,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name=f"{family_name}_theta_s_ref",
            attrs={"units": "K"},
        ),
        surface_layer_index=xr.DataArray(
            surface_layer_index_values,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name=f"{family_name}_surface_layer_index",
        ),
        surface_valid_mask=xr.DataArray(
            np.isfinite(pi_s_values) & np.isfinite(theta_s_ref_values),
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name=f"{family_name}_surface_valid_mask",
        ),
        phis_relative=xr.DataArray(
            phis_relative_values,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name=f"{family_name}_phis_relative",
            attrs={"units": phis_surface.attrs.get("units", "m2 s-2")},
        ),
        reference_surface_pressure=xr.DataArray(
            reference_surface_pressure_values,
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_reference_surface_pressure",
            attrs={"units": "Pa"},
        ),
        deepest_surface_pressure=xr.DataArray(
            deepest_surface_pressure_values,
            dims=("time",),
            coords={"time": time_values},
            name=f"{family_name}_deepest_surface_pressure",
            attrs={"units": "Pa"},
        ),
    )
    reference_mass = _KoehlerReferenceMass(
        interface_pressure=xr.DataArray(
            interface_pressure_values,
            dims=("time", ISENTROPIC_DIM, "latitude", "longitude"),
            coords={
                "time": time_values,
                ISENTROPIC_DIM: theta_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name=f"{family_name}_reference_interface_pressure_on_topography",
            attrs={"units": "Pa"},
        ),
        mean_pressure_on_theta=xr.DataArray(
            mean_pressure_on_theta_values,
            dims=("time", ISENTROPIC_DIM),
            coords={"time": time_values, ISENTROPIC_DIM: theta_values},
            name=f"{family_name}_mean_pressure_on_theta",
            attrs={"units": "Pa"},
        ),
        layer_pressure_thickness=xr.DataArray(
            layer_pressure_thickness_values,
            dims=("time", ISENTROPIC_LAYER_DIM, "latitude", "longitude"),
            coords={
                "time": time_values,
                ISENTROPIC_LAYER_DIM: layer_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
                "lower_theta": xr.DataArray(
                    theta_values[:-1],
                    dims=(ISENTROPIC_LAYER_DIM,),
                    coords={ISENTROPIC_LAYER_DIM: layer_values},
                    attrs={"units": theta_levels.attrs.get("units", "K")},
                ),
                "upper_theta": xr.DataArray(
                    theta_values[1:],
                    dims=(ISENTROPIC_LAYER_DIM,),
                    coords={ISENTROPIC_LAYER_DIM: layer_values},
                    attrs={"units": theta_levels.attrs.get("units", "K")},
                ),
            },
            name=f"{family_name}_reference_layer_pressure_thickness",
            attrs={"units": "Pa"},
        ),
        layer_mass_per_area=xr.DataArray(
            layer_mass_per_area_values,
            dims=("time", ISENTROPIC_LAYER_DIM, "latitude", "longitude"),
            coords={
                "time": time_values,
                ISENTROPIC_LAYER_DIM: layer_values,
                "latitude": surface_theta.coords["latitude"].values,
                "longitude": surface_theta.coords["longitude"].values,
            },
            name=f"{family_name}_reference_layer_mass_per_area",
            attrs={"units": "kg m-2"},
        ),
        layer_mass=xr.DataArray(
            layer_mass_values,
            dims=("time", ISENTROPIC_LAYER_DIM),
            coords={"time": time_values, ISENTROPIC_LAYER_DIM: layer_values},
            name=f"{family_name}_reference_layer_mass",
            attrs={"units": "kg"},
        ),
    )
    return _KoehlerGeometryFamily(
        family_name=family_name,
        profile=profile,
        surface=surface,
        reference_mass=reference_mass,
    )


__all__: list[str] = []
