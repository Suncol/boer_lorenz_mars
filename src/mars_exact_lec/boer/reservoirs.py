"""Exact Boer reservoirs: kinetic-energy stores plus phase-2 APE body terms."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr

from .._validation import (
    FIELD_DIMS,
    ZONAL_DIMS,
    ensure_matching_coordinates,
    normalize_field,
    normalize_zonal_field,
    require_dataarray,
)
from ..common.integrals import MassIntegrator
from ..common.zonal_ops import representative_eddy, representative_zonal_mean, theta_coverage, zonal_mean
from ..constants_mars import MARS, MarsConstants
from ..reference_state.interpolate_isentropes import potential_temperature


_FULL_PRESSURE_KEYS = ("pi", "reference_pressure", "reference_pressure_full")
_ZONAL_PRESSURE_KEYS = ("pi_z", "pi_Z", "reference_pressure_zonal", "representative_reference_pressure")
_FULL_EFFICIENCY_KEYS = ("n", "N", "efficiency_factor", "ape_efficiency")
_ZONAL_EFFICIENCY_KEYS = (
    "n_z",
    "N_Z",
    "efficiency_factor_zonal",
    "representative_efficiency_factor",
)


def _lookup_reference_component(reference_state: Any, candidates: tuple[str, ...]) -> Any | None:
    if reference_state is None:
        return None

    if isinstance(reference_state, Mapping):
        for candidate in candidates:
            if candidate in reference_state and reference_state[candidate] is not None:
                return reference_state[candidate]

    for candidate in candidates:
        if hasattr(reference_state, candidate):
            value = getattr(reference_state, candidate)
            if value is not None:
                return value

    return None


def _ensure_matching_zonal_coordinates(
    reference: xr.DataArray,
    other: xr.DataArray,
    name: str,
) -> xr.DataArray:
    reference = normalize_zonal_field(reference, "reference")
    other = normalize_zonal_field(other, name)

    for coord_name in ZONAL_DIMS:
        if coord_name == "time":
            equal = np.array_equal(reference[coord_name].values, other[coord_name].values)
        else:
            equal = np.allclose(reference[coord_name].values, other[coord_name].values)
        if not equal:
            raise ValueError(f"Coordinate {coord_name!r} of {name!r} does not match the template.")

    return other


def _normalize_reference_field(
    value: Any,
    template: xr.DataArray,
    name: str,
    *,
    zonal: bool,
) -> xr.DataArray:
    if np.isscalar(value):
        field = xr.full_like(template, float(value))
        field.name = name
        return field

    field = require_dataarray(value, name)
    dims = set(field.dims)
    if zonal:
        if dims == set(ZONAL_DIMS):
            return _ensure_matching_zonal_coordinates(template, field, name)
        if dims == set(FIELD_DIMS):
            return _ensure_matching_zonal_coordinates(
                template,
                zonal_mean(normalize_field(field, name)),
                name,
            )
        raise ValueError(
            f"{name!r} must be scalar or contain exactly the dims {ZONAL_DIMS} or {FIELD_DIMS}; "
            f"got {field.dims!r}."
        )

    if dims != set(FIELD_DIMS):
        raise ValueError(
            f"{name!r} must be scalar or contain exactly the dims {FIELD_DIMS}; got {field.dims!r}."
        )

    field = normalize_field(field, name)
    ensure_matching_coordinates(template, [field])
    return field


def _pressure_like(template: xr.DataArray) -> xr.DataArray:
    template = require_dataarray(template, "template")
    pressure = xr.DataArray(
        np.asarray(template.coords["level"].values, dtype=float),
        dims=("level",),
        coords={"level": template.coords["level"].values},
        name="pressure",
        attrs={"units": "Pa"},
    )
    return pressure.broadcast_like(template)


def _resolve_efficiency_factor(
    reference_state: Any,
    explicit_efficiency: Any,
    explicit_pressure: Any,
    template: xr.DataArray,
    *,
    zonal: bool,
    constants: MarsConstants,
    output_name: str,
    pressure_field: xr.DataArray | None = None,
    potential_temperature_field: xr.DataArray | None = None,
) -> xr.DataArray:
    efficiency_keys = _ZONAL_EFFICIENCY_KEYS if zonal else _FULL_EFFICIENCY_KEYS
    pressure_keys = _ZONAL_PRESSURE_KEYS if zonal else _FULL_PRESSURE_KEYS

    efficiency_value = explicit_efficiency
    if efficiency_value is None:
        if zonal and (
            hasattr(reference_state, "zonal_efficiency")
            and potential_temperature_field is not None
            and pressure_field is not None
        ):
            efficiency_value = reference_state.zonal_efficiency(
                potential_temperature_field,
                pressure_field,
            )
        elif (
            not zonal
            and hasattr(reference_state, "efficiency")
            and potential_temperature_field is not None
            and pressure_field is not None
        ):
            efficiency_value = reference_state.efficiency(
                potential_temperature_field,
                pressure_field,
            )
        if efficiency_value is None and zonal and (
            hasattr(reference_state, "zonal_reference_pressure")
            and potential_temperature_field is not None
        ):
            explicit_pressure = reference_state.zonal_reference_pressure(
                potential_temperature_field
            )
        elif efficiency_value is None and (
            hasattr(reference_state, "reference_pressure")
            and potential_temperature_field is not None
        ):
            explicit_pressure = reference_state.reference_pressure(
                potential_temperature_field,
                name="pi_Z" if zonal else "pi",
            )
    if efficiency_value is None:
        efficiency_value = _lookup_reference_component(reference_state, efficiency_keys)

    if efficiency_value is not None:
        efficiency = _normalize_reference_field(
            efficiency_value,
            template,
            output_name,
            zonal=zonal,
        )
        efficiency.name = output_name
        efficiency.attrs.setdefault("units", "1")
        return efficiency

    pressure_value = explicit_pressure
    if pressure_value is None:
        pressure_value = _lookup_reference_component(reference_state, pressure_keys)

    if pressure_value is None:
        scope = "zonal" if zonal else "full"
        raise ValueError(
            f"A reference-state {scope} efficiency factor requires either "
            f"{efficiency_keys[0]!r} or {pressure_keys[0]!r}."
        )

    reference_pressure = _normalize_reference_field(
        pressure_value,
        template,
        pressure_keys[0],
        zonal=zonal,
    )
    if np.any(np.asarray(reference_pressure.values, dtype=float) <= 0.0):
        raise ValueError("Reference-state pressures must be strictly positive.")

    denominator = pressure_field
    if denominator is None:
        denominator = _pressure_like(template)
    elif zonal:
        denominator = _ensure_matching_zonal_coordinates(template, denominator, "pressure")
    else:
        denominator = normalize_field(denominator, "pressure")
        ensure_matching_coordinates(template, [denominator])

    efficiency = 1.0 - (reference_pressure / denominator) ** constants.kappa
    efficiency.name = output_name
    efficiency.attrs["units"] = "1"
    return efficiency


def total_horizontal_ke(
    u: xr.DataArray,
    v: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return ``∫_M 0.5 Theta (u² + v²) dm`` in Joules."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, theta])

    integrand = 0.5 * theta * (u**2 + v**2)
    result = integrator.integrate_full(integrand)
    result.name = "total_horizontal_ke"
    result.attrs["units"] = "J"
    return result


def kinetic_energy_zonal(
    u: xr.DataArray,
    v: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return the zonal kinetic-energy reservoir ``K_Z`` in Joules."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, theta])

    coverage = theta_coverage(theta)
    u_r = representative_zonal_mean(u, theta)
    v_r = representative_zonal_mean(v, theta)
    integrand = 0.5 * coverage * (u_r**2 + v_r**2)
    result = integrator.integrate_zonal(integrand)
    result.name = "K_Z"
    result.attrs["units"] = "J"
    return result


def kinetic_energy_eddy(
    u: xr.DataArray,
    v: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return the eddy kinetic-energy reservoir ``K_E`` in Joules."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, theta])

    u_star = representative_eddy(u, theta)
    v_star = representative_eddy(v, theta)
    integrand = 0.5 * zonal_mean(theta * (u_star**2 + v_star**2))
    result = integrator.integrate_zonal(integrand)
    result.name = "K_E"
    result.attrs["units"] = "J"
    return result


def available_potential_energy_zonal_part1(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    reference_state: Any | None = None,
    *,
    potential_temperature_field: xr.DataArray | None = None,
    pi_z: xr.DataArray | float | None = None,
    n_z: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_Z1 = ∫_M c_p Θ N_Z [T]_R dm`` in Joules.

    Here ``theta`` denotes the phase-1 volume mask ``Θ``, not potential temperature.
    This is the phase-2 body term only and omits the explicit surface contribution
    ``A_Z2``.
    """

    temperature = normalize_field(temperature, "temperature")
    pressure = normalize_field(pressure, "pressure")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(temperature, [pressure, theta])

    constants = integrator.constants
    if potential_temperature_field is None:
        potential_temperature_field = potential_temperature(
            temperature,
            pressure,
            constants=constants,
        )
    else:
        potential_temperature_field = normalize_field(
            potential_temperature_field,
            "potential_temperature_field",
        )
        ensure_matching_coordinates(temperature, [potential_temperature_field])

    coverage = theta_coverage(theta)
    temperature_r = representative_zonal_mean(temperature, theta)
    pressure_r = representative_zonal_mean(pressure, theta)
    representative_theta = representative_zonal_mean(potential_temperature_field, theta)
    n_z_field = _resolve_efficiency_factor(
        reference_state,
        n_z,
        pi_z,
        coverage,
        zonal=True,
        constants=constants,
        output_name="N_Z",
        pressure_field=pressure_r,
        potential_temperature_field=representative_theta,
    )

    integrand = constants.cp * coverage * n_z_field * temperature_r
    result = integrator.integrate_zonal(integrand)
    result.name = "A_Z1"
    result.attrs["units"] = "J"
    result.attrs["long_name"] = "zonal exact available potential energy body term"
    result.attrs["surface_term_included"] = False
    return result


def available_potential_energy_eddy_part1(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    reference_state: Any | None = None,
    *,
    potential_temperature_field: xr.DataArray | None = None,
    pi: xr.DataArray | float | None = None,
    pi_z: xr.DataArray | float | None = None,
    n: xr.DataArray | float | None = None,
    n_z: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_E1 = ∫_M c_p Θ (N - N_Z) T dm`` in Joules.

    This is the phase-2 body term only and omits the explicit surface
    contribution ``A_E2``.
    """

    temperature = normalize_field(temperature, "temperature")
    pressure = normalize_field(pressure, "pressure")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(temperature, [pressure, theta])

    constants = integrator.constants
    if potential_temperature_field is None:
        potential_temperature_field = potential_temperature(
            temperature,
            pressure,
            constants=constants,
        )
    else:
        potential_temperature_field = normalize_field(
            potential_temperature_field,
            "potential_temperature_field",
        )
        ensure_matching_coordinates(temperature, [potential_temperature_field])

    n_field = _resolve_efficiency_factor(
        reference_state,
        n,
        pi,
        temperature,
        zonal=False,
        constants=constants,
        output_name="N",
        pressure_field=pressure,
        potential_temperature_field=potential_temperature_field,
    )
    n_z_field = _resolve_efficiency_factor(
        reference_state,
        n_z,
        pi_z,
        theta_coverage(theta),
        zonal=True,
        constants=constants,
        output_name="N_Z",
        pressure_field=representative_zonal_mean(pressure, theta),
        potential_temperature_field=representative_zonal_mean(
            potential_temperature_field,
            theta,
        ),
    )

    integrand = constants.cp * zonal_mean(
        theta * (n_field - n_z_field.broadcast_like(temperature)) * temperature
    )
    result = integrator.integrate_zonal(integrand)
    result.name = "A_E1"
    result.attrs["units"] = "J"
    result.attrs["long_name"] = "eddy exact available potential energy body term"
    result.attrs["surface_term_included"] = False
    return result


def available_potential_energy_part1(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    reference_state: Any | None = None,
    *,
    potential_temperature_field: xr.DataArray | None = None,
    pi: xr.DataArray | float | None = None,
    n: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_1 = ∫_M c_p Θ N T dm`` in Joules.

    This is the phase-2 body contribution to ``A = A_Z + A_E`` before any
    explicit topographic surface term is added.
    """

    temperature = normalize_field(temperature, "temperature")
    pressure = normalize_field(pressure, "pressure")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(temperature, [pressure, theta])

    constants = integrator.constants
    if potential_temperature_field is None:
        potential_temperature_field = potential_temperature(
            temperature,
            pressure,
            constants=constants,
        )
    else:
        potential_temperature_field = normalize_field(
            potential_temperature_field,
            "potential_temperature_field",
        )
        ensure_matching_coordinates(temperature, [potential_temperature_field])

    n_field = _resolve_efficiency_factor(
        reference_state,
        n,
        pi,
        temperature,
        zonal=False,
        constants=constants,
        output_name="N",
        pressure_field=pressure,
        potential_temperature_field=potential_temperature_field,
    )

    integrand = constants.cp * theta * n_field * temperature
    result = integrator.integrate_full(integrand)
    result.name = "A_1"
    result.attrs["units"] = "J"
    result.attrs["long_name"] = "exact available potential energy body term"
    result.attrs["surface_term_included"] = False
    return result


total_available_potential_energy_part1 = available_potential_energy_part1
A_Z1 = available_potential_energy_zonal_part1
A_E1 = available_potential_energy_eddy_part1
A1 = available_potential_energy_part1

# Phase-2 compatibility aliases. These currently expose only the body terms and
# intentionally do not claim to include the explicit topographic surface terms.
A_Z = available_potential_energy_zonal_part1
A_E = available_potential_energy_eddy_part1
A = available_potential_energy_part1


__all__ = [
    "total_horizontal_ke",
    "kinetic_energy_zonal",
    "kinetic_energy_eddy",
    "available_potential_energy_zonal_part1",
    "available_potential_energy_eddy_part1",
    "available_potential_energy_part1",
    "total_available_potential_energy_part1",
    "A_Z1",
    "A_E1",
    "A1",
    "A_Z",
    "A_E",
    "A",
]
