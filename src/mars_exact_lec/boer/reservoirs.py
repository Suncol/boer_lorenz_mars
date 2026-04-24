"""Exact Boer reservoirs: kinetic energy plus stage-3 total APE terms."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr

from .._validation import (
    FIELD_DIMS,
    SURFACE_DIMS,
    SURFACE_ZONAL_DIMS,
    ZONAL_DIMS,
    ensure_matching_coordinates,
    ensure_matching_surface_coordinates,
    normalize_field,
    normalize_theta_mask,
    normalize_surface_field,
    normalize_surface_zonal_field,
    normalize_zonal_field,
    require_dataarray,
    resolve_deprecated_theta_mask,
)
from ..common.grid_weights import longitude_weights
from ..common.integrals import MassIntegrator
from ..common.topography_measure import TopographyAwareMeasure, resolve_exact_measure
from ..common.zonal_ops import (
    representative_eddy,
    representative_zonal_mean,
    theta_coverage,
    weighted_coverage,
    weighted_representative_eddy,
    weighted_representative_zonal_mean,
    zonal_mean,
)
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
_SURFACE_PRESSURE_KEYS = ("pi_s", "surface_reference_pressure", "reference_surface_pressure_full")
_SURFACE_ZONAL_PRESSURE_KEYS = (
    "pi_sZ",
    "surface_reference_pressure_zonal",
    "reference_surface_pressure_zonal",
)
_REFERENCE_STATUS_MISSING = object()


def _require_integrator(integrator: MassIntegrator | None) -> MassIntegrator:
    if integrator is None:
        raise TypeError("'integrator' is required.")
    return integrator


def _weight_field(theta_mask: xr.DataArray, measure: TopographyAwareMeasure | None) -> xr.DataArray:
    theta_mask = normalize_theta_mask(theta_mask)
    if measure is None:
        return theta_mask
    ensure_matching_coordinates(theta_mask, [measure.cell_fraction])
    return measure.cell_fraction


def _coverage_field(theta_mask: xr.DataArray, measure: TopographyAwareMeasure | None) -> xr.DataArray:
    weight = _weight_field(theta_mask, measure)
    return weighted_coverage(weight) if measure is not None else theta_coverage(weight)


def _representative_mean(
    field: xr.DataArray,
    theta_mask: xr.DataArray,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    weight = _weight_field(theta_mask, measure)
    if measure is None:
        return representative_zonal_mean(field, weight)
    return weighted_representative_zonal_mean(field, weight)


def _representative_eddy(
    field: xr.DataArray,
    theta_mask: xr.DataArray,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    weight = _weight_field(theta_mask, measure)
    if measure is None:
        return representative_eddy(field, weight)
    return weighted_representative_eddy(field, weight)


def _safe_mass_ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    denominator = denominator.broadcast_like(numerator)
    return xr.where(denominator > 0.0, numerator / denominator, 0.0)


def _integrate_full_mass_aware(
    integrand: xr.DataArray,
    weight: xr.DataArray,
    integrator: MassIntegrator,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    if measure is None:
        return integrator.integrate_full(integrand)
    return measure.integrate_full(_safe_mass_ratio(integrand, weight))


def _integrate_zonal_mass_aware(
    integrand: xr.DataArray,
    coverage: xr.DataArray,
    integrator: MassIntegrator,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    if measure is None:
        return integrator.integrate_zonal(integrand)
    return measure.integrate_zonal(_safe_mass_ratio(integrand, coverage))


def _effective_surface_pressure(ps: xr.DataArray, measure: TopographyAwareMeasure | None) -> xr.DataArray:
    ps = normalize_surface_field(ps, "ps")
    if measure is None:
        return ps
    ensure_matching_surface_coordinates(ps, [measure.effective_surface_pressure])
    return measure.effective_surface_pressure


def _annotate_quantity(
    result: xr.DataArray,
    *,
    units: str,
    base_quantity: str,
    measure: TopographyAwareMeasure | None = None,
    surface_term_included: bool | None = None,
    long_name: str | None = None,
) -> xr.DataArray:
    result.attrs["units"] = units
    result.attrs["normalization"] = "global_integral"
    result.attrs["base_quantity"] = base_quantity
    if measure is not None:
        result.attrs.update(measure.domain_metadata)
    if surface_term_included is not None:
        result.attrs["surface_term_included"] = surface_term_included
    if long_name is not None:
        result.attrs["long_name"] = long_name
    return result


def _resolved_measure(
    integrator: MassIntegrator,
    *,
    theta_mask: xr.DataArray | None = None,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    diagnostic_name: str,
) -> TopographyAwareMeasure:
    return resolve_exact_measure(
        integrator,
        measure=measure,
        ps=ps,
        theta_mask=theta_mask,
        theta=theta,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name=diagnostic_name,
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


def _lookup_reference_convergence(reference_state: Any, *, zonal: bool) -> Any:
    if reference_state is None:
        return _REFERENCE_STATUS_MISSING

    field_name = "converged_zonal" if zonal else "converged"
    if isinstance(reference_state, Mapping):
        return reference_state.get(field_name, _REFERENCE_STATUS_MISSING)
    if hasattr(reference_state, field_name):
        return getattr(reference_state, field_name)
    return _REFERENCE_STATUS_MISSING


def _reference_convergence_truth_values(status: Any) -> np.ndarray:
    if isinstance(status, xr.DataArray):
        values = np.asarray(status.values)
    else:
        values = np.asarray(status)
    with np.errstate(invalid="ignore"):
        truth = np.equal(values, True)
    return np.asarray(truth, dtype=bool)


def _ensure_reference_state_converged(
    reference_state: Any,
    *,
    zonal: bool,
    component: str,
) -> None:
    if reference_state is None:
        return

    status = _lookup_reference_convergence(reference_state, zonal=zonal)
    if status is _REFERENCE_STATUS_MISSING:
        field_name = "converged_zonal" if zonal else "converged"
        scope = "zonal" if zonal else "full"
        raise ValueError(
            f"Reference state {scope} solve cannot be verified for {component!r} because "
            f"{field_name!r} is missing."
        )

    field_name = "converged_zonal" if zonal else "converged"
    scope = "zonal" if zonal else "full"
    if status is None:
        raise ValueError(
            f"Reference state {scope} solve cannot be verified for {component!r} because "
            f"{field_name!r} is None."
        )

    truth = _reference_convergence_truth_values(status)
    if truth.size == 0:
        raise ValueError(
            f"Reference state {scope} solve cannot be verified for {component!r} because "
            f"{field_name!r} is empty."
        )
    if not bool(np.all(truth)):
        failed = int(truth.size - np.count_nonzero(truth))
        raise ValueError(
            f"Reference state {scope} solve did not converge for {failed} value(s); "
            f"refusing to use it for {component!r}. Check {field_name!r} or pass explicit "
            "reference diagnostics for this component."
        )


def _surface_zonal_mean(field: xr.DataArray) -> xr.DataArray:
    field = normalize_surface_field(field, "field")
    weights = longitude_weights(field.coords["longitude"], normalize=True)
    return field.weighted(weights).sum(dim="longitude")


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


def _ensure_matching_surface_zonal_coordinates(
    reference: xr.DataArray,
    other: xr.DataArray,
    name: str,
) -> xr.DataArray:
    reference = normalize_surface_zonal_field(reference, "reference")
    other = normalize_surface_zonal_field(other, name)

    for coord_name in SURFACE_ZONAL_DIMS:
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
        raise ValueError(
            f"{name!r} must be scalar or contain exactly the dims {ZONAL_DIMS}; "
            f"got {field.dims!r}."
        )

    if dims != set(FIELD_DIMS):
        raise ValueError(
            f"{name!r} must be scalar or contain exactly the dims {FIELD_DIMS}; got {field.dims!r}."
        )

    field = normalize_field(field, name)
    ensure_matching_coordinates(template, [field])
    return field


def _normalize_surface_component(
    value: Any,
    template: xr.DataArray,
    name: str,
    *,
    zonal: bool,
    allow_scalar: bool = True,
) -> xr.DataArray:
    if np.isscalar(value):
        if not allow_scalar:
            raise ValueError(f"{name!r} must be provided as a canonical surface field, not a scalar.")
        field = xr.full_like(template, float(value))
        field.name = name
        return field

    field = require_dataarray(value, name)
    dims = set(field.dims)
    if zonal:
        if dims == set(SURFACE_ZONAL_DIMS):
            return _ensure_matching_surface_zonal_coordinates(template, field, name)
        raise ValueError(
            f"{name!r} must be scalar or contain exactly the dims {SURFACE_ZONAL_DIMS}; "
            f"got {field.dims!r}."
        )

    if dims != set(SURFACE_DIMS):
        raise ValueError(
            f"{name!r} must be scalar or contain exactly the dims {SURFACE_DIMS}; got {field.dims!r}."
        )

    field = normalize_surface_field(field, name)
    ensure_matching_surface_coordinates(template, [field])
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


def _require_pressure_coordinate_field(pressure: xr.DataArray, template: xr.DataArray, name: str = "pressure") -> xr.DataArray:
    pressure = normalize_field(pressure, name)
    ensure_matching_coordinates(template, [pressure])
    canonical = _pressure_like(template)
    if not np.allclose(
        np.asarray(pressure.values, dtype=float),
        np.asarray(canonical.values, dtype=float),
        equal_nan=True,
    ):
        raise ValueError(
            f"{name!r} must equal the pressure-coordinate level broadcast on the canonical grid."
        )
    canonical.name = pressure.name
    canonical.attrs.update(pressure.attrs)
    return canonical


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
    if efficiency_value is None and explicit_pressure is None and reference_state is not None:
        _ensure_reference_state_converged(
            reference_state,
            zonal=zonal,
            component=output_name,
        )
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


def _resolve_surface_reference_pressure(
    reference_state: Any,
    explicit_pressure: Any,
    template: xr.DataArray,
    *,
    output_name: str,
    pressure_keys: tuple[str, ...],
    zonal: bool,
) -> xr.DataArray:
    pressure_value = explicit_pressure
    if pressure_value is None:
        _ensure_reference_state_converged(
            reference_state,
            zonal=zonal,
            component=output_name,
        )
        pressure_value = _lookup_reference_component(reference_state, pressure_keys)
    if pressure_value is None:
        raise ValueError(
            f"Total exact surface terms require reference-state surface diagnostics {pressure_keys[0]!r}."
        )

    result = _normalize_surface_component(
        pressure_value,
        template,
        output_name,
        zonal=False,
        allow_scalar=False,
    )
    values = np.asarray(result.values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{output_name!r} must remain finite and strictly positive on the canonical surface grid.")
    result.name = output_name
    result.attrs.setdefault("units", "Pa")
    return result


def total_horizontal_ke(
    u: xr.DataArray,
    v: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return ``∫_M 0.5 Theta (u² + v²) dm`` in Joules."""

    integrator = _require_integrator(integrator)
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(u, [v, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="total_horizontal_ke",
    )

    weight = _weight_field(theta_mask, measure)
    integrand = 0.5 * weight * (u**2 + v**2)
    result = _integrate_full_mass_aware(integrand, weight, integrator, measure)
    result.name = "total_horizontal_ke"
    return _annotate_quantity(result, units="J", base_quantity="energy", measure=measure)


def kinetic_energy_zonal(
    u: xr.DataArray,
    v: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return the zonal kinetic-energy reservoir ``K_Z`` in Joules."""

    integrator = _require_integrator(integrator)
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(u, [v, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="kinetic_energy_zonal",
    )

    coverage = _coverage_field(theta_mask, measure)
    u_r = _representative_mean(u, theta_mask, measure)
    v_r = _representative_mean(v, theta_mask, measure)
    integrand = 0.5 * coverage * (u_r**2 + v_r**2)
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "K_Z"
    return _annotate_quantity(result, units="J", base_quantity="energy", measure=measure)


def kinetic_energy_eddy(
    u: xr.DataArray,
    v: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return the eddy kinetic-energy reservoir ``K_E`` in Joules."""

    integrator = _require_integrator(integrator)
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(u, [v, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="kinetic_energy_eddy",
    )

    weight = _weight_field(theta_mask, measure)
    coverage = _coverage_field(theta_mask, measure)
    u_star = _representative_eddy(u, theta_mask, measure)
    v_star = _representative_eddy(v, theta_mask, measure)
    integrand = 0.5 * zonal_mean(weight * (u_star**2 + v_star**2))
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "K_E"
    return _annotate_quantity(result, units="J", base_quantity="energy", measure=measure)


def available_potential_energy_zonal_part1(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    reference_state: Any | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    potential_temperature_field: xr.DataArray | None = None,
    pi_z: xr.DataArray | float | None = None,
    n_z: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_Z1 = ∫_M c_p Θ N_Z [T]_R dm`` in Joules."""

    integrator = _require_integrator(integrator)
    temperature = normalize_field(temperature, "temperature")
    pressure = _require_pressure_coordinate_field(pressure, temperature)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(temperature, [pressure, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_Z1",
    )

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

    coverage = _coverage_field(theta_mask, measure)
    temperature_r = _representative_mean(temperature, theta_mask, measure)
    pressure_r = _representative_mean(pressure, theta_mask, measure)
    representative_theta = _representative_mean(potential_temperature_field, theta_mask, measure)
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
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "A_Z1"
    return _annotate_quantity(
        result,
        units="J",
        base_quantity="energy",
        measure=measure,
        long_name="zonal exact available potential energy body term",
        surface_term_included=False,
    )


def available_potential_energy_eddy_part1(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    reference_state: Any | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    potential_temperature_field: xr.DataArray | None = None,
    pi: xr.DataArray | float | None = None,
    pi_z: xr.DataArray | float | None = None,
    n: xr.DataArray | float | None = None,
    n_z: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_E1 = ∫_M c_p Θ (N - N_Z) T dm`` in Joules."""

    integrator = _require_integrator(integrator)
    temperature = normalize_field(temperature, "temperature")
    pressure = _require_pressure_coordinate_field(pressure, temperature)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(temperature, [pressure, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_E1",
    )

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
    coverage = _coverage_field(theta_mask, measure)
    weight = _weight_field(theta_mask, measure)
    n_z_field = _resolve_efficiency_factor(
        reference_state,
        n_z,
        pi_z,
        coverage,
        zonal=True,
        constants=constants,
        output_name="N_Z",
        pressure_field=_representative_mean(pressure, theta_mask, measure),
        potential_temperature_field=_representative_mean(
            potential_temperature_field,
            theta_mask,
            measure,
        ),
    )

    integrand = constants.cp * zonal_mean(
        weight * (n_field - n_z_field.broadcast_like(temperature)) * temperature
    )
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "A_E1"
    return _annotate_quantity(
        result,
        units="J",
        base_quantity="energy",
        measure=measure,
        long_name="eddy exact available potential energy body term",
        surface_term_included=False,
    )


def available_potential_energy_part1(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    reference_state: Any | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    potential_temperature_field: xr.DataArray | None = None,
    pi: xr.DataArray | float | None = None,
    n: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_1 = ∫_M c_p Θ N T dm`` in Joules."""

    integrator = _require_integrator(integrator)
    temperature = normalize_field(temperature, "temperature")
    pressure = _require_pressure_coordinate_field(pressure, temperature)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(temperature, [pressure, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_1",
    )

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

    weight = _weight_field(theta_mask, measure)
    integrand = constants.cp * weight * n_field * temperature
    result = _integrate_full_mass_aware(integrand, weight, integrator, measure)
    result.name = "A_1"
    return _annotate_quantity(
        result,
        units="J",
        base_quantity="energy",
        measure=measure,
        long_name="exact available potential energy body term",
        surface_term_included=False,
    )


def available_potential_energy_zonal_part2(
    ps: xr.DataArray,
    phis: xr.DataArray,
    integrator: MassIntegrator,
    reference_state: Any | None = None,
    *,
    measure: TopographyAwareMeasure | None = None,
    surface_pressure_policy: str = "raise",
    pi_sZ: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_Z2 = ∫_S (ps - pi_sZ) Phi_s dσ / g`` in Joules."""

    ps = normalize_surface_field(ps, "ps")
    phis = normalize_surface_field(phis, "phis")
    ensure_matching_surface_coordinates(ps, [phis])
    measure = _resolved_measure(
        integrator,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_Z2",
    )

    ps_effective = _effective_surface_pressure(ps, measure)
    pi_sZ_field = _resolve_surface_reference_pressure(
        reference_state,
        pi_sZ,
        ps_effective,
        output_name="pi_sZ",
        pressure_keys=_SURFACE_ZONAL_PRESSURE_KEYS,
        zonal=True,
    )
    integrand = (ps_effective - pi_sZ_field) * phis
    result = integrator.integrate_surface(integrand)
    result.name = "A_Z2"
    return _annotate_quantity(
        result,
        units="J",
        base_quantity="energy",
        measure=measure,
        long_name="zonal exact available potential energy topographic term",
        surface_term_included=True,
    )


def available_potential_energy_eddy_part2(
    ps: xr.DataArray,
    phis: xr.DataArray,
    integrator: MassIntegrator,
    reference_state: Any | None = None,
    *,
    measure: TopographyAwareMeasure | None = None,
    surface_pressure_policy: str = "raise",
    pi_s: xr.DataArray | float | None = None,
    pi_sZ: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_E2 = ∫_S (pi_sZ - pi_s) Phi_s dσ / g`` in Joules."""

    ps = normalize_surface_field(ps, "ps")
    phis = normalize_surface_field(phis, "phis")
    ensure_matching_surface_coordinates(ps, [phis])
    measure = _resolved_measure(
        integrator,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_E2",
    )

    ps_effective = _effective_surface_pressure(ps, measure)
    pi_s_field = _resolve_surface_reference_pressure(
        reference_state,
        pi_s,
        ps_effective,
        output_name="pi_s",
        pressure_keys=_SURFACE_PRESSURE_KEYS,
        zonal=False,
    )
    pi_sZ_field = _resolve_surface_reference_pressure(
        reference_state,
        pi_sZ,
        ps_effective,
        output_name="pi_sZ",
        pressure_keys=_SURFACE_ZONAL_PRESSURE_KEYS,
        zonal=True,
    )
    integrand = (pi_sZ_field - pi_s_field) * phis
    result = integrator.integrate_surface(integrand)
    result.name = "A_E2"
    return _annotate_quantity(
        result,
        units="J",
        base_quantity="energy",
        measure=measure,
        long_name="eddy exact available potential energy topographic term",
        surface_term_included=True,
    )


def available_potential_energy_part2(
    ps: xr.DataArray,
    phis: xr.DataArray,
    integrator: MassIntegrator,
    reference_state: Any | None = None,
    *,
    measure: TopographyAwareMeasure | None = None,
    surface_pressure_policy: str = "raise",
    pi_s: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return ``A_2 = ∫_S (ps - pi_s) Phi_s dσ / g`` in Joules."""

    ps = normalize_surface_field(ps, "ps")
    phis = normalize_surface_field(phis, "phis")
    ensure_matching_surface_coordinates(ps, [phis])
    measure = _resolved_measure(
        integrator,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_2",
    )

    ps_effective = _effective_surface_pressure(ps, measure)
    pi_s_field = _resolve_surface_reference_pressure(
        reference_state,
        pi_s,
        ps_effective,
        output_name="pi_s",
        pressure_keys=_SURFACE_PRESSURE_KEYS,
        zonal=False,
    )
    integrand = (ps_effective - pi_s_field) * phis
    result = integrator.integrate_surface(integrand)
    result.name = "A_2"
    return _annotate_quantity(
        result,
        units="J",
        base_quantity="energy",
        measure=measure,
        long_name="exact available potential energy topographic term",
        surface_term_included=True,
    )


def available_potential_energy_zonal(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    reference_state: Any | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    potential_temperature_field: xr.DataArray | None = None,
    pi_z: xr.DataArray | float | None = None,
    n_z: xr.DataArray | float | None = None,
    pi_sZ: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return the total zonal exact APE ``A_Z = A_Z1 + A_Z2`` in Joules."""

    integrator = _require_integrator(integrator)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    if ps is None or phis is None:
        raise ValueError("Total exact A_Z requires both 'ps' and 'phis'.")
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_Z",
    )
    result = available_potential_energy_zonal_part1(
        temperature,
        pressure,
        theta_mask,
        integrator,
        reference_state=reference_state,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        potential_temperature_field=potential_temperature_field,
        pi_z=pi_z,
        n_z=n_z,
    ) + available_potential_energy_zonal_part2(
        ps,
        phis,
        integrator,
        reference_state=reference_state,
        measure=measure,
        surface_pressure_policy=surface_pressure_policy,
        pi_sZ=pi_sZ,
    )
    result.name = "A_Z"
    return _annotate_quantity(result, units="J", base_quantity="energy", measure=measure, surface_term_included=True)


def available_potential_energy_eddy(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    reference_state: Any | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    potential_temperature_field: xr.DataArray | None = None,
    pi: xr.DataArray | float | None = None,
    pi_z: xr.DataArray | float | None = None,
    n: xr.DataArray | float | None = None,
    n_z: xr.DataArray | float | None = None,
    pi_s: xr.DataArray | float | None = None,
    pi_sZ: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return the total eddy exact APE ``A_E = A_E1 + A_E2`` in Joules."""

    integrator = _require_integrator(integrator)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    if ps is None or phis is None:
        raise ValueError("Total exact A_E requires both 'ps' and 'phis'.")
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A_E",
    )
    result = available_potential_energy_eddy_part1(
        temperature,
        pressure,
        theta_mask,
        integrator,
        reference_state=reference_state,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        potential_temperature_field=potential_temperature_field,
        pi=pi,
        pi_z=pi_z,
        n=n,
        n_z=n_z,
    ) + available_potential_energy_eddy_part2(
        ps,
        phis,
        integrator,
        reference_state=reference_state,
        measure=measure,
        surface_pressure_policy=surface_pressure_policy,
        pi_s=pi_s,
        pi_sZ=pi_sZ,
    )
    result.name = "A_E"
    return _annotate_quantity(result, units="J", base_quantity="energy", measure=measure, surface_term_included=True)


def available_potential_energy(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    reference_state: Any | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    potential_temperature_field: xr.DataArray | None = None,
    pi: xr.DataArray | float | None = None,
    n: xr.DataArray | float | None = None,
    pi_s: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Return the total exact APE ``A = A_1 + A_2`` in Joules."""

    integrator = _require_integrator(integrator)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    if ps is None or phis is None:
        raise ValueError("Total exact A requires both 'ps' and 'phis'.")
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="A",
    )
    result = available_potential_energy_part1(
        temperature,
        pressure,
        theta_mask,
        integrator,
        reference_state=reference_state,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        potential_temperature_field=potential_temperature_field,
        pi=pi,
        n=n,
    ) + available_potential_energy_part2(
        ps,
        phis,
        integrator,
        reference_state=reference_state,
        measure=measure,
        surface_pressure_policy=surface_pressure_policy,
        pi_s=pi_s,
    )
    result.name = "A"
    return _annotate_quantity(result, units="J", base_quantity="energy", measure=measure, surface_term_included=True)


total_available_potential_energy_part1 = available_potential_energy_part1
total_available_potential_energy_part2 = available_potential_energy_part2
A_Z1 = available_potential_energy_zonal_part1
A_E1 = available_potential_energy_eddy_part1
A1 = available_potential_energy_part1
A_Z2 = available_potential_energy_zonal_part2
A_E2 = available_potential_energy_eddy_part2
A2 = available_potential_energy_part2
A_Z = available_potential_energy_zonal
A_E = available_potential_energy_eddy
A = available_potential_energy


__all__ = [
    "total_horizontal_ke",
    "kinetic_energy_zonal",
    "kinetic_energy_eddy",
    "available_potential_energy_zonal_part1",
    "available_potential_energy_eddy_part1",
    "available_potential_energy_part1",
    "available_potential_energy_zonal_part2",
    "available_potential_energy_eddy_part2",
    "available_potential_energy_part2",
    "available_potential_energy_zonal",
    "available_potential_energy_eddy",
    "available_potential_energy",
    "total_available_potential_energy_part1",
    "total_available_potential_energy_part2",
    "A_Z1",
    "A_E1",
    "A1",
    "A_Z2",
    "A_E2",
    "A2",
    "A_Z",
    "A_E",
    "A",
]
