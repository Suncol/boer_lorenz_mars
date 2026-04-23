"""Legacy finite-volume terrain-dependent reference-state solver for Mars exact APE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .._validation import (
    ensure_matching_coordinates,
    normalize_field,
)
from ..common.geopotential import broadcast_surface_field
from ..common.integrals import build_mass_integrator, pressure_level_edges
from ..common.topography_measure import TopographyAwareMeasure
from ..common.zonal_ops import weighted_representative_zonal_mean
from ..constants_mars import MARS, MarsConstants
from .solution import REFERENCE_INTERFACE_DIM, REFERENCE_SAMPLE_DIM, ReferenceStateSolution


_LOWER_BOUNDARY_MAX_ITERATIONS = 64
_LOWER_BOUNDARY_PRESSURE_TOLERANCE = 1.0e-6
_MASS_RESIDUAL_FACTOR = 10.0
_PRESSURE_EPSILON = 1.0e-12


@dataclass(frozen=True)
class _MassSpectrum:
    theta_layers: np.ndarray
    layer_mass: np.ndarray
    total_mass: float


@dataclass(frozen=True)
class _MarchedProfile:
    theta_layers: np.ndarray
    layer_mass: np.ndarray
    p_interfaces: np.ndarray
    phi_interfaces: np.ndarray
    top_residual: float
    feasible: bool


@dataclass(frozen=True)
class _SolvedTimeSlice:
    theta_reference: np.ndarray
    pi_reference: np.ndarray
    mass_reference: np.ndarray
    reference_interface_pressure: np.ndarray
    reference_interface_geopotential: np.ndarray
    pi_s: np.ndarray
    reference_surface_pressure: float
    reference_bottom_pressure: float
    total_mass: float
    iterations: int
    converged: bool


def _build_theta_mass_spectrum(theta_3d: np.ndarray, parcel_mass_3d: np.ndarray) -> _MassSpectrum:
    theta_flat = np.asarray(theta_3d, dtype=float).reshape(-1)
    mass_flat = np.asarray(parcel_mass_3d, dtype=float).reshape(-1)
    valid = np.isfinite(theta_flat) & np.isfinite(mass_flat) & (mass_flat > 0.0)
    if not np.any(valid):
        raise ValueError("Reference-state solve requires at least one above-ground parcel.")

    theta_valid = theta_flat[valid]
    mass_valid = mass_flat[valid]
    order = np.argsort(theta_valid, kind="mergesort")
    theta_sorted = theta_valid[order]
    mass_sorted = mass_valid[order]

    theta_layers, group_start, _ = np.unique(
        theta_sorted,
        return_index=True,
        return_counts=True,
    )
    layer_mass = np.add.reduceat(mass_sorted, group_start)
    total_mass = float(layer_mass.sum())
    return _MassSpectrum(
        theta_layers=np.asarray(theta_layers, dtype=float),
        layer_mass=np.asarray(layer_mass, dtype=float),
        total_mass=total_mass,
    )


def _normalize_relative_surface_geopotential(surface_geopotential: np.ndarray) -> np.ndarray:
    phis = np.asarray(surface_geopotential, dtype=float).reshape(-1)
    if not np.all(np.isfinite(phis)):
        raise ValueError("Surface geopotential must remain finite for the stage-3 reference-state solve.")
    return phis - float(np.min(phis))


def _layer_geopotential_drop(
    theta_layer: float,
    p_bottom: float,
    p_top: float,
    *,
    constants: MarsConstants,
) -> float:
    if not (p_bottom > p_top > 0.0):
        raise ValueError("Layer geopotential thickness requires p_bottom > p_top > 0.")
    exner_bottom = (p_bottom / constants.p00) ** constants.kappa
    exner_top = (p_top / constants.p00) ** constants.kappa
    return float(constants.cp * theta_layer * (exner_bottom - exner_top))


def _pressure_at_geopotential_within_layer(
    phi_target: np.ndarray,
    phi_bottom: float,
    p_bottom: float,
    theta_layer: float,
    *,
    constants: MarsConstants,
) -> np.ndarray:
    phi = np.asarray(phi_target, dtype=float)
    exner_bottom = (p_bottom / constants.p00) ** constants.kappa
    exner = exner_bottom - (phi - phi_bottom) / (constants.cp * theta_layer)
    if np.any(exner < -_PRESSURE_EPSILON):
        raise ValueError("Requested geopotential lies outside the solved reference layer.")
    exner = np.maximum(exner, 0.0)
    return constants.p00 * np.power(exner, 1.0 / constants.kappa)


def _layer_local_bottom_pressures(
    p_bottom: float,
    p_top: float,
    phi_bottom: float,
    phi_top: float,
    theta_layer: float,
    phis_rel_flat: np.ndarray,
    *,
    constants: MarsConstants,
) -> np.ndarray:
    local_bottom = np.full_like(phis_rel_flat, p_top, dtype=float)
    full_mask = phis_rel_flat <= phi_bottom
    if np.any(full_mask):
        local_bottom[full_mask] = p_bottom

    partial_mask = (phis_rel_flat > phi_bottom) & (phis_rel_flat < phi_top)
    if np.any(partial_mask):
        local_bottom[partial_mask] = _pressure_at_geopotential_within_layer(
            phis_rel_flat[partial_mask],
            phi_bottom,
            p_bottom,
            theta_layer,
            constants=constants,
        )
    return local_bottom


def _reference_layer_mass(
    p_bottom: float,
    p_top: float,
    phi_bottom: float,
    theta_layer: float,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    *,
    constants: MarsConstants,
) -> tuple[float, float]:
    phi_top = phi_bottom + _layer_geopotential_drop(
        theta_layer,
        p_bottom,
        p_top,
        constants=constants,
    )
    local_bottom = _layer_local_bottom_pressures(
        p_bottom,
        p_top,
        phi_bottom,
        phi_top,
        theta_layer,
        phis_rel_flat,
        constants=constants,
    )
    layer_mass = float(np.sum((local_bottom - p_top) * area_weights_flat) / constants.g)
    return layer_mass, float(phi_top)


def _solve_layer_top_pressure(
    p_bottom: float,
    phi_bottom: float,
    theta_layer: float,
    target_mass: float,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    reference_top_pressure: float,
    planetary_area: float,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> tuple[float, float] | None:
    max_phi = float(np.max(phis_rel_flat))
    if phi_bottom >= max_phi:
        p_top = p_bottom - target_mass * constants.g / planetary_area
        if p_top <= reference_top_pressure:
            if np.isclose(
                p_top,
                reference_top_pressure,
                rtol=pressure_tolerance,
                atol=0.0,
            ):
                p_top = reference_top_pressure
            else:
                return None
        phi_top = phi_bottom + _layer_geopotential_drop(
            theta_layer,
            p_bottom,
            p_top,
            constants=constants,
        )
        return float(p_top), float(phi_top)

    lower = reference_top_pressure
    max_mass, max_phi_top = _reference_layer_mass(
        p_bottom,
        lower,
        phi_bottom,
        theta_layer,
        phis_rel_flat,
        area_weights_flat,
        constants=constants,
    )
    mass_tolerance = pressure_tolerance * planetary_area * max(p_bottom, 1.0) / constants.g
    if max_mass + mass_tolerance < target_mass:
        return None

    upper = p_bottom
    upper_mass = 0.0
    if abs(max_mass - target_mass) <= mass_tolerance:
        return float(lower), float(max_phi_top)

    p_top = 0.5 * (lower + upper)
    phi_top = max_phi_top
    for _ in range(max_iterations):
        p_top = 0.5 * (lower + upper)
        layer_mass, phi_top = _reference_layer_mass(
            p_bottom,
            p_top,
            phi_bottom,
            theta_layer,
            phis_rel_flat,
            area_weights_flat,
            constants=constants,
        )
        residual = layer_mass - target_mass
        if abs(residual) <= mass_tolerance or abs(upper - lower) <= pressure_tolerance * max(p_bottom, 1.0):
            return float(p_top), float(phi_top)
        if residual > 0.0:
            lower = p_top
            upper_mass = layer_mass
        else:
            upper = p_top

    # Use the last midpoint if it stays physically ordered and close enough.
    if upper > lower and upper_mass >= 0.0:
        return float(p_top), float(phi_top)
    return None


def _march_reference_interfaces(
    spectrum: _MassSpectrum,
    reference_bottom_pressure: float,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    reference_top_pressure: float,
    planetary_area: float,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> _MarchedProfile:
    nlayers = spectrum.theta_layers.size
    p_interfaces = np.full(nlayers + 1, np.nan, dtype=float)
    phi_interfaces = np.full(nlayers + 1, np.nan, dtype=float)
    p_interfaces[0] = reference_bottom_pressure
    phi_interfaces[0] = 0.0

    for layer_index, (theta_layer, target_mass) in enumerate(
        zip(spectrum.theta_layers, spectrum.layer_mass, strict=True)
    ):
        solved = _solve_layer_top_pressure(
            float(p_interfaces[layer_index]),
            float(phi_interfaces[layer_index]),
            float(theta_layer),
            float(target_mass),
            phis_rel_flat,
            area_weights_flat,
            reference_top_pressure,
            planetary_area,
            constants=constants,
            max_iterations=max_iterations,
            pressure_tolerance=pressure_tolerance,
        )
        if solved is None:
            return _MarchedProfile(
                theta_layers=spectrum.theta_layers,
                layer_mass=spectrum.layer_mass,
                p_interfaces=p_interfaces,
                phi_interfaces=phi_interfaces,
                top_residual=-np.inf,
                feasible=False,
            )
        p_top, phi_top = solved
        p_interfaces[layer_index + 1] = p_top
        phi_interfaces[layer_index + 1] = phi_top

    return _MarchedProfile(
        theta_layers=spectrum.theta_layers,
        layer_mass=spectrum.layer_mass,
        p_interfaces=p_interfaces,
        phi_interfaces=phi_interfaces,
        top_residual=float(p_interfaces[-1] - reference_top_pressure),
        feasible=True,
    )


def _profile_sign(profile: _MarchedProfile) -> int:
    if not profile.feasible:
        return -1
    return 1 if profile.top_residual >= 0.0 else -1


def _bracket_reference_bottom_pressure(
    spectrum: _MassSpectrum,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    reference_top_pressure: float,
    planetary_area: float,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> tuple[float, float, _MarchedProfile, _MarchedProfile]:
    pressure_scale = spectrum.total_mass * constants.g / planetary_area
    lower = max(reference_top_pressure * (1.0 + pressure_tolerance), reference_top_pressure + pressure_scale)
    profile_lower = _march_reference_interfaces(
        spectrum,
        lower,
        phis_rel_flat,
        area_weights_flat,
        reference_top_pressure,
        planetary_area,
        constants=constants,
        max_iterations=max_iterations,
        pressure_tolerance=pressure_tolerance,
    )

    lower_adjustments = 0
    while _profile_sign(profile_lower) > 0 and lower_adjustments < max_iterations:
        candidate = 0.5 * (lower + reference_top_pressure)
        if candidate <= reference_top_pressure * (1.0 + pressure_tolerance):
            break
        lower = candidate
        profile_lower = _march_reference_interfaces(
            spectrum,
            lower,
            phis_rel_flat,
            area_weights_flat,
            reference_top_pressure,
            planetary_area,
            constants=constants,
            max_iterations=max_iterations,
            pressure_tolerance=pressure_tolerance,
        )
        lower_adjustments += 1

    upper = max(lower + pressure_scale, lower * 1.25)
    profile_upper = _march_reference_interfaces(
        spectrum,
        upper,
        phis_rel_flat,
        area_weights_flat,
        reference_top_pressure,
        planetary_area,
        constants=constants,
        max_iterations=max_iterations,
        pressure_tolerance=pressure_tolerance,
    )

    upper_adjustments = 0
    while _profile_sign(profile_upper) < 0 and upper_adjustments < 2 * max_iterations:
        upper = reference_top_pressure + 2.0 * (upper - reference_top_pressure)
        profile_upper = _march_reference_interfaces(
            spectrum,
            upper,
            phis_rel_flat,
            area_weights_flat,
            reference_top_pressure,
            planetary_area,
            constants=constants,
            max_iterations=max_iterations,
            pressure_tolerance=pressure_tolerance,
        )
        upper_adjustments += 1

    if _profile_sign(profile_lower) > 0 or _profile_sign(profile_upper) < 0:
        raise ValueError("Failed to bracket the terrain-dependent reference bottom pressure solve.")

    return lower, upper, profile_lower, profile_upper


def _solve_reference_bottom_pressure(
    spectrum: _MassSpectrum,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    reference_top_pressure: float,
    planetary_area: float,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> tuple[_MarchedProfile, int, bool]:
    lower, upper, profile_lower, profile_upper = _bracket_reference_bottom_pressure(
        spectrum,
        phis_rel_flat,
        area_weights_flat,
        reference_top_pressure,
        planetary_area,
        constants=constants,
        max_iterations=max_iterations,
        pressure_tolerance=pressure_tolerance,
    )

    # In partial-cell and clipped-column cases, the exact root can sit at the
    # feasible-profile boundary. Using the top-pressure scale alone can make the
    # bisection reject otherwise closed solutions with sub-millipascal residuals.
    pressure_scale = spectrum.total_mass * constants.g / planetary_area
    tolerance = pressure_tolerance * max(reference_top_pressure, pressure_scale, 1.0)
    if profile_lower.feasible and abs(profile_lower.top_residual) <= tolerance:
        return profile_lower, 1, True
    if profile_upper.feasible and abs(profile_upper.top_residual) <= tolerance:
        return profile_upper, 1, True

    best_profile = profile_upper if profile_upper.feasible else profile_lower
    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (lower + upper)
        profile_mid = _march_reference_interfaces(
            spectrum,
            midpoint,
            phis_rel_flat,
            area_weights_flat,
            reference_top_pressure,
            planetary_area,
            constants=constants,
            max_iterations=max_iterations,
            pressure_tolerance=pressure_tolerance,
        )
        if profile_mid.feasible:
            best_profile = profile_mid
            if abs(profile_mid.top_residual) <= tolerance:
                return profile_mid, iteration, True

        if _profile_sign(profile_mid) < 0:
            lower = midpoint
            profile_lower = profile_mid
        else:
            upper = midpoint
            profile_upper = profile_mid

        if best_profile.feasible and abs(upper - lower) <= tolerance:
            return best_profile, iteration, True

    return best_profile, max_iterations, False


def _surface_pressure_from_solved_profile(
    profile: _MarchedProfile,
    phis_rel_flat: np.ndarray,
    *,
    constants: MarsConstants,
) -> np.ndarray:
    if not profile.feasible:
        raise ValueError("Surface pressure reconstruction requires a feasible marched profile.")
    if np.max(phis_rel_flat) > profile.phi_interfaces[-1] + _PRESSURE_EPSILON:
        raise ValueError("Solved reference profile does not extend above the maximum relative topography.")

    nlayers = profile.theta_layers.size
    layer_index = np.searchsorted(profile.phi_interfaces, phis_rel_flat, side="right") - 1
    layer_index = np.clip(layer_index, 0, nlayers - 1)

    pi_s = np.empty_like(phis_rel_flat, dtype=float)
    for k in range(nlayers):
        mask = layer_index == k
        if not np.any(mask):
            continue
        pi_s[mask] = _pressure_at_geopotential_within_layer(
            phis_rel_flat[mask],
            float(profile.phi_interfaces[k]),
            float(profile.p_interfaces[k]),
            float(profile.theta_layers[k]),
            constants=constants,
        )
    return pi_s


def _half_mass_pressure_in_layer(
    p_bottom: float,
    p_top: float,
    phi_bottom: float,
    phi_top: float,
    theta_layer: float,
    layer_mass: float,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> float:
    local_bottom = _layer_local_bottom_pressures(
        p_bottom,
        p_top,
        phi_bottom,
        phi_top,
        theta_layer,
        phis_rel_flat,
        constants=constants,
    )
    target_mass = 0.5 * layer_mass
    lower = p_top
    upper = p_bottom
    mass_tolerance = pressure_tolerance * max(layer_mass, 1.0)
    p_mid = 0.5 * (lower + upper)
    for _ in range(max_iterations):
        p_mid = 0.5 * (lower + upper)
        upper_half_mass = float(
            np.sum(np.maximum(np.minimum(local_bottom, p_mid) - p_top, 0.0) * area_weights_flat) / constants.g
        )
        residual = upper_half_mass - target_mass
        if abs(residual) <= mass_tolerance or abs(upper - lower) <= pressure_tolerance * max(p_bottom, 1.0):
            return float(p_mid)
        if residual > 0.0:
            upper = p_mid
        else:
            lower = p_mid
    return float(p_mid)


def _reference_curve_from_solved_profile(
    profile: _MarchedProfile,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> np.ndarray:
    pi_reference = np.empty(profile.theta_layers.size, dtype=float)
    for layer_index in range(profile.theta_layers.size):
        pi_reference[layer_index] = _half_mass_pressure_in_layer(
            float(profile.p_interfaces[layer_index]),
            float(profile.p_interfaces[layer_index + 1]),
            float(profile.phi_interfaces[layer_index]),
            float(profile.phi_interfaces[layer_index + 1]),
            float(profile.theta_layers[layer_index]),
            float(profile.layer_mass[layer_index]),
            phis_rel_flat,
            area_weights_flat,
            constants=constants,
            max_iterations=max_iterations,
            pressure_tolerance=pressure_tolerance,
        )
    return pi_reference


def _validate_time_slice_solution(
    spectrum: _MassSpectrum,
    profile: _MarchedProfile,
    pi_reference: np.ndarray,
    pi_s_flat: np.ndarray,
    phis_rel_flat: np.ndarray,
    area_weights_flat: np.ndarray,
    reference_top_pressure: float,
    *,
    constants: MarsConstants,
    pressure_tolerance: float,
) -> None:
    if not np.all(np.diff(spectrum.theta_layers) > 0.0):
        raise ValueError("Reference-state theta layers must remain strictly increasing.")
    if not np.all(np.diff(profile.p_interfaces) < 0.0):
        raise ValueError("Reference-state pressure interfaces must remain strictly decreasing.")
    if not np.all(np.diff(profile.phi_interfaces) > 0.0):
        raise ValueError("Reference-state geopotential interfaces must remain strictly increasing.")
    if not np.all(np.isfinite(pi_s_flat)) or np.any(pi_s_flat <= 0.0):
        raise ValueError("Reference-state surface pressures must remain finite and strictly positive.")
    if not np.all(np.diff(pi_reference) < 0.0):
        raise ValueError("Reference-state pressure curve must remain strictly decreasing with theta.")

    reference_bottom_pressure = float(profile.p_interfaces[0])
    pressure_scale = max(reference_bottom_pressure, 1.0)
    if not np.isclose(
        float(np.max(pi_s_flat)),
        reference_bottom_pressure,
        rtol=pressure_tolerance,
        atol=0.0,
    ):
        raise ValueError("Reference-state bottom pressure must match the deepest surface pressure.")

    total_mass_from_surface = float(np.sum((pi_s_flat - reference_top_pressure) * area_weights_flat) / constants.g)
    if not np.isclose(
        total_mass_from_surface,
        spectrum.total_mass,
        rtol=_MASS_RESIDUAL_FACTOR * pressure_tolerance,
        atol=0.0,
    ):
        raise ValueError("Terrain-dependent reference-state surface pressures do not reproduce the total mass.")

    mass_tolerance = _MASS_RESIDUAL_FACTOR * pressure_tolerance * max(spectrum.total_mass, 1.0)
    for layer_index, target_mass in enumerate(spectrum.layer_mass):
        ref_mass, _ = _reference_layer_mass(
            float(profile.p_interfaces[layer_index]),
            float(profile.p_interfaces[layer_index + 1]),
            float(profile.phi_interfaces[layer_index]),
            float(profile.theta_layers[layer_index]),
            phis_rel_flat,
            area_weights_flat,
            constants=constants,
        )
        if abs(ref_mass - float(target_mass)) > mass_tolerance:
            raise ValueError("Terrain-dependent reference-state layer masses failed the exact-mass closure check.")

        pi_k = float(pi_reference[layer_index])
        if not (profile.p_interfaces[layer_index] > pi_k > profile.p_interfaces[layer_index + 1]):
            raise ValueError("Reference pressure samples must lie strictly inside their solved isentropic layers.")

        local_bottom = _layer_local_bottom_pressures(
            float(profile.p_interfaces[layer_index]),
            float(profile.p_interfaces[layer_index + 1]),
            float(profile.phi_interfaces[layer_index]),
            float(profile.phi_interfaces[layer_index + 1]),
            float(profile.theta_layers[layer_index]),
            phis_rel_flat,
            constants=constants,
        )
        upper_half_mass = float(
            np.sum(np.maximum(np.minimum(local_bottom, pi_k) - profile.p_interfaces[layer_index + 1], 0.0) * area_weights_flat)
            / constants.g
        )
        if abs(upper_half_mass - 0.5 * float(target_mass)) > mass_tolerance:
            raise ValueError("Reference pressure samples must be half-mass pressures within each solved layer.")

    if np.max(phis_rel_flat) > profile.phi_interfaces[-1] + _PRESSURE_EPSILON * pressure_scale:
        raise ValueError("Terrain-dependent reference atmosphere does not extend above the maximum relative topography.")


def _solve_time_slice_reference_state(
    theta_3d: np.ndarray,
    parcel_mass_3d: np.ndarray,
    phis_2d: np.ndarray,
    area_weights_2d: np.ndarray,
    reference_top_pressure: float,
    planetary_area: float,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
) -> _SolvedTimeSlice:
    spectrum = _build_theta_mass_spectrum(theta_3d, parcel_mass_3d)
    phis_rel_flat = _normalize_relative_surface_geopotential(phis_2d)
    area_weights_flat = np.asarray(area_weights_2d, dtype=float).reshape(-1)

    profile, iterations, converged = _solve_reference_bottom_pressure(
        spectrum,
        phis_rel_flat,
        area_weights_flat,
        float(reference_top_pressure),
        float(planetary_area),
        constants=constants,
        max_iterations=max_iterations,
        pressure_tolerance=pressure_tolerance,
    )
    if not converged:
        raise ValueError("Stage-3 terrain-dependent reference-state solve failed to converge.")

    pi_s_flat = _surface_pressure_from_solved_profile(profile, phis_rel_flat, constants=constants)
    pi_reference = _reference_curve_from_solved_profile(
        profile,
        phis_rel_flat,
        area_weights_flat,
        constants=constants,
        max_iterations=max_iterations,
        pressure_tolerance=pressure_tolerance,
    )
    _validate_time_slice_solution(
        spectrum,
        profile,
        pi_reference,
        pi_s_flat,
        phis_rel_flat,
        area_weights_flat,
        float(reference_top_pressure),
        constants=constants,
        pressure_tolerance=pressure_tolerance,
    )

    reference_surface_pressure = float(np.sum(pi_s_flat * area_weights_flat) / np.sum(area_weights_flat))
    return _SolvedTimeSlice(
        theta_reference=spectrum.theta_layers,
        pi_reference=pi_reference,
        mass_reference=spectrum.layer_mass,
        reference_interface_pressure=np.asarray(profile.p_interfaces, dtype=float),
        reference_interface_geopotential=np.asarray(profile.phi_interfaces, dtype=float),
        pi_s=pi_s_flat.reshape(phis_2d.shape),
        reference_surface_pressure=reference_surface_pressure,
        reference_bottom_pressure=float(profile.p_interfaces[0]),
        total_mass=spectrum.total_mass,
        iterations=iterations,
        converged=converged,
    )


def _solve_reference_family(
    potential_temperature: xr.DataArray,
    parcel_mass: xr.DataArray,
    surface_geopotential: xr.DataArray,
    reference_top_pressure: float,
    *,
    constants: MarsConstants,
    max_iterations: int,
    pressure_tolerance: float,
    integrator_cell_area: xr.DataArray,
) -> dict[str, np.ndarray]:
    ntime = potential_temperature.sizes["time"]
    nlat = potential_temperature.sizes["latitude"]
    nlon = potential_temperature.sizes["longitude"]
    max_groups = potential_temperature.sizes["level"] * nlat * nlon
    max_interfaces = max_groups + 1

    theta_reference = np.full((ntime, max_groups), np.nan, dtype=float)
    pi_reference = np.full((ntime, max_groups), np.nan, dtype=float)
    mass_reference = np.full((ntime, max_groups), np.nan, dtype=float)
    reference_interface_pressure = np.full((ntime, max_interfaces), np.nan, dtype=float)
    reference_interface_geopotential = np.full((ntime, max_interfaces), np.nan, dtype=float)
    total_mass = np.zeros(ntime, dtype=float)
    reference_surface_pressure = np.zeros(ntime, dtype=float)
    reference_bottom_pressure = np.zeros(ntime, dtype=float)
    pi_s_values = np.full((ntime, nlat, nlon), np.nan, dtype=float)
    iteration_values = np.zeros(ntime, dtype=int)
    converged_values = np.zeros(ntime, dtype=bool)

    theta_values = np.asarray(potential_temperature.values, dtype=float)
    mass_values = np.asarray(parcel_mass.values, dtype=float)
    surface_values = np.asarray(surface_geopotential.values, dtype=float)
    area_weights = np.asarray(integrator_cell_area.values, dtype=float)
    planetary_area = float(np.sum(area_weights))

    for time_index in range(ntime):
        solved = _solve_time_slice_reference_state(
            theta_values[time_index],
            mass_values[time_index],
            surface_values[time_index],
            area_weights,
            reference_top_pressure,
            planetary_area,
            constants=constants,
            max_iterations=max_iterations,
            pressure_tolerance=pressure_tolerance,
        )
        ngroups = solved.theta_reference.size
        ninterfaces = solved.reference_interface_pressure.size
        theta_reference[time_index, :ngroups] = solved.theta_reference
        pi_reference[time_index, :ngroups] = solved.pi_reference
        mass_reference[time_index, :ngroups] = solved.mass_reference
        reference_interface_pressure[time_index, :ninterfaces] = solved.reference_interface_pressure
        reference_interface_geopotential[time_index, :ninterfaces] = solved.reference_interface_geopotential
        total_mass[time_index] = solved.total_mass
        reference_surface_pressure[time_index] = solved.reference_surface_pressure
        reference_bottom_pressure[time_index] = solved.reference_bottom_pressure
        pi_s_values[time_index] = solved.pi_s
        iteration_values[time_index] = solved.iterations
        converged_values[time_index] = solved.converged

    return {
        "theta_reference": theta_reference,
        "pi_reference": pi_reference,
        "mass_reference": mass_reference,
        "reference_interface_pressure": reference_interface_pressure,
        "reference_interface_geopotential": reference_interface_geopotential,
        "total_mass": total_mass,
        "reference_surface_pressure": reference_surface_pressure,
        "reference_bottom_pressure": reference_bottom_pressure,
        "pi_s": pi_s_values,
        "iterations": iteration_values,
        "converged": converged_values,
        "max_groups": np.asarray([max_groups], dtype=int),
        "max_interfaces": np.asarray([max_interfaces], dtype=int),
    }


class FiniteVolumeReferenceState:
    """Legacy pressure-grid finite-volume reference-state solver.

    This solver preserves the existing parcel-sorted, terrain-aware marched-profile
    implementation that was previously exported as ``KoehlerReferenceState``.
    It is retained as a stable legacy reference-state branch while the full
    fixed-isentrope ``Koehler1986ReferenceState`` is developed separately.
    """

    def __init__(
        self,
        constants: MarsConstants = MARS,
        *,
        max_iterations: int = _LOWER_BOUNDARY_MAX_ITERATIONS,
        pressure_tolerance: float = _LOWER_BOUNDARY_PRESSURE_TOLERANCE,
        surface_pressure_policy: str = "raise",
    ) -> None:
        self.constants = constants
        self.max_iterations = max_iterations
        self.pressure_tolerance = pressure_tolerance
        self.surface_pressure_policy = surface_pressure_policy

    def solve(
        self,
        potential_temperature: xr.DataArray,
        pressure: xr.DataArray,
        ps: xr.DataArray,
        phis: xr.DataArray | None = None,
        *,
        level_bounds: xr.DataArray | None = None,
    ) -> ReferenceStateSolution:
        """Return the legacy finite-volume terrain-dependent reference-state solution."""

        potential_temperature = normalize_field(
            potential_temperature,
            "potential_temperature",
        )
        pressure = normalize_field(pressure, "pressure")
        ensure_matching_coordinates(potential_temperature, [pressure])

        surface_pressure = broadcast_surface_field(ps, potential_temperature, "ps")
        integrator = build_mass_integrator(
            potential_temperature.coords["level"],
            potential_temperature.coords["latitude"],
            potential_temperature.coords["longitude"],
            constants=self.constants,
            level_bounds=level_bounds,
        )
        measure = TopographyAwareMeasure.from_surface_pressure(
            potential_temperature.coords["level"],
            surface_pressure,
            integrator,
            level_bounds=level_bounds,
            pressure_tolerance=self.pressure_tolerance,
            surface_pressure_policy=self.surface_pressure_policy,
        )
        parcel_mass = measure.parcel_mass

        if phis is None:
            surface_geopotential = xr.zeros_like(potential_temperature.isel(level=0, drop=True), dtype=float)
            surface_geopotential.name = "phis"
        else:
            surface_geopotential = broadcast_surface_field(phis, potential_temperature, "phis")

        level_edges = pressure_level_edges(potential_temperature.coords["level"], bounds=level_bounds)
        reference_top_pressure_value = float(level_edges.isel(level_edge=-1))
        reference_top_pressure = xr.full_like(
            potential_temperature.coords["time"],
            reference_top_pressure_value,
            dtype=float,
        )

        full_solution = _solve_reference_family(
            potential_temperature,
            parcel_mass,
            surface_geopotential,
            reference_top_pressure_value,
            constants=self.constants,
            max_iterations=self.max_iterations,
            pressure_tolerance=self.pressure_tolerance,
            integrator_cell_area=integrator.cell_area,
        )

        representative_theta = weighted_representative_zonal_mean(
            potential_temperature,
            measure.cell_fraction,
        ).broadcast_like(
            potential_temperature
        )
        zonal_solution = _solve_reference_family(
            representative_theta,
            parcel_mass,
            surface_geopotential,
            reference_top_pressure_value,
            constants=self.constants,
            max_iterations=self.max_iterations,
            pressure_tolerance=self.pressure_tolerance,
            integrator_cell_area=integrator.cell_area,
        )

        max_groups = int(full_solution["max_groups"][0])
        max_interfaces = int(full_solution["max_interfaces"][0])
        sample_coords = {
            "time": potential_temperature.coords["time"].values,
            REFERENCE_SAMPLE_DIM: np.arange(max_groups),
        }
        interface_coords = {
            "time": potential_temperature.coords["time"].values,
            REFERENCE_INTERFACE_DIM: np.arange(max_interfaces),
        }
        surface_coords = {
            "time": potential_temperature.coords["time"].values,
            "latitude": potential_temperature.coords["latitude"].values,
            "longitude": potential_temperature.coords["longitude"].values,
        }

        def _annotate_domain(field: xr.DataArray) -> xr.DataArray:
            return measure.annotate_domain_metadata(field)

        return ReferenceStateSolution(
            theta_reference=xr.DataArray(
                full_solution["theta_reference"],
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=sample_coords,
                name="theta_reference",
                attrs={
                    "units": "K",
                    "reference_coordinate_semantics": "finite_volume_theta_groups",
                },
            ),
            pi_reference=xr.DataArray(
                full_solution["pi_reference"],
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=sample_coords,
                name="pi_reference",
                attrs={
                    "units": "Pa",
                    "reference_pressure_sampling": "half_mass_pressure_sample",
                },
            ),
            mass_reference=xr.DataArray(
                full_solution["mass_reference"],
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=sample_coords,
                name="isentropic_mass",
                attrs={"units": "kg"},
            ),
            reference_interface_pressure=xr.DataArray(
                full_solution["reference_interface_pressure"],
                dims=("time", REFERENCE_INTERFACE_DIM),
                coords=interface_coords,
                name="reference_interface_pressure",
                attrs={"units": "Pa", "long_name": "reference-state pressure interfaces"},
            ),
            reference_interface_geopotential=xr.DataArray(
                full_solution["reference_interface_geopotential"],
                dims=("time", REFERENCE_INTERFACE_DIM),
                coords=interface_coords,
                name="reference_interface_geopotential",
                attrs={"units": "m2 s-2", "long_name": "reference-state geopotential interfaces"},
            ),
            total_mass=_annotate_domain(
                xr.DataArray(
                    full_solution["total_mass"],
                    dims=("time",),
                    coords={"time": potential_temperature.coords["time"].values},
                    name="total_mass",
                    attrs={"units": "kg"},
                )
            ),
            reference_surface_pressure=_annotate_domain(
                xr.DataArray(
                    full_solution["reference_surface_pressure"],
                    dims=("time",),
                    coords={"time": potential_temperature.coords["time"].values},
                    name="reference_surface_pressure",
                    attrs={"units": "Pa", "long_name": "area-weighted mean reference-state surface pressure"},
                )
            ),
            reference_bottom_pressure=_annotate_domain(
                xr.DataArray(
                    full_solution["reference_bottom_pressure"],
                    dims=("time",),
                    coords={"time": potential_temperature.coords["time"].values},
                    name="reference_bottom_pressure",
                    attrs={"units": "Pa", "long_name": "deepest reference-state surface pressure"},
                )
            ),
            reference_top_pressure=reference_top_pressure.rename("reference_top_pressure"),
            ps_effective=_annotate_domain(measure.effective_surface_pressure.rename("ps_effective")),
            pi_s=_annotate_domain(
                xr.DataArray(
                    full_solution["pi_s"],
                    dims=("time", "latitude", "longitude"),
                    coords=surface_coords,
                    name="pi_s",
                    attrs={"units": "Pa", "long_name": "reference-state surface pressure"},
                )
            ),
            pi_sZ=_annotate_domain(
                xr.DataArray(
                    zonal_solution["pi_s"],
                    dims=("time", "latitude", "longitude"),
                    coords=surface_coords,
                    name="pi_sZ",
                    attrs={
                        "units": "Pa",
                        "long_name": "zonal-thermodynamic reference-state surface pressure on actual topography",
                    },
                )
            ),
            iterations=xr.DataArray(
                full_solution["iterations"],
                dims=("time",),
                coords={"time": potential_temperature.coords["time"].values},
                name="reference_state_iterations",
            ),
            converged=xr.DataArray(
                full_solution["converged"],
                dims=("time",),
                coords={"time": potential_temperature.coords["time"].values},
                name="reference_state_converged",
            ),
            iterations_zonal=xr.DataArray(
                zonal_solution["iterations"],
                dims=("time",),
                coords={"time": potential_temperature.coords["time"].values},
                name="reference_state_iterations_zonal",
            ),
            converged_zonal=xr.DataArray(
                zonal_solution["converged"],
                dims=("time",),
                coords={"time": potential_temperature.coords["time"].values},
                name="reference_state_converged_zonal",
            ),
            method="finite_volume_parcel_sorted",
            constants=self.constants,
            _theta_reference_zonal=xr.DataArray(
                zonal_solution["theta_reference"],
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=sample_coords,
                name="_theta_reference_zonal",
                attrs={"units": "K"},
            ),
            _pi_reference_zonal=xr.DataArray(
                zonal_solution["pi_reference"],
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=sample_coords,
                name="_pi_reference_zonal",
                attrs={"units": "Pa"},
            ),
            _reference_bottom_pressure_zonal=xr.DataArray(
                zonal_solution["reference_bottom_pressure"],
                dims=("time",),
                coords={"time": potential_temperature.coords["time"].values},
                name="_reference_bottom_pressure_zonal",
                attrs={"units": "Pa"},
            ),
        )


__all__ = [
    "REFERENCE_SAMPLE_DIM",
    "REFERENCE_INTERFACE_DIM",
    "ReferenceStateSolution",
    "FiniteVolumeReferenceState",
    "_solve_reference_family",
]
