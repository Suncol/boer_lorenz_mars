"""Fixed-isentrope, surface-aware Koehler (1986) reference-state solver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy import optimize

from ..common.geopotential import broadcast_surface_field
from ..common.grid_weights import longitude_weights
from ..common.zonal_ops import weighted_representative_zonal_mean
from ..constants_mars import MARS, MarsConstants
from .interpolate_isentropes import ISENTROPIC_DIM, ISENTROPIC_LAYER_DIM
from .koehler1986_geometry import (
    _KoehlerGeometryFamily,
    _build_koehler_geometry_family_xr,
    _pressure_to_exner,
    _exner_to_pressure,
    _reference_geopotential_with_anchor_details,
    _reference_mass_from_profile_numpy,
)
from .koehler1986_preprocessing import (
    _KoehlerObservedStateContext,
    _prepare_koehler1986_observed_context,
)
from .solution import REFERENCE_INTERFACE_DIM, REFERENCE_SAMPLE_DIM, ReferenceStateSolution


@dataclass(frozen=True)
class _KoehlerSolvedFamily:
    """Solved full or zonal Koehler family on the fixed-isentrope grid."""

    family_name: str
    context: _KoehlerObservedStateContext
    geometry: _KoehlerGeometryFamily
    mean_pressure_observed: xr.DataArray
    mean_pressure_reference: xr.DataArray
    layer_mass_observed: xr.DataArray
    layer_mass_reference: xr.DataArray
    mass_residual: xr.DataArray
    iterations: xr.DataArray
    converged: xr.DataArray
    solver_strategy: str


@dataclass(frozen=True)
class _KoehlerGeometryState:
    """Private Phase-4 cache kept on the solver instance for debugging/tests."""

    phis_surface: xr.DataArray
    representative_theta: xr.DataArray
    representative_pressure: xr.DataArray
    surface_potential_temperature_zonal: xr.DataArray
    full_family: _KoehlerSolvedFamily
    zonal_family: _KoehlerSolvedFamily


@dataclass(frozen=True)
class _KoehlerSolvedTimeSlice:
    """Per-time solved K86 pressure profile."""

    pi_levels: np.ndarray
    iterations: int
    converged: bool


class Koehler1986ReferenceState:
    """Fixed-isentrope, surface-aware Koehler (1986) reference-state solver."""

    def __init__(
        self,
        constants: MarsConstants = MARS,
        *,
        theta_levels: xr.DataArray | Sequence[float] | None = None,
        theta_increment: float | None = None,
        surface_pressure_policy: str = "raise",
        interpolation_space: str = "exner",
        monotonic_policy: str = "repair",
        max_iterations: int = 50,
        pressure_tolerance: float = 1.0e-3,
        iteration_tolerance: float = 1.0e-3,
        solver_strategy: str = "koehler_iteration",
    ) -> None:
        self.constants = constants
        self.theta_levels = theta_levels
        self.theta_increment = theta_increment
        self.surface_pressure_policy = surface_pressure_policy
        self.interpolation_space = interpolation_space
        self.monotonic_policy = monotonic_policy
        self.max_iterations = max_iterations
        self.pressure_tolerance = pressure_tolerance
        self.iteration_tolerance = iteration_tolerance
        self.solver_strategy = solver_strategy
        self._last_observed_state: xr.Dataset | None = None
        self._last_geometry_state: _KoehlerGeometryState | None = None

    def solve(
        self,
        potential_temperature: xr.DataArray,
        pressure: xr.DataArray,
        ps: xr.DataArray,
        phis: xr.DataArray | None = None,
        *,
        surface_potential_temperature: xr.DataArray | None = None,
        surface_temperature: xr.DataArray | None = None,
        theta_levels: xr.DataArray | Sequence[float] | None = None,
        theta_increment: float | None = None,
        level_bounds: xr.DataArray | None = None,
    ) -> ReferenceStateSolution:
        self._last_observed_state = None
        self._last_geometry_state = None

        full_context = _prepare_koehler1986_observed_context(
            potential_temperature,
            pressure,
            ps,
            surface_potential_temperature=surface_potential_temperature,
            surface_temperature=surface_temperature,
            theta_levels=self.theta_levels if theta_levels is None else theta_levels,
            theta_increment=self.theta_increment if theta_increment is None else theta_increment,
            level_bounds=level_bounds,
            constants=self.constants,
            surface_pressure_policy=self.surface_pressure_policy,
            pressure_tolerance=self.pressure_tolerance,
            interpolation_space=self.interpolation_space,
            monotonic_policy=self.monotonic_policy,
        )
        self._last_observed_state = full_context.observed_state

        if phis is None:
            phis_surface = xr.zeros_like(full_context.surface_pressure, dtype=float)
            phis_surface.name = "phis"
            phis_surface.attrs["units"] = "m2 s-2"
        else:
            phis_surface = broadcast_surface_field(phis, full_context.surface_pressure, "phis")

        full_family = _solve_reference_family_koehler1986(
            "full",
            full_context,
            phis_surface,
            constants=self.constants,
            max_iterations=self.max_iterations,
            pressure_tolerance=self.pressure_tolerance,
            iteration_tolerance=self.iteration_tolerance,
            solver_strategy=self.solver_strategy,
        )

        representative_theta = weighted_representative_zonal_mean(
            full_context.theta,
            full_context.measure.cell_fraction,
        ).broadcast_like(full_context.theta)
        representative_pressure = weighted_representative_zonal_mean(
            full_context.pressure,
            full_context.measure.cell_fraction,
        ).broadcast_like(full_context.pressure)
        surface_theta_zonal = _broadcast_surface_zonal(
            _surface_zonal_mean_geometric(full_context.surface_potential_temperature),
            full_context.theta.coords["longitude"],
            name="surface_potential_temperature_zonal",
        )

        zonal_context = _prepare_koehler1986_observed_context(
            representative_theta,
            representative_pressure,
            ps,
            surface_potential_temperature=surface_theta_zonal,
            theta_levels=full_context.theta_levels,
            level_bounds=level_bounds,
            constants=self.constants,
            surface_pressure_policy=self.surface_pressure_policy,
            pressure_tolerance=self.pressure_tolerance,
            interpolation_space=self.interpolation_space,
            monotonic_policy=self.monotonic_policy,
        )
        zonal_family = _solve_reference_family_koehler1986(
            "zonal",
            zonal_context,
            phis_surface,
            constants=self.constants,
            max_iterations=self.max_iterations,
            pressure_tolerance=self.pressure_tolerance,
            iteration_tolerance=self.iteration_tolerance,
            solver_strategy=self.solver_strategy,
        )

        self._last_geometry_state = _KoehlerGeometryState(
            phis_surface=phis_surface,
            representative_theta=representative_theta,
            representative_pressure=representative_pressure,
            surface_potential_temperature_zonal=surface_theta_zonal,
            full_family=full_family,
            zonal_family=zonal_family,
        )
        return _build_reference_state_solution(
            full_family,
            zonal_family,
            constants=self.constants,
        )


def _validate_solver_strategy(solver_strategy: str) -> str:
    normalized = str(solver_strategy).strip().lower()
    if normalized not in {"koehler_iteration", "root"}:
        raise ValueError("'solver_strategy' must be either 'koehler_iteration' or 'root'.")
    return normalized


def _surface_zonal_mean_geometric(field: xr.DataArray) -> xr.DataArray:
    weights = longitude_weights(field.coords["longitude"], normalize=True)
    return field.weighted(weights).sum(dim="longitude")


def _broadcast_surface_zonal(field_zonal: xr.DataArray, longitude: xr.DataArray, *, name: str) -> xr.DataArray:
    field = field_zonal.expand_dims(longitude=longitude).transpose("time", "latitude", "longitude")
    field.name = name
    field.attrs.update(field_zonal.attrs)
    return field


def _method_name_for_strategy(solver_strategy: str) -> str:
    if solver_strategy == "root":
        return "koehler1986_isentropic_surface_root"
    return "koehler1986_isentropic_surface_iteration"


def _initial_log_delta_exner(
    mean_pressure_on_theta: np.ndarray,
    *,
    constants: MarsConstants = MARS,
) -> tuple[np.ndarray, float]:
    mean_pressure_on_theta = np.asarray(mean_pressure_on_theta, dtype=float)
    if mean_pressure_on_theta.ndim != 1 or mean_pressure_on_theta.size < 2:
        raise ValueError("Koehler pressure profiles require at least two fixed isentropic levels.")
    if np.any(~np.isfinite(mean_pressure_on_theta)) or np.any(mean_pressure_on_theta <= 0.0):
        raise ValueError("Observed mean pressure on theta must remain finite and strictly positive.")

    min_delta_exner = 1.0e-10
    exner = np.maximum(_pressure_to_exner(mean_pressure_on_theta, constants), min_delta_exner)
    top_exner = float(max(exner[-1], min_delta_exner))
    delta_exner = np.maximum(exner[:-1] - exner[1:], min_delta_exner)
    log_delta_exner = np.log(delta_exner)
    return log_delta_exner, float(_exner_to_pressure(top_exner, constants))


def _pressure_profile_from_log_delta_exner(
    log_delta_exner: np.ndarray,
    top_pressure: float,
    *,
    constants: MarsConstants = MARS,
) -> np.ndarray:
    delta_exner = np.exp(np.asarray(log_delta_exner, dtype=float))
    if delta_exner.ndim != 1:
        raise ValueError("'log_delta_exner' must be one-dimensional.")
    exner = np.empty(delta_exner.size + 1, dtype=float)
    exner[-1] = float(_pressure_to_exner(top_pressure, constants))
    for index in range(delta_exner.size - 1, -1, -1):
        exner[index] = exner[index + 1] + delta_exner[index]
    return np.asarray(_exner_to_pressure(exner, constants), dtype=float)


def _evaluate_reference_profile_1d(
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
    *,
    surface_theta: np.ndarray,
    phis: np.ndarray,
    area: np.ndarray,
    layer_mass_observed: np.ndarray,
    mean_pressure_observed: np.ndarray,
    constants: MarsConstants = MARS,
) -> dict[str, np.ndarray | float]:
    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    surface_theta = np.asarray(surface_theta, dtype=float)
    phis = np.asarray(phis, dtype=float)
    area = np.asarray(area, dtype=float)
    layer_mass_observed = np.asarray(layer_mass_observed, dtype=float)
    mean_pressure_observed = np.asarray(mean_pressure_observed, dtype=float)

    theta_anchor = float(np.nanmin(surface_theta))
    phi_levels, _, _, _ = _reference_geopotential_with_anchor_details(
        theta_levels,
        pi_levels,
        theta_anchor=theta_anchor,
        phi_anchor=0.0,
        constants=constants,
    )
    reference = _reference_mass_from_profile_numpy(
        theta_levels,
        pi_levels,
        phi_levels,
        phis,
        area,
        constants=constants,
    )
    total_area = float(np.sum(area))
    pressure_thickness_observed = layer_mass_observed * constants.g / total_area
    pressure_thickness_reference = np.asarray(reference["layer_mass"], dtype=float) * constants.g / total_area
    mass_residual = np.asarray(reference["layer_mass"], dtype=float) - layer_mass_observed
    pressure_residual = pressure_thickness_reference - pressure_thickness_observed
    mean_pressure_reference = np.asarray(reference["mean_pressure_on_theta"], dtype=float)
    mean_pressure_residual = mean_pressure_reference - mean_pressure_observed
    return {
        "phi_levels": phi_levels,
        "pi_s": reference["pi_s"],
        "theta_s_ref": reference["theta_s_ref"],
        "surface_layer_index": reference["surface_layer_index"],
        "interface_pressure": reference["interface_pressure"],
        "layer_mass_reference": reference["layer_mass"],
        "layer_pressure_thickness_reference": reference["layer_pressure_thickness"],
        "mean_pressure_reference": mean_pressure_reference,
        "reference_surface_pressure": reference["reference_surface_pressure"],
        "reference_bottom_pressure": reference["deepest_surface_pressure"],
        "pressure_residual": pressure_residual,
        "mass_residual": mass_residual,
        "mean_pressure_residual": mean_pressure_residual,
    }


def koehler_pressure_update(
    delta_exner_old: np.ndarray,
    pressure_thickness_observed: np.ndarray,
    pressure_thickness_reference: np.ndarray,
    *,
    damping: float = 0.5,
) -> np.ndarray:
    delta_exner_old = np.asarray(delta_exner_old, dtype=float)
    pressure_thickness_observed = np.asarray(pressure_thickness_observed, dtype=float)
    pressure_thickness_reference = np.asarray(pressure_thickness_reference, dtype=float)
    ratio = np.where(
        pressure_thickness_reference > 0.0,
        pressure_thickness_observed / pressure_thickness_reference,
        1.0,
    )
    ratio = np.clip(ratio, 0.25, 4.0)
    return np.maximum(delta_exner_old * np.power(ratio, damping), 1.0e-12)


def solve_reference_pressure_profile_root(
    theta_levels: np.ndarray,
    pbar_observed: np.ndarray,
    layer_mass_observed: np.ndarray,
    surface_theta: np.ndarray,
    phis: np.ndarray,
    area: np.ndarray,
    *,
    constants: MarsConstants = MARS,
    max_iterations: int = 50,
    pressure_tolerance: float = 1.0e-3,
    initial_log_delta_exner: np.ndarray | None = None,
) -> _KoehlerSolvedTimeSlice:
    theta_levels = np.asarray(theta_levels, dtype=float)
    pbar_observed = np.asarray(pbar_observed, dtype=float)
    layer_mass_observed = np.asarray(layer_mass_observed, dtype=float)
    if initial_log_delta_exner is None:
        log_delta_exner, top_pressure = _initial_log_delta_exner(pbar_observed, constants=constants)
    else:
        log_delta_exner = np.asarray(initial_log_delta_exner, dtype=float)
        _, top_pressure = _initial_log_delta_exner(pbar_observed, constants=constants)

    def residual(log_delta: np.ndarray) -> np.ndarray:
        pi_levels = _pressure_profile_from_log_delta_exner(log_delta, top_pressure, constants=constants)
        try:
            evaluation = _evaluate_reference_profile_1d(
                theta_levels,
                pi_levels,
                surface_theta=surface_theta,
                phis=phis,
                area=area,
                layer_mass_observed=layer_mass_observed,
                mean_pressure_observed=pbar_observed,
                constants=constants,
            )
            return np.asarray(evaluation["pressure_residual"], dtype=float)
        except ValueError:
            return np.full(theta_levels.size - 1, 1.0e9, dtype=float)

    result = optimize.root(
        residual,
        log_delta_exner,
        method="hybr",
        options={"maxfev": max(50, max_iterations * 40)},
    )
    log_delta_final = result.x if result.x is not None else log_delta_exner
    pi_levels = _pressure_profile_from_log_delta_exner(log_delta_final, top_pressure, constants=constants)
    evaluation = _evaluate_reference_profile_1d(
        theta_levels,
        pi_levels,
        surface_theta=surface_theta,
        phis=phis,
        area=area,
        layer_mass_observed=layer_mass_observed,
        mean_pressure_observed=pbar_observed,
        constants=constants,
    )
    converged = bool(result.success) and np.nanmax(np.abs(np.asarray(evaluation["pressure_residual"], dtype=float))) <= float(
        pressure_tolerance
    )
    return _KoehlerSolvedTimeSlice(
        pi_levels=pi_levels,
        iterations=int(getattr(result, "nfev", max_iterations)),
        converged=converged,
    )


def solve_reference_pressure_profile_koehler_iteration(
    theta_levels: np.ndarray,
    pbar_observed: np.ndarray,
    layer_mass_observed: np.ndarray,
    surface_theta: np.ndarray,
    phis: np.ndarray,
    area: np.ndarray,
    *,
    constants: MarsConstants = MARS,
    max_iterations: int = 50,
    pressure_tolerance: float = 1.0e-3,
    iteration_tolerance: float = 1.0e-3,
) -> _KoehlerSolvedTimeSlice:
    theta_levels = np.asarray(theta_levels, dtype=float)
    pbar_observed = np.asarray(pbar_observed, dtype=float)
    layer_mass_observed = np.asarray(layer_mass_observed, dtype=float)
    log_delta_exner, top_pressure = _initial_log_delta_exner(pbar_observed, constants=constants)
    previous_pi_levels = None
    total_area = float(np.sum(area))
    pressure_thickness_observed = layer_mass_observed * constants.g / total_area

    for iteration in range(1, max_iterations + 1):
        pi_levels = _pressure_profile_from_log_delta_exner(log_delta_exner, top_pressure, constants=constants)
        try:
            evaluation = _evaluate_reference_profile_1d(
                theta_levels,
                pi_levels,
                surface_theta=surface_theta,
                phis=phis,
                area=area,
                layer_mass_observed=layer_mass_observed,
                mean_pressure_observed=pbar_observed,
                constants=constants,
            )
        except ValueError:
            log_delta_exner = log_delta_exner + np.log(1.25)
            previous_pi_levels = pi_levels
            continue

        pressure_residual = np.asarray(evaluation["pressure_residual"], dtype=float)
        max_pressure_residual = float(np.nanmax(np.abs(pressure_residual)))
        max_pressure_change = np.inf
        if previous_pi_levels is not None:
            max_pressure_change = float(np.nanmax(np.abs(pi_levels - previous_pi_levels)))
        if max_pressure_residual <= float(pressure_tolerance) and max_pressure_change <= float(iteration_tolerance):
            return _KoehlerSolvedTimeSlice(
                pi_levels=pi_levels,
                iterations=iteration,
                converged=True,
            )

        pressure_thickness_reference = np.asarray(evaluation["layer_pressure_thickness_reference"], dtype=float)
        pressure_thickness_reference = np.sum(pressure_thickness_reference * area[None, :, :] / total_area, axis=(1, 2))
        delta_exner_old = np.exp(log_delta_exner)
        delta_exner_new = koehler_pressure_update(
            delta_exner_old,
            pressure_thickness_observed,
            pressure_thickness_reference,
        )
        previous_pi_levels = pi_levels
        log_delta_exner = np.log(np.maximum(delta_exner_new, 1.0e-12))

    root_solution = solve_reference_pressure_profile_root(
        theta_levels,
        pbar_observed,
        layer_mass_observed,
        surface_theta,
        phis,
        area,
        constants=constants,
        max_iterations=max_iterations,
        pressure_tolerance=pressure_tolerance,
        initial_log_delta_exner=log_delta_exner,
    )
    return _KoehlerSolvedTimeSlice(
        pi_levels=root_solution.pi_levels,
        iterations=max_iterations + root_solution.iterations,
        converged=root_solution.converged,
    )


def _solve_reference_family_koehler1986(
    family_name: str,
    context: _KoehlerObservedStateContext,
    phis_surface: xr.DataArray,
    *,
    constants: MarsConstants = MARS,
    max_iterations: int,
    pressure_tolerance: float,
    iteration_tolerance: float,
    solver_strategy: str,
) -> _KoehlerSolvedFamily:
    solver_strategy = _validate_solver_strategy(solver_strategy)
    top_above_fraction = context.observed_state["above_model_top_area_fraction"].isel({ISENTROPIC_DIM: -1})
    if np.any(np.asarray(top_above_fraction.values, dtype=float) < 1.0 - 1.0e-12):
        raise ValueError(
            "Koehler1986ReferenceState requires the highest fixed isentropic level to lie "
            "above the resolved model top in every valid column; provide a larger explicit "
            "'theta_levels' top value or a finer/higher 'theta_increment'."
        )
    theta_levels = np.asarray(context.theta_levels.values, dtype=float)
    cell_area = np.asarray(context.integrator.cell_area.values, dtype=float)
    solved_pi_levels = np.empty((context.theta.sizes["time"], theta_levels.size), dtype=float)
    iterations = np.empty(context.theta.sizes["time"], dtype=np.int64)
    converged = np.empty(context.theta.sizes["time"], dtype=bool)

    for time_index in range(context.theta.sizes["time"]):
        pbar_observed = np.asarray(
            context.observed_state["mean_pressure_on_theta"].isel(time=time_index).values,
            dtype=float,
        )
        layer_mass_observed = np.asarray(
            context.observed_state["layer_mass"].isel(time=time_index).values,
            dtype=float,
        )
        surface_theta = np.asarray(
            context.surface_potential_temperature.isel(time=time_index).values,
            dtype=float,
        )
        phis_t = np.asarray(phis_surface.isel(time=time_index).values, dtype=float)
        if solver_strategy == "root":
            solved = solve_reference_pressure_profile_root(
                theta_levels,
                pbar_observed,
                layer_mass_observed,
                surface_theta,
                phis_t,
                cell_area,
                constants=constants,
                max_iterations=max_iterations,
                pressure_tolerance=pressure_tolerance,
            )
        else:
            solved = solve_reference_pressure_profile_koehler_iteration(
                theta_levels,
                pbar_observed,
                layer_mass_observed,
                surface_theta,
                phis_t,
                cell_area,
                constants=constants,
                max_iterations=max_iterations,
                pressure_tolerance=pressure_tolerance,
                iteration_tolerance=iteration_tolerance,
            )
        solved_pi_levels[time_index] = solved.pi_levels
        iterations[time_index] = solved.iterations
        converged[time_index] = solved.converged

    pi_levels = xr.DataArray(
        solved_pi_levels,
        dims=("time", ISENTROPIC_DIM),
        coords={
            "time": context.theta.coords["time"].values,
            ISENTROPIC_DIM: theta_levels,
        },
        name=f"{family_name}_pi_levels",
        attrs={
            "units": "Pa",
            "reference_pressure_sampling": "fixed_isentropic_level_pressure",
            "reference_curve_interpolation_space": "exner",
        },
    )
    geometry = _build_koehler_geometry_family_xr(
        family_name,
        context.theta_levels,
        pi_levels,
        context.surface_potential_temperature,
        phis_surface,
        area=context.integrator.cell_area,
        constants=constants,
    )
    mean_pressure_observed = context.observed_state["mean_pressure_on_theta"].rename(
        f"{family_name}_mean_pressure_on_theta_observed"
    )
    layer_mass_observed = context.observed_state["layer_mass"].rename(f"{family_name}_layer_mass_observed")
    layer_mass_reference = geometry.reference_mass.layer_mass.rename(f"{family_name}_layer_mass_reference")
    mass_residual = (layer_mass_reference - layer_mass_observed).rename(f"{family_name}_mass_residual")
    return _KoehlerSolvedFamily(
        family_name=family_name,
        context=context,
        geometry=geometry,
        mean_pressure_observed=mean_pressure_observed,
        mean_pressure_reference=geometry.reference_mass.mean_pressure_on_theta.rename(
            f"{family_name}_mean_pressure_on_theta_reference"
        ),
        layer_mass_observed=layer_mass_observed,
        layer_mass_reference=layer_mass_reference,
        mass_residual=mass_residual,
        iterations=xr.DataArray(
            iterations,
            dims=("time",),
            coords={"time": context.theta.coords["time"].values},
            name=f"{family_name}_reference_state_iterations",
        ),
        converged=xr.DataArray(
            converged,
            dims=("time",),
            coords={"time": context.theta.coords["time"].values},
            name=f"{family_name}_reference_state_converged",
        ),
        solver_strategy=solver_strategy,
    )


def _pad_layer_mass_to_reference_samples(
    layer_mass: xr.DataArray,
    theta_levels: xr.DataArray,
) -> xr.DataArray:
    theta_values = np.asarray(theta_levels.values, dtype=float)
    layer_values = np.asarray(layer_mass.values, dtype=float)
    padded = np.zeros((layer_values.shape[0], theta_values.size), dtype=float)
    padded[:, : layer_values.shape[1]] = layer_values
    return xr.DataArray(
        padded,
        dims=("time", REFERENCE_SAMPLE_DIM),
        coords={
            "time": layer_mass.coords["time"].values,
            REFERENCE_SAMPLE_DIM: np.arange(theta_values.size),
        },
        name="isentropic_mass",
        attrs={"units": "kg"},
    )


def _build_reference_state_solution(
    full_family: _KoehlerSolvedFamily,
    zonal_family: _KoehlerSolvedFamily,
    *,
    constants: MarsConstants = MARS,
) -> ReferenceStateSolution:
    context = full_family.context
    measure = context.measure
    time_values = context.theta.coords["time"].values
    theta_values = np.asarray(context.theta_levels.values, dtype=float)
    sample_coords = {
        "time": time_values,
        REFERENCE_SAMPLE_DIM: np.arange(theta_values.size),
    }
    interface_coords = {
        "time": time_values,
        REFERENCE_INTERFACE_DIM: np.arange(theta_values.size),
    }
    surface_coords = {
        "time": time_values,
        "latitude": context.theta.coords["latitude"].values,
        "longitude": context.theta.coords["longitude"].values,
    }
    theta_reference_values = np.broadcast_to(theta_values[None, :], (time_values.size, theta_values.size)).copy()

    def _annotate_domain(field: xr.DataArray) -> xr.DataArray:
        return measure.annotate_domain_metadata(field)

    method = _method_name_for_strategy(full_family.solver_strategy)
    theta_reference = xr.DataArray(
        theta_reference_values,
        dims=("time", REFERENCE_SAMPLE_DIM),
        coords=sample_coords,
        name="theta_reference",
        attrs={
            "units": "K",
            "reference_coordinate_semantics": "fixed_isentropic_levels",
            "meaning": "fixed isentropic levels",
            "reference_curve_interpolation_space": "exner",
        },
    )
    theta_reference_zonal = xr.DataArray(
        theta_reference_values.copy(),
        dims=("time", REFERENCE_SAMPLE_DIM),
        coords=sample_coords,
        name="_theta_reference_zonal",
        attrs={
            "units": "K",
            "reference_coordinate_semantics": "fixed_isentropic_levels",
            "reference_curve_interpolation_space": "exner",
        },
    )

    return ReferenceStateSolution(
        theta_reference=theta_reference,
        pi_reference=xr.DataArray(
            np.asarray(full_family.geometry.profile.pi_levels.values, dtype=float),
            dims=("time", REFERENCE_SAMPLE_DIM),
            coords=sample_coords,
            name="pi_reference",
            attrs={
                "units": "Pa",
                "reference_pressure_sampling": "fixed_isentropic_level_pressure",
                "meaning": "reference pressure on fixed isentropic levels",
                "reference_curve_interpolation_space": "exner",
            },
        ),
        mass_reference=_pad_layer_mass_to_reference_samples(
            full_family.layer_mass_reference,
            full_family.context.theta_levels,
        ),
        reference_interface_pressure=xr.DataArray(
            np.asarray(full_family.geometry.profile.pi_levels.values, dtype=float),
            dims=("time", REFERENCE_INTERFACE_DIM),
            coords=interface_coords,
            name="reference_interface_pressure",
            attrs={"units": "Pa", "long_name": "reference-state pressure on fixed isentropic levels"},
        ),
        reference_interface_geopotential=xr.DataArray(
            np.asarray(full_family.geometry.profile.phi_levels.values, dtype=float),
            dims=("time", REFERENCE_INTERFACE_DIM),
            coords=interface_coords,
            name="reference_interface_geopotential",
            attrs={"units": "m2 s-2", "long_name": "reference-state geopotential on fixed isentropic levels"},
        ),
        total_mass=_annotate_domain(
            xr.DataArray(
                context.measure.parcel_mass.sum(dim=("level", "latitude", "longitude")).values,
                dims=("time",),
                coords={"time": time_values},
                name="total_mass",
                attrs={"units": "kg"},
            )
        ),
        reference_surface_pressure=_annotate_domain(
            full_family.geometry.surface.reference_surface_pressure.rename("reference_surface_pressure")
        ),
        reference_bottom_pressure=_annotate_domain(
            full_family.geometry.surface.deepest_surface_pressure.rename("reference_bottom_pressure")
        ),
        reference_top_pressure=full_family.geometry.profile.profile_top_pressure.rename("reference_top_pressure"),
        ps_effective=_annotate_domain(context.ps_effective.rename("ps_effective")),
        pi_s=_annotate_domain(
            xr.DataArray(
                np.asarray(full_family.geometry.surface.pi_s.values, dtype=float),
                dims=("time", "latitude", "longitude"),
                coords=surface_coords,
                name="pi_s",
                attrs=full_family.geometry.surface.pi_s.attrs,
            )
        ),
        pi_sZ=_annotate_domain(
            xr.DataArray(
                np.asarray(zonal_family.geometry.surface.pi_s.values, dtype=float),
                dims=("time", "latitude", "longitude"),
                coords=surface_coords,
                name="pi_sZ",
                attrs=zonal_family.geometry.surface.pi_s.attrs,
            )
        ),
        iterations=full_family.iterations.rename("reference_state_iterations"),
        converged=full_family.converged.rename("reference_state_converged"),
        iterations_zonal=zonal_family.iterations.rename("reference_state_iterations_zonal"),
        converged_zonal=zonal_family.converged.rename("reference_state_converged_zonal"),
        method=method,
        constants=constants,
        _theta_reference_zonal=theta_reference_zonal,
        _pi_reference_zonal=xr.DataArray(
            np.asarray(zonal_family.geometry.profile.pi_levels.values, dtype=float),
            dims=("time", REFERENCE_SAMPLE_DIM),
            coords=sample_coords,
            name="_pi_reference_zonal",
            attrs={
                "units": "Pa",
                "reference_pressure_sampling": "fixed_isentropic_level_pressure",
                "reference_curve_interpolation_space": "exner",
            },
        ),
        _reference_bottom_pressure_zonal=_annotate_domain(
            zonal_family.geometry.surface.deepest_surface_pressure.rename("_reference_bottom_pressure_zonal")
        ),
    )


__all__ = ["Koehler1986ReferenceState"]
