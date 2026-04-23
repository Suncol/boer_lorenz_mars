from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.integrals import build_mass_integrator, pressure_level_edges
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.common.zonal_ops import weighted_representative_zonal_mean
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import (
    FiniteVolumeReferenceState,
    Koehler1986ReferenceState,
    potential_temperature,
)

from .helpers import (
    full_field,
    make_coords,
    pressure_field,
    reference_case_surface_geopotential_values,
    surface_geopotential,
    surface_pressure,
    surface_pressure_policy_for_case,
    temperature_from_theta_values,
)

ISENTROPIC_DIM = "isentropic_level"


def surface_theta_field(ps: xr.DataArray, value: float) -> xr.DataArray:
    field = xr.full_like(ps, float(value), dtype=float)
    field.name = "surface_potential_temperature"
    field.attrs["units"] = "K"
    return field


def _analytic_exner(pressure: np.ndarray | float) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=float)
    return np.power(pressure / MARS.p00, MARS.kappa)


def _analytic_pressure(exner: np.ndarray | float) -> np.ndarray:
    exner = np.asarray(exner, dtype=float)
    return MARS.p00 * np.power(np.maximum(exner, 0.0), 1.0 / MARS.kappa)


def _analytic_phi_levels(theta_levels: np.ndarray, pi_levels: np.ndarray) -> np.ndarray:
    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    exner_levels = _analytic_exner(pi_levels)
    delta_phi = MARS.cp * 0.5 * (theta_levels[:-1] + theta_levels[1:]) * (exner_levels[:-1] - exner_levels[1:])
    phi_levels = np.empty(theta_levels.size, dtype=float)
    phi_levels[0] = 0.0
    phi_levels[1:] = np.cumsum(delta_phi, dtype=float)
    return phi_levels


def _analytic_layer_index_from_theta(theta_target: np.ndarray, theta_levels: np.ndarray) -> np.ndarray:
    theta_target = np.asarray(theta_target, dtype=float)
    theta_levels = np.asarray(theta_levels, dtype=float)
    layer_index = np.searchsorted(theta_levels, theta_target, side="right") - 1
    return np.clip(layer_index, 0, theta_levels.size - 2)


def _analytic_surface_from_theta(
    surface_theta: np.ndarray,
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
    phi_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_theta = np.asarray(surface_theta, dtype=float)
    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    phi_levels = np.asarray(phi_levels, dtype=float)
    exner_levels = _analytic_exner(pi_levels)
    layer_index = _analytic_layer_index_from_theta(surface_theta, theta_levels)

    theta_lower = theta_levels[layer_index]
    theta_upper = theta_levels[layer_index + 1]
    phi_lower = phi_levels[layer_index]
    phi_upper = phi_levels[layer_index + 1]
    exner_lower = exner_levels[layer_index]
    exner_upper = exner_levels[layer_index + 1]

    theta_fraction = np.clip((surface_theta - theta_lower) / (theta_upper - theta_lower), 0.0, 1.0)
    exner_surface = exner_lower + theta_fraction * (exner_upper - exner_lower)
    phi_fraction = np.clip(
        (surface_theta**2 - theta_lower**2) / (theta_upper**2 - theta_lower**2),
        0.0,
        1.0,
    )
    phi_surface = phi_lower + phi_fraction * (phi_upper - phi_lower)
    return _analytic_pressure(exner_surface), phi_surface, layer_index.astype(np.int64)


def _analytic_theta_from_pressure(
    pressure_target: np.ndarray,
    theta_levels: np.ndarray,
    pi_levels: np.ndarray,
) -> np.ndarray:
    pressure_target = np.asarray(pressure_target, dtype=float)
    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    exner_levels = _analytic_exner(pi_levels)
    target_exner = _analytic_exner(pressure_target)

    layer_index = np.searchsorted(-pi_levels, -pressure_target, side="right") - 1
    layer_index = np.clip(layer_index, 0, theta_levels.size - 2)

    theta_lower = theta_levels[layer_index]
    theta_upper = theta_levels[layer_index + 1]
    exner_lower = exner_levels[layer_index]
    exner_upper = exner_levels[layer_index + 1]
    theta_fraction = np.clip(
        (target_exner - exner_lower) / (exner_upper - exner_lower),
        0.0,
        1.0,
    )
    return theta_lower + theta_fraction * (theta_upper - theta_lower)


def _default_k86_theta_levels(theta_values, surface_theta_value: float) -> np.ndarray:
    theta_values = np.asarray(theta_values, dtype=float)
    finite_theta = theta_values[np.isfinite(theta_values)]
    if finite_theta.size == 0:
        raise ValueError("Cannot construct K86 theta levels without finite theta values.")
    surface_theta_value = float(surface_theta_value)
    rounded_surface = 10.0 * np.floor(surface_theta_value / 10.0)
    rounded_top = 10.0 * np.ceil(float(np.nanmax(finite_theta)) / 10.0) + 20.0
    regular = np.arange(rounded_surface, rounded_top + 0.5, 10.0, dtype=float)
    exact_values = np.unique(np.round(finite_theta, decimals=10))
    levels = np.unique(np.concatenate(([surface_theta_value], regular, exact_values, [rounded_top])))
    levels.sort()
    return levels


def solver_for_case(
    ps,
    level,
    *,
    solver_kind: str = "fv",
    level_bounds=None,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
    theta_levels=None,
    theta_increment: float | None = None,
    solver_strategy: str = "koehler_iteration",
):
    policy = surface_pressure_policy_for_case(
        ps,
        level,
        level_bounds=level_bounds,
        pressure_tolerance=pressure_tolerance,
    )
    solver_kind = str(solver_kind).strip().lower()
    if solver_kind == "fv":
        solver = FiniteVolumeReferenceState(
            pressure_tolerance=pressure_tolerance,
            max_iterations=max_iterations,
            surface_pressure_policy=policy,
        )
    elif solver_kind == "k86":
        solver = Koehler1986ReferenceState(
            theta_levels=theta_levels,
            theta_increment=theta_increment,
            pressure_tolerance=pressure_tolerance,
            max_iterations=max_iterations,
            surface_pressure_policy=policy,
            solver_strategy=solver_strategy,
        )
    else:
        raise ValueError("solver_kind must be either 'fv' or 'k86'.")
    return policy, solver


def build_reference_case_bundle(
    *,
    solver_kind: str,
    case_kind: str,
    include_reference: bool = True,
    ntime: int = 1,
    level_values=None,
    level_bounds=None,
    ps_value: float | None = None,
    phis_value: float | None = None,
    theta_profile=None,
    time_theta_offsets=None,
    surface_theta_value: float | None = None,
    theta_levels=None,
    theta_increment: float | None = None,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
    solver_strategy: str = "koehler_iteration",
):
    solver_kind = str(solver_kind).strip().lower()
    case_kind = str(case_kind).strip().lower()
    if case_kind == "flat":
        if level_values is None:
            level_values = (900.0, 700.0, 500.0)
        if ps_value is None:
            ps_value = 950.0 if solver_kind == "k86" else 850.0
        if phis_value is None:
            phis_value = 0.0 if solver_kind == "k86" else 2000.0
        if theta_profile is None:
            theta_profile = (190.0, 210.0, 230.0) if solver_kind == "k86" else (210.0, 230.0, 250.0)
        return _build_flat_reference_case_bundle(
            solver_kind=solver_kind,
            include_reference=include_reference,
            ntime=ntime,
            level_values=level_values,
            level_bounds=level_bounds,
            ps_value=float(ps_value),
            phis_value=float(phis_value),
            theta_profile=theta_profile,
            time_theta_offsets=time_theta_offsets,
            surface_theta_value=surface_theta_value,
            theta_levels=theta_levels,
            theta_increment=theta_increment,
            pressure_tolerance=pressure_tolerance,
            max_iterations=max_iterations,
            solver_strategy=solver_strategy,
        )
    if case_kind == "asymmetric":
        return _build_asymmetric_reference_case_bundle(
            solver_kind=solver_kind,
            include_reference=include_reference,
            ntime=ntime,
            level_values=level_values,
            level_bounds=level_bounds,
            surface_theta_value=surface_theta_value,
            theta_levels=theta_levels,
            theta_increment=theta_increment,
            pressure_tolerance=pressure_tolerance,
            max_iterations=max_iterations,
            solver_strategy=solver_strategy,
        )
    raise ValueError("case_kind must be either 'flat' or 'asymmetric'.")


def _build_flat_reference_case_bundle(
    *,
    solver_kind: str,
    include_reference: bool,
    ntime: int,
    level_values,
    level_bounds,
    ps_value: float,
    phis_value: float,
    theta_profile,
    time_theta_offsets,
    surface_theta_value: float | None,
    theta_levels,
    theta_increment: float | None,
    pressure_tolerance: float,
    max_iterations: int,
    solver_strategy: str,
):
    time, level, latitude, longitude = make_coords(ntime=ntime, level_values=level_values)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, ps_value)
    phis = surface_geopotential(time, latitude, longitude, phis_value)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)

    theta_profile_values = np.asarray(theta_profile, dtype=float)
    if theta_profile_values.shape != (level.size,):
        raise ValueError("theta_profile must have shape (level,).")
    if time_theta_offsets is None:
        time_offsets = np.zeros(time.size, dtype=float)
    else:
        time_offsets = np.asarray(time_theta_offsets, dtype=float)
        if time_offsets.ndim == 0:
            time_offsets = np.full(time.size, float(time_offsets), dtype=float)
        if time_offsets.shape != (time.size,):
            raise ValueError("time_theta_offsets must be scalar or have shape (time,).")

    theta_values = theta_profile_values[None, :, None, None] + time_offsets[:, None, None, None]
    temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    pt = potential_temperature(temperature, pressure)
    if solver_kind == "k86":
        if surface_theta_value is None:
            surface_theta_value = float(np.nanmin(theta_values)) - 10.0
        surface_theta = surface_theta_field(ps, surface_theta_value)
        if theta_levels is None and theta_increment is None:
            theta_levels = _default_k86_theta_levels(theta_values, surface_theta_value)
    else:
        surface_theta = None

    policy, solver = solver_for_case(
        ps,
        level,
        solver_kind=solver_kind,
        level_bounds=level_bounds,
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        theta_levels=theta_levels,
        theta_increment=theta_increment,
        solver_strategy=solver_strategy,
    )
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=level_bounds,
        pressure_tolerance=solver.pressure_tolerance,
        surface_pressure_policy=policy,
    )
    solve_kwargs = {"level_bounds": level_bounds}
    if surface_theta is not None:
        solve_kwargs["surface_potential_temperature"] = surface_theta
    solution = None
    if include_reference:
        solution = solver.solve(pt, pressure, ps, phis=phis, **solve_kwargs)

    representative_theta = weighted_representative_zonal_mean(pt, measure.cell_fraction)
    representative_pressure = weighted_representative_zonal_mean(pressure, measure.cell_fraction)

    bundle = {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "temperature": temperature,
        "potential_temperature": pt,
        "pt": pt,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "measure": measure,
        "policy": policy,
        "solver": solver,
        "solution": solution,
        "solve_kwargs": solve_kwargs,
        "representative_theta": representative_theta,
        "representative_pressure": representative_pressure,
    }
    if surface_theta is not None:
        bundle["surface_potential_temperature"] = surface_theta
        bundle["theta_levels"] = np.asarray(theta_levels, dtype=float) if theta_levels is not None else None
    return bundle


def _build_asymmetric_reference_case_bundle(
    *,
    solver_kind: str,
    include_reference: bool,
    ntime: int,
    level_values,
    level_bounds,
    surface_theta_value: float | None,
    theta_levels,
    theta_increment: float | None,
    pressure_tolerance: float,
    max_iterations: int,
    solver_strategy: str,
):
    if solver_kind == "k86":
        if level_values is None:
            level_values = (900.0, 700.0, 500.0, 300.0)
        if surface_theta_value is None:
            surface_theta_value = 170.0
        if theta_levels is None and theta_increment is None:
            theta_levels = np.asarray([170.0, 190.0, 210.0, 230.0, 250.0, 270.0, 290.0], dtype=float)
    else:
        if level_values is None:
            level_values = (700.0, 500.0, 300.0)

    time, level, latitude, longitude = make_coords(ntime=ntime, level_values=level_values)
    pressure = pressure_field(time, level, latitude, longitude)
    if solver_kind == "k86":
        ps_2d = np.asarray(
            [
                [950.0, 900.0, 850.0, 800.0],
                [920.0, 870.0, 820.0, 770.0],
                [900.0, 850.0, 800.0, 750.0],
                [880.0, 830.0, 780.0, 730.0],
            ],
            dtype=float,
        )
    else:
        ps_2d = np.asarray(
            [
                [760.0, 700.0, 620.0, 540.0],
                [740.0, 680.0, 600.0, 520.0],
                [720.0, 660.0, 580.0, 500.0],
                [700.0, 640.0, 560.0, 480.0],
            ],
            dtype=float,
        )
    ps = surface_pressure(time, latitude, longitude, ps_2d)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(
            latitude,
            longitude,
            base=100.0,
            lat_range=240.0,
            lon_range=360.0,
        ),
    )
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)
    if solver_kind == "k86":
        theta_values = (
            np.linspace(190.0, 250.0, level.size, dtype=float)[None, :, None, None]
            + np.linspace(0.0, 8.0, latitude.size, dtype=float)[None, None, :, None]
            + np.linspace(0.0, 4.0, longitude.size, dtype=float)[None, None, None, :]
        )
        temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    else:
        temperature = full_field(
            time,
            level,
            latitude,
            longitude,
            np.linspace(
                170.0,
                255.0,
                time.size * level.size * latitude.size * longitude.size,
            ).reshape(time.size, level.size, latitude.size, longitude.size),
            name="temperature",
            units="K",
        )
    pt = potential_temperature(temperature, pressure)
    surface_theta = surface_theta_field(ps, surface_theta_value) if solver_kind == "k86" else None
    policy, solver = solver_for_case(
        ps,
        level,
        solver_kind=solver_kind,
        level_bounds=level_bounds,
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        theta_levels=theta_levels,
        theta_increment=theta_increment,
        solver_strategy=solver_strategy,
    )
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=level_bounds,
        pressure_tolerance=solver.pressure_tolerance,
        surface_pressure_policy=policy,
    )
    solve_kwargs = {"level_bounds": level_bounds}
    if surface_theta is not None:
        solve_kwargs["surface_potential_temperature"] = surface_theta
    solution = None
    if include_reference:
        solution = solver.solve(pt, pressure, ps, phis=phis, **solve_kwargs)

    representative_theta = weighted_representative_zonal_mean(pt, measure.cell_fraction)
    representative_pressure = weighted_representative_zonal_mean(pressure, measure.cell_fraction)

    bundle = {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "temperature": temperature,
        "potential_temperature": pt,
        "pt": pt,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "measure": measure,
        "policy": policy,
        "solver": solver,
        "solution": solution,
        "solve_kwargs": solve_kwargs,
        "representative_theta": representative_theta,
        "representative_pressure": representative_pressure,
    }
    if surface_theta is not None:
        bundle["surface_potential_temperature"] = surface_theta
        bundle["theta_levels"] = np.asarray(theta_levels, dtype=float) if theta_levels is not None else None
    return bundle


def build_flat_reference_case(
    *,
    solver_kind: str = "fv",
    ntime: int = 1,
    level_values=(900.0, 700.0, 500.0),
    level_bounds=None,
    ps_value: float = 850.0,
    phis_value: float = 2000.0,
    theta_profile=(210.0, 230.0, 250.0),
    time_theta_offsets=None,
    include_reference: bool = True,
    surface_theta_value: float | None = None,
    theta_levels=None,
    theta_increment: float | None = None,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
    solver_strategy: str = "koehler_iteration",
):
    return build_reference_case_bundle(
        solver_kind=solver_kind,
        case_kind="flat",
        include_reference=include_reference,
        ntime=ntime,
        level_values=level_values,
        level_bounds=level_bounds,
        ps_value=ps_value,
        phis_value=phis_value,
        theta_profile=theta_profile,
        time_theta_offsets=time_theta_offsets,
        surface_theta_value=surface_theta_value,
        theta_levels=theta_levels,
        theta_increment=theta_increment,
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        solver_strategy=solver_strategy,
    )


def build_asymmetric_reference_case(
    *,
    solver_kind: str = "fv",
    ntime: int = 2,
    level_values=None,
    level_bounds=None,
    include_reference: bool = True,
    surface_theta_value: float | None = None,
    theta_levels=None,
    theta_increment: float | None = None,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
    solver_strategy: str = "koehler_iteration",
):
    return build_reference_case_bundle(
        solver_kind=solver_kind,
        case_kind="asymmetric",
        include_reference=include_reference,
        ntime=ntime,
        level_values=level_values,
        level_bounds=level_bounds,
        surface_theta_value=surface_theta_value,
        theta_levels=theta_levels,
        theta_increment=theta_increment,
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        solver_strategy=solver_strategy,
    )


def build_asymmetric_exact_case(**kwargs):
    kwargs.setdefault("solver_kind", "fv")
    return build_asymmetric_reference_case(**kwargs)


def build_k86_exact_topographic_reference_case(
    *,
    ntime: int = 1,
    level_values=(820.0, 660.0, 520.0, 400.0, 290.0),
    level_bounds=None,
    theta_levels=(170.0, 190.0, 210.0, 230.0, 250.0, 270.0, 290.0),
    pi_levels=(1000.0, 820.0, 660.0, 520.0, 400.0, 290.0, 235.0),
    surface_theta_min: float = 170.0,
    surface_theta_max: float = 218.0,
    phis_offset: float = 750.0,
    include_reference: bool = True,
    pressure_tolerance: float = 1.0e-8,
    max_iterations: int = 120,
    solver_strategy: str = "root",
):
    time, level, latitude, longitude = make_coords(ntime=ntime, level_values=level_values)
    if level_bounds is None:
        edges = np.asarray([1000.0, 740.0, 590.0, 460.0, 345.0, 235.0], dtype=float)
        level_bounds = xr.DataArray(
            np.column_stack([edges[:-1], edges[1:]]),
            dims=("level", "bounds"),
            coords={"level": level.values, "bounds": [0, 1]},
            name="level_bounds",
            attrs={"units": "Pa"},
        )
    pressure = pressure_field(time, level, latitude, longitude)
    theta_levels = np.asarray(theta_levels, dtype=float)
    pi_levels = np.asarray(pi_levels, dtype=float)
    if theta_levels.ndim != 1 or pi_levels.ndim != 1 or theta_levels.size != pi_levels.size:
        raise ValueError("'theta_levels' and 'pi_levels' must be one-dimensional arrays with the same length.")
    if not np.all(np.diff(theta_levels) > 0.0):
        raise ValueError("'theta_levels' must be strictly increasing.")
    if not np.all(np.diff(pi_levels) < 0.0):
        raise ValueError("'pi_levels' must be strictly decreasing.")

    phi_levels = _analytic_phi_levels(theta_levels, pi_levels)

    lat_fraction = np.linspace(0.0, 1.0, latitude.size, dtype=float)[:, None]
    surface_theta_2d = surface_theta_min + (surface_theta_max - surface_theta_min) * lat_fraction
    surface_theta_2d = np.broadcast_to(surface_theta_2d, (latitude.size, longitude.size)).copy()
    exact_ps_2d, phis_relative_2d, surface_layer_index_2d = _analytic_surface_from_theta(
        surface_theta_2d,
        theta_levels,
        pi_levels,
        phi_levels,
    )
    phis_2d = float(phis_offset) + phis_relative_2d

    ps = surface_pressure(time, latitude, longitude, exact_ps_2d)
    phis = surface_geopotential(time, latitude, longitude, phis_2d)
    surface_theta = xr.DataArray(
        np.broadcast_to(surface_theta_2d[None, :, :], (time.size, latitude.size, longitude.size)).copy(),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        name="surface_potential_temperature",
        attrs={"units": "K"},
    )

    theta_profile = _analytic_theta_from_pressure(np.asarray(level.values, dtype=float), theta_levels, pi_levels)
    theta_values = np.broadcast_to(theta_profile[None, :, None, None], pressure.shape).copy()
    temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    pt = potential_temperature(temperature, pressure)
    theta_mask = make_theta(pressure, ps)

    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)
    policy, solver = solver_for_case(
        ps,
        level,
        solver_kind="k86",
        level_bounds=level_bounds,
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        theta_levels=theta_levels,
        solver_strategy=solver_strategy,
    )
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=level_bounds,
        pressure_tolerance=solver.pressure_tolerance,
        surface_pressure_policy=policy,
    )
    solve_kwargs = {
        "level_bounds": level_bounds,
        "surface_potential_temperature": surface_theta,
    }
    solution = None
    if include_reference:
        solution = solver.solve(pt, pressure, ps, phis=phis, **solve_kwargs)

    representative_theta = weighted_representative_zonal_mean(pt, measure.cell_fraction)
    representative_pressure = weighted_representative_zonal_mean(pressure, measure.cell_fraction)

    theta_top_resolved = float(theta_profile[-1])
    top_edge_pressure = float(pressure_level_edges(level, bounds=level_bounds).isel(level_edge=-1))
    below_surface = theta_levels[None, :, None, None] <= surface_theta.values[:, None, :, :]
    above_model_top = np.broadcast_to(
        theta_levels[None, :, None, None] > theta_top_resolved + 1.0e-12,
        below_surface.shape,
    ).copy()
    free_atmosphere = (~below_surface) & (~above_model_top)

    exact_pressure_on_theta_values = np.where(
        below_surface,
        ps.values[:, None, :, :],
        np.where(
            above_model_top,
            np.nan,
            pi_levels[None, :, None, None],
        ),
    )
    exact_interface_pressure_values = np.where(
        below_surface,
        ps.values[:, None, :, :],
        np.where(
            above_model_top,
            top_edge_pressure,
            pi_levels[None, :, None, None],
        ),
    )
    cell_area = integrator.cell_area.values[None, None, :, :]
    exact_mean_pressure_values = np.sum(exact_interface_pressure_values * cell_area, axis=(2, 3)) / float(
        integrator.cell_area.sum()
    )

    bundle = {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "temperature": temperature,
        "potential_temperature": pt,
        "pt": pt,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "measure": measure,
        "policy": policy,
        "solver": solver,
        "solution": solution,
        "solve_kwargs": solve_kwargs,
        "surface_potential_temperature": surface_theta,
        "theta_levels": theta_levels,
        "exact_pi_levels": xr.DataArray(
            np.broadcast_to(pi_levels[None, :], (time.size, pi_levels.size)).copy(),
            dims=("time", ISENTROPIC_DIM),
            coords={"time": time, ISENTROPIC_DIM: theta_levels},
            name="exact_pi_levels",
            attrs={"units": "Pa"},
        ),
        "exact_phi_levels": xr.DataArray(
            np.broadcast_to(phi_levels[None, :], (time.size, phi_levels.size)).copy(),
            dims=("time", ISENTROPIC_DIM),
            coords={"time": time, ISENTROPIC_DIM: theta_levels},
            name="exact_phi_levels",
            attrs={"units": "m2 s-2"},
        ),
        "exact_pi_s": xr.DataArray(
            np.broadcast_to(exact_ps_2d[None, :, :], (time.size, latitude.size, longitude.size)).copy(),
            dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            name="exact_pi_s",
            attrs={"units": "Pa"},
        ),
        "exact_theta_s": xr.DataArray(
            np.broadcast_to(surface_theta_2d[None, :, :], (time.size, latitude.size, longitude.size)).copy(),
            dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            name="exact_theta_s",
            attrs={"units": "K"},
        ),
        "exact_surface_layer_index": xr.DataArray(
            np.broadcast_to(surface_layer_index_2d[None, :, :], (time.size, latitude.size, longitude.size)).copy(),
            dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            name="exact_surface_layer_index",
        ),
        "exact_pressure_on_theta": xr.DataArray(
            exact_pressure_on_theta_values,
            dims=("time", ISENTROPIC_DIM, "latitude", "longitude"),
            coords={"time": time, ISENTROPIC_DIM: theta_levels, "latitude": latitude, "longitude": longitude},
            name="exact_pressure_on_theta",
            attrs={"units": "Pa"},
        ),
        "exact_interface_pressure": xr.DataArray(
            exact_interface_pressure_values,
            dims=("time", ISENTROPIC_DIM, "latitude", "longitude"),
            coords={"time": time, ISENTROPIC_DIM: theta_levels, "latitude": latitude, "longitude": longitude},
            name="exact_interface_pressure",
            attrs={"units": "Pa"},
        ),
        "exact_mean_pressure_on_theta": xr.DataArray(
            exact_mean_pressure_values,
            dims=("time", ISENTROPIC_DIM),
            coords={"time": time, ISENTROPIC_DIM: theta_levels},
            name="exact_mean_pressure_on_theta",
            attrs={"units": "Pa"},
        ),
        "exact_is_below_surface": xr.DataArray(
            below_surface,
            dims=("time", ISENTROPIC_DIM, "latitude", "longitude"),
            coords={"time": time, ISENTROPIC_DIM: theta_levels, "latitude": latitude, "longitude": longitude},
            name="exact_is_below_surface",
        ),
        "exact_is_above_model_top": xr.DataArray(
            above_model_top,
            dims=("time", ISENTROPIC_DIM, "latitude", "longitude"),
            coords={"time": time, ISENTROPIC_DIM: theta_levels, "latitude": latitude, "longitude": longitude},
            name="exact_is_above_model_top",
        ),
        "exact_is_free_atmosphere": xr.DataArray(
            free_atmosphere,
            dims=("time", ISENTROPIC_DIM, "latitude", "longitude"),
            coords={"time": time, ISENTROPIC_DIM: theta_levels, "latitude": latitude, "longitude": longitude},
            name="exact_is_free_atmosphere",
        ),
        "exact_top_edge_pressure": xr.DataArray(
            np.full(time.size, top_edge_pressure, dtype=float),
            dims=("time",),
            coords={"time": time},
            name="exact_top_edge_pressure",
            attrs={"units": "Pa"},
        ),
        "representative_theta": representative_theta,
        "representative_pressure": representative_pressure,
    }
    return bundle
