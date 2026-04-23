from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import KoehlerReferenceState, potential_temperature

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


def solver_for_case(
    ps,
    level,
    *,
    level_bounds=None,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
):
    policy = surface_pressure_policy_for_case(
        ps,
        level,
        level_bounds=level_bounds,
        pressure_tolerance=pressure_tolerance,
    )
    solver = KoehlerReferenceState(
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        surface_pressure_policy=policy,
    )
    return policy, solver


def build_flat_reference_case(
    *,
    ntime: int = 1,
    level_values=(900.0, 700.0, 500.0),
    level_bounds=None,
    ps_value: float = 850.0,
    phis_value: float = 2000.0,
    theta_profile=(210.0, 230.0, 250.0),
    time_theta_offsets=None,
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
    policy, solver = solver_for_case(ps, level, level_bounds=level_bounds)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=level_bounds,
        pressure_tolerance=solver.pressure_tolerance,
        surface_pressure_policy=policy,
    )
    solution = solver.solve(pt, pressure, ps, phis=phis, level_bounds=level_bounds)

    return {
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
    }


def build_asymmetric_exact_case(
    *,
    ntime: int = 2,
    level_values=(700.0, 500.0, 300.0),
    level_bounds=None,
    include_reference: bool = True,
):
    time, level, latitude, longitude = make_coords(ntime=ntime, level_values=level_values)
    pressure = pressure_field(time, level, latitude, longitude)
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
    policy, solver = solver_for_case(ps, level, level_bounds=level_bounds)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=level_bounds,
        pressure_tolerance=solver.pressure_tolerance,
        surface_pressure_policy=policy,
    )
    solution = None
    if include_reference:
        solution = solver.solve(pt, pressure, ps, phis=phis, level_bounds=level_bounds)

    return {
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
    }
