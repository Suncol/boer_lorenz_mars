from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.reference_state import KoehlerReferenceState, potential_temperature

from .helpers import (
    finite_reference_profile,
    full_field,
    make_coords,
    pressure_field,
    surface_geopotential,
    surface_mass_from_pi_s,
    surface_pressure,
    temperature_from_theta_values,
)


pytestmark = pytest.mark.slow_reference


def _max_abs(values) -> float:
    return float(np.nanmax(np.abs(np.asarray(values, dtype=float))))


def _difference_threshold(ps, solver: KoehlerReferenceState, factor: float = 100.0) -> float:
    return factor * solver.pressure_tolerance * float(np.asarray(ps.values, dtype=float).max())


def _build_reference_case(*, grid: str = "regular", ntime: int = 2, level_bounds=None):
    time, level, latitude, longitude = make_coords(grid=grid, ntime=ntime)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [900.0, 700.0, 500.0, 350.0],
                [900.0, 700.0, 500.0, 350.0],
                [900.0, 700.0, 500.0, 350.0],
                [900.0, 700.0, 500.0, 350.0],
            ]
        ),
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [0.0, 100.0, 200.0, 300.0],
                [50.0, 150.0, 250.0, 350.0],
                [100.0, 200.0, 300.0, 400.0],
                [150.0, 250.0, 350.0, 450.0],
            ]
        ),
    )
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)
    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(170.0, 255.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="temperature",
        units="K",
    )
    pt = potential_temperature(temperature, pressure)
    solution = KoehlerReferenceState().solve(pt, pressure, ps, phis=phis, level_bounds=level_bounds)
    return {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "integrator": integrator,
        "temperature": temperature,
        "pt": pt,
        "solution": solution,
    }


def test_reference_state_regression_topography_response_grows_with_terrain_amplitude():
    time, level, latitude, longitude = make_coords(ntime=2)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [900.0, 650.0, 520.0, 420.0],
                [900.0, 650.0, 520.0, 420.0],
                [900.0, 650.0, 520.0, 420.0],
                [900.0, 650.0, 520.0, 420.0],
            ]
        ),
    )
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solver = KoehlerReferenceState()

    base = solver.solve(pt, pressure, ps, phis=surface_geopotential(time, latitude, longitude, 2000.0))
    anomaly_small = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [-750.0, -250.0, 250.0, 750.0],
                [-750.0, -250.0, 250.0, 750.0],
                [-750.0, -250.0, 250.0, 750.0],
                [-750.0, -250.0, 250.0, 750.0],
            ]
        )
        + 2000.0,
    )
    anomaly_large = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [-1500.0, -500.0, 500.0, 1500.0],
                [-1500.0, -500.0, 500.0, 1500.0],
                [-1500.0, -500.0, 500.0, 1500.0],
                [-1500.0, -500.0, 500.0, 1500.0],
            ]
        )
        + 2000.0,
    )

    small = solver.solve(pt, pressure, ps, phis=anomaly_small)
    large = solver.solve(pt, pressure, ps, phis=anomaly_large)
    diff_small = _max_abs(small.pi_reference.values - base.pi_reference.values)
    diff_large = _max_abs(large.pi_reference.values - base.pi_reference.values)
    threshold = _difference_threshold(ps, solver, factor=0.1)

    assert diff_small > threshold
    assert diff_large > diff_small + 0.1 * threshold


@pytest.mark.parametrize("use_bounds", [False, True])
def test_reference_state_regression_level_bounds_and_midpoint_paths_remain_closed_and_monotone(use_bounds):
    level_bounds = None
    if use_bounds:
        level_bounds = xr.DataArray(
            np.asarray([[820.0, 600.0], [600.0, 400.0], [400.0, 200.0]]),
            dims=("level", "bounds"),
            coords={"level": [700.0, 500.0, 300.0], "bounds": [0, 1]},
            name="level_bounds",
        )

    case = _build_reference_case(level_bounds=level_bounds)
    solution = case["solution"]

    np.testing.assert_allclose(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            case["integrator"].cell_area,
        ).values,
        solution.total_mass.values,
        rtol=10.0 * KoehlerReferenceState().pressure_tolerance,
        atol=0.0,
    )

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)


@pytest.mark.parametrize("grid", ["regular", "gaussian"])
def test_reference_state_regression_regular_and_gaussian_grids_produce_finite_closed_profiles(grid):
    case = _build_reference_case(grid=grid)
    solution = case["solution"]

    assert np.isfinite(solution.pi_reference.values).any()
    assert np.isfinite(solution.reference_interface_pressure.values).any()
    assert np.isfinite(solution.reference_interface_geopotential.values).any()
    assert np.isfinite(solution.pi_s.values).all()
    assert np.isfinite(solution.pi_sZ.values).all()

    np.testing.assert_allclose(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            case["integrator"].cell_area,
        ).values,
        solution.total_mass.values,
        rtol=10.0 * KoehlerReferenceState().pressure_tolerance,
        atol=0.0,
    )

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert profile["reference_interface_pressure"].size == profile["theta_reference"].size + 1
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)
