from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.reference_state import FiniteVolumeReferenceState, potential_temperature

from .helpers import (
    finite_reference_profile,
    pressure_field,
    reference_case_coords,
    reference_case_level_bounds,
    reference_case_surface_geopotential_values,
    reference_case_surface_pressure_values,
    reference_case_theta_profile,
    surface_geopotential,
    surface_mass_from_pi_s,
    surface_pressure,
    temperature_from_theta_values,
)


pytestmark = [pytest.mark.slow_reference, pytest.mark.legacy_reference_internal]


def _build_reference_case(*, grid: str = "regular", ntime: int = 2, level_bounds=None):
    time, level, latitude, longitude = reference_case_coords(grid=grid, ntime=ntime)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, reference_case_surface_pressure_values(latitude, longitude))
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(latitude, longitude),
    )
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        reference_case_theta_profile(level)[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solution = FiniteVolumeReferenceState().solve(pt, pressure, ps, phis=phis, level_bounds=level_bounds)
    return {
        "time": time,
        "level": level,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "integrator": integrator,
        "temperature": temperature,
        "pt": pt,
        "solution": solution,
    }


@pytest.mark.parametrize("use_bounds", [False, True])
def test_reference_state_regression_level_bounds_and_midpoint_paths_remain_closed_and_monotone(use_bounds):
    level_bounds = None
    if use_bounds:
        _, level, _, _ = reference_case_coords(ntime=1)
        level_bounds = reference_case_level_bounds(level)

    case = _build_reference_case(level_bounds=level_bounds)
    solution = case["solution"]

    np.testing.assert_allclose(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            case["integrator"].cell_area,
        ).values,
        solution.total_mass.values,
        rtol=10.0 * FiniteVolumeReferenceState().pressure_tolerance,
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
        rtol=10.0 * FiniteVolumeReferenceState().pressure_tolerance,
        atol=0.0,
    )

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert profile["reference_interface_pressure"].size == profile["theta_reference"].size + 1
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)
