from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.reservoirs import A1, A_E1, A_Z1
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import FiniteVolumeReferenceState, potential_temperature

from .helpers import (
    finite_reference_profile,
    full_field,
    make_coords,
    pressure_field,
    pressure_inside_reference_layer,
    reference_layer_mass_from_interfaces,
    surface_geopotential,
    surface_pressure,
    surface_pressure_policy_for_case,
    temperature_from_theta_values,
)
from .helpers_exact import build_asymmetric_reference_case


pytestmark = pytest.mark.legacy_reference_internal


def _solver_for_case(ps, level, *, level_bounds=None):
    return FiniteVolumeReferenceState(
        surface_pressure_policy=surface_pressure_policy_for_case(ps, level, level_bounds=level_bounds)
    )


def test_reference_state_solve_falls_back_to_midpoint_interfaces_without_bounds():
    time = xr.DataArray([0.0], dims=("time",), coords={"time": [0.0]}, attrs={"units": "hours"})
    level = xr.DataArray(
        np.asarray([820.0, 560.0, 240.0]),
        dims=("level",),
        coords={"level": [820.0, 560.0, 240.0]},
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )
    _, _, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 650.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    solution = _solver_for_case(ps, level).solve(
        potential_temperature(temperature, pressure),
        pressure,
        ps,
        phis=phis,
    )

    np.testing.assert_allclose(solution.reference_top_pressure.values, 80.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.pi_s.values, 650.0, atol=1.0e-12)


def test_single_sample_reference_pressure_requires_pressure_and_gives_zero_ape():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[900.0, 700.0, 500.0])
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 850.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    pt = full_field(time, level, latitude, longitude, 220.0, name="potential_temperature", units="K")
    temperature = temperature_from_theta_values(time, level, latitude, longitude, 220.0)
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

    assert int(np.isfinite(solution.theta_reference.isel(time=0)).sum()) == 1
    with pytest.raises(ValueError, match="Single-sample"):
        solution.reference_pressure(pt)
    np.testing.assert_allclose(solution.reference_pressure(pt, pressure=pressure).values, pressure.values)
    np.testing.assert_allclose(solution.efficiency(pt, pressure).values, 0.0, atol=1.0e-12)
    for term in (
        A1(temperature, pressure, theta_mask, integrator, reference_state=solution, ps=ps, potential_temperature_field=pt),
        A_Z1(temperature, pressure, theta_mask, integrator, reference_state=solution, ps=ps, potential_temperature_field=pt),
        A_E1(temperature, pressure, theta_mask, integrator, reference_state=solution, ps=ps, potential_temperature_field=pt),
    ):
        np.testing.assert_allclose(term.values, 0.0, atol=1.0e-12)


def test_reference_state_interface_profiles_are_strictly_monotone():
    case = build_asymmetric_reference_case(solver_kind="fv")
    solution = case["solution"]

    assert solution.reference_interface_pressure.dims == ("time", "reference_interface")
    assert solution.reference_interface_geopotential.dims == ("time", "reference_interface")
    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)


def test_reference_state_layer_mass_closure_from_public_diagnostics():
    case = build_asymmetric_reference_case(solver_kind="fv")
    solution = case["solution"]
    solver = case["solver"]

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        phis_t = case["phis"].isel(time=time_index)
        phis_rel = phis_t - phis_t.min(dim=("latitude", "longitude"))
        total_mass = float(solution.total_mass.isel(time=time_index))
        mass_tolerance = 10.0 * solver.pressure_tolerance * max(total_mass, 1.0)
        for layer_index, theta_layer in enumerate(profile["theta_reference"]):
            reconstructed_mass = reference_layer_mass_from_interfaces(
                profile["reference_interface_pressure"][layer_index],
                profile["reference_interface_pressure"][layer_index + 1],
                profile["reference_interface_geopotential"][layer_index],
                profile["reference_interface_geopotential"][layer_index + 1],
                theta_layer,
                phis_rel,
                case["integrator"].cell_area,
            )
            assert abs(reconstructed_mass - profile["mass_reference"][layer_index]) <= mass_tolerance


def test_reference_state_half_mass_pressure_samples_lie_inside_layers_and_split_mass_in_half():
    case = build_asymmetric_reference_case(solver_kind="fv")
    solution = case["solution"]
    area_values = np.asarray(case["integrator"].cell_area.values, dtype=float)
    solver = case["solver"]

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        phis_t = case["phis"].isel(time=time_index)
        phis_rel = phis_t - phis_t.min(dim=("latitude", "longitude"))
        phis_values = np.asarray(phis_rel.values, dtype=float)
        total_mass = float(solution.total_mass.isel(time=time_index))
        mass_tolerance = 10.0 * solver.pressure_tolerance * max(total_mass, 1.0)
        for layer_index, theta_layer in enumerate(profile["theta_reference"]):
            p_bottom = float(profile["reference_interface_pressure"][layer_index])
            p_top = float(profile["reference_interface_pressure"][layer_index + 1])
            phi_bottom = float(profile["reference_interface_geopotential"][layer_index])
            phi_top = float(profile["reference_interface_geopotential"][layer_index + 1])
            pi_sample = float(profile["pi_reference"][layer_index])
            layer_mass = float(profile["mass_reference"][layer_index])

            assert p_bottom > pi_sample > p_top
            local_bottom = np.full_like(phis_values, p_top, dtype=float)
            full_mask = phis_values <= phi_bottom
            if np.any(full_mask):
                local_bottom[full_mask] = p_bottom
            partial_mask = (phis_values > phi_bottom) & (phis_values < phi_top)
            if np.any(partial_mask):
                local_bottom[partial_mask] = pressure_inside_reference_layer(
                    phis_values[partial_mask],
                    phi_bottom,
                    p_bottom,
                    theta_layer,
                )
            upper_half_mass = float(
                np.sum(np.maximum(np.minimum(local_bottom, pi_sample) - p_top, 0.0) * area_values) / MARS.g
            )
            assert abs(upper_half_mass - 0.5 * layer_mass) <= mass_tolerance
