from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.conversions import C_A, C_K1
from mars_exact_lec.boer.reservoirs import A, A1, A2, A_E, A_E1, A_E2, A_Z, A_Z1, A_Z2
from mars_exact_lec.common.integrals import build_mass_integrator, pressure_level_edges
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.common.zonal_ops import representative_zonal_mean, weighted_representative_zonal_mean
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import KoehlerReferenceState, potential_temperature
from mars_exact_lec.reference_state.koehler_solver import _solve_reference_family

from .helpers import (
    broadcast_surface_zonal,
    finite_reference_profile,
    full_field,
    make_coords,
    pressure_inside_reference_layer,
    pressure_field,
    reference_layer_mass_from_interfaces,
    reference_case_coords,
    reference_case_surface_geopotential_values,
    reference_case_surface_pressure_values,
    reference_case_surface_time_series,
    reference_case_terrain_anomaly_values,
    reference_case_theta_field_values,
    reference_case_theta_profile,
    surface_mass_from_pi_s,
    surface_pressure_policy_for_case,
    surface_geopotential,
    surface_pressure,
    surface_zonal_mean,
    temperature_from_theta_values,
)


def _max_abs(values) -> float:
    return float(np.nanmax(np.abs(np.asarray(values, dtype=float))))


def _difference_threshold(ps, solver: KoehlerReferenceState, factor: float = 100.0) -> float:
    return factor * solver.pressure_tolerance * float(np.asarray(ps.values, dtype=float).max())


def _physical_column_mass(ps, level, integrator, *, bounds=None):
    edges = pressure_level_edges(level, bounds=bounds)
    lower_edge = xr.DataArray(
        np.asarray(edges.values[:-1], dtype=float),
        dims=("level",),
        coords={"level": level.values},
    )
    upper_edge = xr.DataArray(
        np.asarray(edges.values[1:], dtype=float),
        dims=("level",),
        coords={"level": level.values},
    )
    ps_4d = ps.expand_dims(level=level).transpose("time", "level", "latitude", "longitude")
    lower_4d = lower_edge.broadcast_like(ps_4d)
    upper_4d = upper_edge.broadcast_like(ps_4d)
    area_4d = integrator.cell_area.expand_dims(time=ps.coords["time"], level=level).transpose(
        "time",
        "level",
        "latitude",
        "longitude",
    )
    dp = xr.where(ps_4d > upper_4d, xr.apply_ufunc(np.minimum, ps_4d, lower_4d) - upper_4d, 0.0)
    return ((dp * area_4d) / MARS.g).sum(dim=("level", "latitude", "longitude"))


def _solver_for_case(
    ps,
    level,
    *,
    level_bounds=None,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
) -> KoehlerReferenceState:
    return KoehlerReferenceState(
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        surface_pressure_policy=surface_pressure_policy_for_case(
            ps,
            level,
            level_bounds=level_bounds,
            pressure_tolerance=pressure_tolerance,
        ),
    )


def test_reference_state_reproduces_stable_flat_reference_and_zero_ape():
    time, level, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    theta_profile = np.asarray([180.0, 200.0, 220.0])
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_profile[None, :, None, None],
    )

    pt = potential_temperature(temperature, pressure)
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)
    pi = solution.reference_pressure(pt)
    policy = surface_pressure_policy_for_case(ps, level)

    np.testing.assert_allclose(pi.values, pressure.values, atol=1e-12)
    np.testing.assert_allclose(
        solution.pi_s.values,
        np.broadcast_to(
            solution.reference_surface_pressure.values[:, None, None],
            solution.pi_s.shape,
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(solution.pi_sZ.values, solution.pi_s.values, atol=1e-12)
    np.testing.assert_allclose(solution.reference_bottom_pressure.values, solution.reference_surface_pressure.values, atol=1e-12)
    np.testing.assert_allclose(
        A(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
            surface_pressure_policy=policy,
        ).values,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        A_Z(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
            surface_pressure_policy=policy,
        ).values,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        A_E(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
            surface_pressure_policy=policy,
        ).values,
        0.0,
        atol=1e-12,
    )


def test_reference_state_flat_partial_cell_surface_pressure_matches_ps_inside_bottom_layer():
    time = xr.DataArray([0.0], dims=("time",), coords={"time": [0.0]}, attrs={"units": "hours"})
    level = xr.DataArray(
        np.asarray([900.0, 700.0, 500.0]),
        dims=("level",),
        coords={"level": [900.0, 700.0, 500.0]},
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )
    _, _, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 850.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    integrator = build_mass_integrator(level, latitude, longitude)

    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

    np.testing.assert_allclose(solution.pi_s.values, 850.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.reference_surface_pressure.values, 850.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.reference_bottom_pressure.values, 850.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.total_mass.values, _physical_column_mass(ps, level, integrator).values)


def test_available_potential_energy_body_terms_require_ps_or_explicit_measure():
    time, level, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )

    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        A1(temperature, pressure, theta, integrator, n=1.0)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        A_Z1(temperature, pressure, theta, integrator, n_z=1.0)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        A_E1(temperature, pressure, theta, integrator, n=1.0, n_z=1.0)


def test_reference_state_solve_uses_explicit_level_bounds_when_provided():
    time = xr.DataArray([0.0], dims=("time",), coords={"time": [0.0]}, attrs={"units": "hours"})
    level = xr.DataArray(
        np.asarray([820.0, 560.0, 240.0]),
        dims=("level",),
        coords={"level": [820.0, 560.0, 240.0]},
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )
    level_bounds = xr.DataArray(
        np.asarray([[980.0, 700.0], [700.0, 420.0], [420.0, 60.0]]),
        dims=("level", "bounds"),
        coords={"level": level.values, "bounds": [0, 1]},
        name="level_bounds",
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
    pt = potential_temperature(temperature, pressure)
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)

    solution = _solver_for_case(ps, level, level_bounds=level_bounds).solve(pt, pressure, ps, phis=phis, level_bounds=level_bounds)

    np.testing.assert_allclose(solution.reference_top_pressure.values, 60.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.pi_s.values, 650.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.reference_surface_pressure.values, 650.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.total_mass.values, _physical_column_mass(ps, level, integrator, bounds=level_bounds).values)

    reconstructed_mass = (
        ((solution.pi_s - solution.reference_top_pressure) * integrator.cell_area).sum(dim=("latitude", "longitude"))
        / MARS.g
    )
    np.testing.assert_allclose(reconstructed_mass.values, solution.total_mass.values, atol=1.0e-12)


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
    pt = potential_temperature(temperature, pressure)
    integrator = build_mass_integrator(level, latitude, longitude)

    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

    np.testing.assert_allclose(solution.reference_top_pressure.values, 80.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.pi_s.values, 650.0, atol=1.0e-12)
    np.testing.assert_allclose(solution.total_mass.values, _physical_column_mass(ps, level, integrator).values)


def test_single_sample_reference_pressure_requires_pressure_and_gives_zero_ape():
    time = xr.DataArray([0.0], dims=("time",), coords={"time": [0.0]}, attrs={"units": "hours"})
    level = xr.DataArray(
        np.asarray([900.0, 700.0, 500.0]),
        dims=("level",),
        coords={"level": [900.0, 700.0, 500.0]},
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )
    _, _, latitude, longitude = make_coords(ntime=1)
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
    np.testing.assert_allclose(
        A1(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            potential_temperature_field=pt,
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        A_Z1(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            potential_temperature_field=pt,
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        A_E1(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            potential_temperature_field=pt,
        ).values,
        0.0,
        atol=1.0e-12,
    )


def test_reference_state_preserves_total_mass_and_monotonic_reference_curve():
    time, level, latitude, longitude = reference_case_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(
            latitude,
            longitude,
            lat_range=180.0,
            lon_range=360.0,
        ),
    )
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        reference_case_surface_pressure_values(
            latitude,
            longitude,
            base=910.0,
            lon_drop=480.0,
            lat_drop=120.0,
        ),
    )
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(165.0, 245.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="temperature",
        units="K",
    )
    pt = potential_temperature(temperature, pressure)
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

    direct_mass = _physical_column_mass(ps, level, integrator)
    np.testing.assert_allclose(solution.total_mass.values, direct_mass.values)

    reconstructed_mass = (
        ((solution.pi_s - solution.reference_top_pressure) * integrator.cell_area).sum(dim=("latitude", "longitude"))
        / MARS.g
    )
    np.testing.assert_allclose(reconstructed_mass.values, solution.total_mass.values, rtol=5.0e-6, atol=0.0)
    np.testing.assert_allclose(
        solution.reference_bottom_pressure.values,
        solution.pi_s.max(dim=("latitude", "longitude")).values,
    )
    assert solution.pi_s.dims == ps.dims
    assert solution.pi_sZ.dims == ps.dims
    assert solution.iterations is not None
    assert solution.converged is not None
    assert solution.iterations_zonal is not None
    assert solution.converged_zonal is not None
    assert solution.converged.values.all()
    assert solution.converged_zonal.values.all()
    assert np.all(solution.iterations.values >= 1)
    assert np.all(solution.iterations_zonal.values >= 1)
    assert np.isfinite(solution.pi_s.values).all()
    assert np.isfinite(solution.pi_sZ.values).all()

    area_weighted_surface_pressure = (
        solution.pi_s * integrator.cell_area
    ).sum(dim=("latitude", "longitude")) / integrator.cell_area.sum()
    np.testing.assert_allclose(
        area_weighted_surface_pressure.values,
        solution.reference_surface_pressure.values,
        rtol=1.0e-6,
        atol=0.0,
    )
    area_weighted_surface_pressure_zonal = (
        solution.pi_sZ * integrator.cell_area
    ).sum(dim=("latitude", "longitude")) / integrator.cell_area.sum()
    np.testing.assert_allclose(
        area_weighted_surface_pressure_zonal.values,
        solution.reference_surface_pressure.values,
        rtol=2.0e-6,
        atol=0.0,
    )

    for time_index in range(time.size):
        theta_ref = solution.theta_reference.isel(time=time_index).values
        pi_ref = solution.pi_reference.isel(time=time_index).values
        valid = np.isfinite(theta_ref) & np.isfinite(pi_ref)
        assert np.all(np.diff(theta_ref[valid]) > 0.0)
        assert np.all(np.diff(pi_ref[valid]) < 0.0)


def test_reference_state_uniform_phis_offset_leaves_solution_invariant():
    time, level, latitude, longitude = reference_case_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, reference_case_surface_pressure_values(latitude, longitude))
    theta_profile = reference_case_theta_profile(level)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_profile[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solver = _solver_for_case(ps, level)

    flat = solver.solve(
        pt,
        pressure,
        ps,
        phis=surface_geopotential(time, latitude, longitude, 0.0),
    )
    shifted = solver.solve(
        pt,
        pressure,
        ps,
        phis=surface_geopotential(time, latitude, longitude, 2000.0),
    )

    np.testing.assert_allclose(flat.pi_reference.values, shifted.pi_reference.values)
    np.testing.assert_allclose(flat.reference_pressure(pt).values, shifted.reference_pressure(pt).values)
    np.testing.assert_allclose(flat.pi_s.values, shifted.pi_s.values)
    np.testing.assert_allclose(flat.pi_sZ.values, shifted.pi_sZ.values)
    np.testing.assert_allclose(
        flat.reference_surface_pressure.values,
        shifted.reference_surface_pressure.values,
    )
    np.testing.assert_allclose(
        flat.reference_bottom_pressure.values,
        shifted.reference_bottom_pressure.values,
    )


def test_uneven_topography_changes_pi_reference_at_fixed_theta_and_ps():
    time, level, latitude, longitude = reference_case_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        reference_case_surface_pressure_values(
            latitude,
            longitude,
            base=910.0,
            lon_drop=390.0,
            lat_drop=70.0,
        ),
    )
    theta_profile = reference_case_theta_profile(level)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_profile[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solver = _solver_for_case(ps, level)
    phis_base = surface_geopotential(time, latitude, longitude, 2000.0)
    phis_terrain = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_terrain_anomaly_values(latitude, longitude, 1500.0) + 2000.0,
    )

    base = solver.solve(pt, pressure, ps, phis=phis_base)
    terrain = solver.solve(pt, pressure, ps, phis=phis_terrain)
    threshold = _difference_threshold(ps, solver, factor=0.1)

    np.testing.assert_allclose(base.total_mass.values, terrain.total_mass.values)
    np.testing.assert_allclose(
        base.reference_surface_pressure.values,
        terrain.reference_surface_pressure.values,
        rtol=1.0e-6,
        atol=0.0,
    )
    assert _max_abs(terrain.pi_reference.values - base.pi_reference.values) > threshold
    assert _max_abs(terrain.reference_pressure(pt).values - base.reference_pressure(pt).values) > threshold


def test_topography_response_grows_with_terrain_amplitude():
    time, level, latitude, longitude = reference_case_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        reference_case_surface_pressure_values(
            latitude,
            longitude,
            base=910.0,
            lon_drop=390.0,
            lat_drop=70.0,
        ),
    )
    theta_profile = reference_case_theta_profile(level)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_profile[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solver = _solver_for_case(ps, level)
    base = solver.solve(pt, pressure, ps, phis=surface_geopotential(time, latitude, longitude, 2000.0))

    anomaly_small = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_terrain_anomaly_values(latitude, longitude, 750.0) + 2000.0,
    )
    anomaly_large = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_terrain_anomaly_values(latitude, longitude, 1500.0) + 2000.0,
    )
    small = solver.solve(pt, pressure, ps, phis=anomaly_small)
    large = solver.solve(pt, pressure, ps, phis=anomaly_large)
    diff_small = _max_abs(small.pi_reference.values - base.pi_reference.values)
    diff_large = _max_abs(large.pi_reference.values - base.pi_reference.values)
    threshold = _difference_threshold(ps, solver, factor=0.1)

    assert diff_small > threshold
    assert diff_large > diff_small + 0.1 * threshold


def test_pi_sz_matches_pi_s_for_zonal_thermodynamic_state():
    time, level, latitude, longitude = reference_case_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 950.0)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(latitude, longitude),
    )
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        reference_case_theta_profile(level)[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solver = _solver_for_case(ps, level)
    solution = solver.solve(pt, pressure, ps, phis=phis)

    pi_s_mean = broadcast_surface_zonal(surface_zonal_mean(solution.pi_s), longitude, name="pi_s_mean")
    assert _max_abs(solution.pi_s.values - pi_s_mean.values) > _difference_threshold(ps, solver, factor=50.0)
    np.testing.assert_allclose(solution.pi_sZ.values, solution.pi_s.values)


def test_pi_sz_differs_from_weighted_zonal_mean_of_pi_s_for_asymmetric_case():
    time, level, latitude, longitude = reference_case_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, reference_case_surface_pressure_values(latitude, longitude))
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(
            latitude,
            longitude,
            lat_range=0.0,
            lon_range=2400.0,
        ),
    )
    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            170.0,
            270.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="temperature",
        units="K",
    )
    pt = potential_temperature(temperature, pressure)
    solver = _solver_for_case(ps, level)
    solution = solver.solve(pt, pressure, ps, phis=phis)

    pi_s_mean = broadcast_surface_zonal(surface_zonal_mean(solution.pi_s), longitude, name="pi_s_mean")
    assert np.isfinite(solution.pi_sZ.values).all()
    assert np.isfinite(pi_s_mean.values).all()
    assert _max_abs(solution.pi_sZ.values - pi_s_mean.values) > _difference_threshold(ps, solver, factor=100.0)


def test_available_potential_energy_exact_partition_closes():
    case = _build_asymmetric_reference_case()
    pressure = case["pressure"]
    temperature = case["temperature"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    ps = case["ps"]
    phis = case["phis"]
    solution = case["solution"]

    a_total = A(
        temperature,
        pressure,
        theta_mask,
        integrator,
        reference_state=solution,
        ps=ps,
        phis=phis,
    )
    a_z = A_Z(
        temperature,
        pressure,
        theta_mask,
        integrator,
        reference_state=solution,
        ps=ps,
        phis=phis,
    )
    a_e = A_E(
        temperature,
        pressure,
        theta_mask,
        integrator,
        reference_state=solution,
        ps=ps,
        phis=phis,
    )
    az1 = A_Z1(temperature, pressure, theta_mask, integrator, reference_state=solution, ps=ps)
    ae1 = A_E1(temperature, pressure, theta_mask, integrator, reference_state=solution, ps=ps)
    a1 = A1(temperature, pressure, theta_mask, integrator, reference_state=solution, ps=ps)
    a2 = A2(ps, phis, integrator, reference_state=solution)
    az2 = A_Z2(ps, phis, integrator, reference_state=solution)
    ae2 = A_E2(ps, phis, integrator, reference_state=solution)

    np.testing.assert_allclose((a_z + a_e).values, a_total.values)
    np.testing.assert_allclose((az1 + ae1).values, a1.values)
    np.testing.assert_allclose((az2 + ae2).values, a2.values)
    np.testing.assert_allclose((a_total - a1).values, a2.values)


def test_ca_and_ck1_vanish_without_longitudinal_eddies():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    theta_values = np.asarray(
        [
            [180.0, 180.0, 180.0, 180.0],
            [200.0, 200.0, 200.0, 200.0],
            [220.0, 220.0, 220.0, 220.0],
        ]
    )
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_values[None, :, None, :],
    )
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(5.0, 20.0, time.size * level.size * latitude.size).reshape(
            time.size, level.size, latitude.size
        )[..., None],
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-3.0, 2.0, time.size * level.size * latitude.size).reshape(
            time.size, level.size, latitude.size
        )[..., None],
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.1, 0.2, time.size * level.size * latitude.size).reshape(
            time.size, level.size, latitude.size
        )[..., None],
        name="omega",
        units="Pa s-1",
    )

    pt = potential_temperature(temperature, pressure)
    solution = _solver_for_case(ps, level).solve(
        pt,
        pressure,
        ps,
        phis=surface_geopotential(time, latitude, longitude, 0.0),
    )
    policy = surface_pressure_policy_for_case(ps, level)
    n_z = solution.zonal_efficiency(
        representative_zonal_mean(pt, theta_mask),
        representative_zonal_mean(pressure, theta_mask),
    )

    np.testing.assert_allclose(
        C_A(temperature, u, v, omega, n_z, theta_mask, integrator, ps=ps, surface_pressure_policy=policy).values,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        C_K1(u, v, omega, theta_mask, integrator, ps=ps, surface_pressure_policy=policy).values,
        0.0,
        atol=1e-12,
    )


def test_zonal_reference_pressure_requires_zonal_inputs():
    time, level, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta_mask = make_theta(pressure, ps)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)
    representative_theta = representative_zonal_mean(pt, theta_mask)
    representative_pressure = representative_zonal_mean(pressure, theta_mask)

    with pytest.raises(ValueError):
        solution.zonal_reference_pressure(pt)
    with pytest.raises(ValueError):
        solution.zonal_efficiency(representative_theta, pressure)

    pi_z = solution.zonal_reference_pressure(representative_theta, pressure=representative_pressure)
    n_z = solution.zonal_efficiency(representative_theta, representative_pressure)
    assert pi_z.dims == ("time", "level", "latitude")
    assert n_z.dims == ("time", "level", "latitude")


def _build_flat_reference_case(*, grid: str = "regular", ntime: int = 1):
    time, level, latitude, longitude = reference_case_coords(grid=grid, ntime=ntime)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 950.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        reference_case_theta_profile(level)[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solver = _solver_for_case(ps, level)
    solution = solver.solve(pt, pressure, ps, phis=phis)
    return {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "temperature": temperature,
        "pt": pt,
        "solver": solver,
        "solution": solution,
    }


def _build_asymmetric_reference_case(*, grid: str = "regular", ntime: int = 2, level_bounds=None):
    time, level, latitude, longitude = reference_case_coords(grid=grid, ntime=ntime)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, reference_case_surface_pressure_values(latitude, longitude))
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(latitude, longitude),
    )
    theta_mask = make_theta(pressure, ps)
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
    solver = _solver_for_case(ps, level, level_bounds=level_bounds)
    solution = solver.solve(pt, pressure, ps, phis=phis, level_bounds=level_bounds)
    return {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "temperature": temperature,
        "pt": pt,
        "solver": solver,
        "solution": solution,
    }


def test_reference_state_zonal_surface_pressure_uses_measure_aware_representative_theta():
    time, level, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [750.0, 650.0, 450.0, 250.0],
                [750.0, 650.0, 450.0, 250.0],
                [750.0, 650.0, 450.0, 250.0],
                [750.0, 650.0, 450.0, 250.0],
            ]
        ),
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(
            latitude,
            longitude,
            lat_range=320.0,
            lon_range=480.0,
        ),
    )
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    theta_values = np.asarray([180.0, 200.0, 220.0], dtype=float)[None, :, None, None] + np.asarray(
        [0.0, 12.0, 24.0, 36.0],
        dtype=float,
    )[None, None, None, :]
    temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    pt = potential_temperature(temperature, pressure)
    weighted_theta = weighted_representative_zonal_mean(pt, measure.cell_fraction)
    sharp_theta = representative_zonal_mean(pt, theta_mask)
    solver = _solver_for_case(ps, level)
    solution = solver.solve(pt, pressure, ps, phis=phis)
    reference_top_pressure = float(pressure_level_edges(level).isel(level_edge=-1))

    weighted_family = _solve_reference_family(
        weighted_theta.broadcast_like(pt),
        measure.parcel_mass,
        phis,
        reference_top_pressure,
        constants=MARS,
        max_iterations=solver.max_iterations,
        pressure_tolerance=solver.pressure_tolerance,
        integrator_cell_area=integrator.cell_area,
    )
    sharp_family = _solve_reference_family(
        sharp_theta.broadcast_like(pt),
        measure.parcel_mass,
        phis,
        reference_top_pressure,
        constants=MARS,
        max_iterations=solver.max_iterations,
        pressure_tolerance=solver.pressure_tolerance,
        integrator_cell_area=integrator.cell_area,
    )

    assert _max_abs(weighted_theta - sharp_theta) > 1.0e-3
    np.testing.assert_allclose(solution.ps_effective.values, measure.effective_surface_pressure.values)
    np.testing.assert_allclose(solution.pi_sZ.values, weighted_family["pi_s"])
    assert not np.allclose(solution.pi_sZ.values, sharp_family["pi_s"], rtol=1.0e-6, atol=1.0e-8)


def test_reference_state_fast_stress_sentinel_remains_converged_finite_and_closed():
    time, level, latitude, longitude = reference_case_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_2d = reference_case_surface_pressure_values(
        latitude,
        longitude,
        base=930.0,
        lon_drop=560.0,
        lat_drop=140.0,
    )
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        reference_case_surface_time_series(ps_2d, [20.0, -40.0, 60.0]),
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(
            latitude,
            longitude,
            lat_range=900.0,
            lon_range=1400.0,
        ),
    )
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    theta_values = reference_case_theta_field_values(
        time,
        level,
        latitude,
        longitude,
        base=175.0,
        step=5.0,
        time_offsets=[20.0, -40.0, 60.0],
        lat_amplitude=8.0,
        lon_amplitude=12.0,
    )
    temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    pt = potential_temperature(temperature, pressure)
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)
    pi = solution.reference_pressure(pt, pressure=pressure)
    n = solution.efficiency(pt, pressure)

    assert solution.converged is not None
    assert solution.converged_zonal is not None
    assert solution.converged.values.all()
    assert solution.converged_zonal.values.all()
    assert np.isfinite(pi.values).all()
    assert np.isfinite(n.values).all()
    assert np.isfinite(solution.pi_s.values).all()
    assert np.isfinite(solution.pi_sZ.values).all()

    np.testing.assert_allclose(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            integrator.cell_area,
        ).values,
        solution.total_mass.values,
        rtol=20.0 * _solver_for_case(ps, level).pressure_tolerance,
        atol=0.0,
    )

    for time_index in range(time.size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)


def test_reference_state_default_phis_none_matches_zero_topography():
    case = _build_flat_reference_case(ntime=1)
    pressure = case["pressure"]
    ps = case["ps"]
    pt = case["pt"]
    level = case["level"]

    implicit = _solver_for_case(ps, level).solve(pt, pressure, ps)
    explicit = case["solution"]

    np.testing.assert_allclose(implicit.theta_reference.values, explicit.theta_reference.values, atol=1.0e-12)
    np.testing.assert_allclose(implicit.pi_reference.values, explicit.pi_reference.values, atol=1.0e-12)
    np.testing.assert_allclose(implicit.pi_s.values, explicit.pi_s.values, atol=1.0e-12)
    np.testing.assert_allclose(implicit.pi_sZ.values, explicit.pi_sZ.values, atol=1.0e-12)
    np.testing.assert_allclose(
        implicit.reference_surface_pressure.values,
        explicit.reference_surface_pressure.values,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        implicit.reference_bottom_pressure.values,
        explicit.reference_bottom_pressure.values,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        implicit.reference_interface_pressure.values,
        explicit.reference_interface_pressure.values,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        implicit.reference_interface_geopotential.values,
        explicit.reference_interface_geopotential.values,
        atol=1.0e-12,
    )


def test_reference_state_public_mass_reference_sums_to_total_mass():
    case = _build_asymmetric_reference_case()
    solution = case["solution"]

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        np.testing.assert_allclose(
            profile["mass_reference"].sum(),
            float(solution.total_mass.isel(time=time_index)),
            rtol=1.0e-12,
            atol=0.0,
        )


def test_reference_state_interface_profiles_are_strictly_monotone():
    case = _build_asymmetric_reference_case()
    solution = case["solution"]
    solver = case["solver"]

    assert solution.reference_interface_pressure.dims == ("time", "reference_interface")
    assert solution.reference_interface_geopotential.dims == ("time", "reference_interface")
    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)
        np.testing.assert_allclose(
            profile["reference_interface_pressure"][0],
            float(solution.reference_bottom_pressure.isel(time=time_index)),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            profile["reference_interface_pressure"][-1],
            float(solution.reference_top_pressure.isel(time=time_index)),
            rtol=10.0 * solver.pressure_tolerance,
            atol=0.0,
        )


def test_reference_state_layer_mass_closure_from_public_diagnostics():
    case = _build_asymmetric_reference_case()
    solution = case["solution"]
    integrator = case["integrator"]
    phis = case["phis"]
    solver = case["solver"]

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        total_mass = float(solution.total_mass.isel(time=time_index))
        mass_tolerance = 10.0 * solver.pressure_tolerance * max(total_mass, 1.0)
        for layer_index, theta_layer in enumerate(profile["theta_reference"]):
            reconstructed_mass = reference_layer_mass_from_interfaces(
                profile["reference_interface_pressure"][layer_index],
                profile["reference_interface_pressure"][layer_index + 1],
                profile["reference_interface_geopotential"][layer_index],
                profile["reference_interface_geopotential"][layer_index + 1],
                theta_layer,
                phis.isel(time=time_index),
                integrator.cell_area,
            )
            assert abs(reconstructed_mass - profile["mass_reference"][layer_index]) <= mass_tolerance


def test_reference_state_half_mass_pressure_samples_lie_inside_layers_and_split_mass_in_half():
    case = _build_asymmetric_reference_case()
    solution = case["solution"]
    phis = case["phis"]
    area_values = np.asarray(case["integrator"].cell_area.values, dtype=float)
    solver = case["solver"]

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        phis_values = np.asarray(phis.isel(time=time_index).values, dtype=float)
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


def test_reference_pressure_and_efficiency_are_formula_consistent():
    case = _build_asymmetric_reference_case()
    pressure = case["pressure"]
    pt = case["pt"]
    theta_mask = case["theta_mask"]
    solution = case["solution"]

    pi = solution.reference_pressure(pt, pressure=pressure)
    n = solution.efficiency(pt, pressure)
    np.testing.assert_allclose(
        n.values,
        (1.0 - (pi / pressure) ** MARS.kappa).values,
        atol=1.0e-12,
    )

    representative_theta = representative_zonal_mean(pt, theta_mask)
    representative_pressure = representative_zonal_mean(pressure, theta_mask)
    pi_z = solution.zonal_reference_pressure(representative_theta, pressure=representative_pressure)
    n_z = solution.zonal_efficiency(representative_theta, representative_pressure)
    np.testing.assert_allclose(
        n_z.values,
        (1.0 - (pi_z / representative_pressure) ** MARS.kappa).values,
        atol=1.0e-12,
    )


def test_reference_pressure_boundary_fill_and_coordinate_guards():
    case = _build_asymmetric_reference_case(ntime=1)
    pressure = case["pressure"]
    pt = case["pt"]
    theta_mask = case["theta_mask"]
    solution = case["solution"]
    profile = finite_reference_profile(solution, time_index=0)

    low_theta = xr.full_like(pt, float(profile["theta_reference"][0] - 25.0))
    high_theta = xr.full_like(pt, float(profile["theta_reference"][-1] + 25.0))

    expected_bottom = xr.broadcast(solution.reference_bottom_pressure, low_theta)[0]
    expected_top = xr.broadcast(solution.reference_top_pressure, high_theta)[0]
    np.testing.assert_allclose(
        solution.reference_pressure(low_theta).values,
        expected_bottom.values,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        solution.reference_pressure(high_theta).values,
        expected_top.values,
        atol=1.0e-12,
    )

    representative_theta = representative_zonal_mean(pt, theta_mask)
    representative_pressure = representative_zonal_mean(pressure, theta_mask)
    bad_full_pressure = pressure.assign_coords(longitude=pressure.coords["longitude"].values + 0.5)
    bad_zonal_pressure = representative_pressure.assign_coords(latitude=representative_pressure.coords["latitude"].values + 0.5)

    with pytest.raises(ValueError, match="share the same dims"):
        solution.reference_pressure(pt, pressure=representative_pressure)
    with pytest.raises(ValueError, match="Coordinate 'longitude'"):
        solution.reference_pressure(pt, pressure=bad_full_pressure)
    with pytest.raises(ValueError, match="Coordinate 'latitude'"):
        solution.zonal_reference_pressure(representative_theta, pressure=bad_zonal_pressure)
    with pytest.raises(ValueError, match="Coordinate 'latitude'"):
        solution.zonal_efficiency(representative_theta, bad_zonal_pressure)


def test_reference_state_rejects_invalid_surface_inputs():
    case = _build_flat_reference_case(ntime=1)
    pressure = case["pressure"]
    pt = case["pt"]
    time = case["time"]
    latitude = case["latitude"]
    longitude = case["longitude"]

    invalid_phis = surface_geopotential(time, latitude, longitude, 0.0)
    invalid_phis.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Surface geopotential must remain finite"):
        _solver_for_case(case["ps"], case["level"]).solve(pt, pressure, case["ps"], phis=invalid_phis)

    no_air_ps = surface_pressure(time, latitude, longitude, 50.0)
    with pytest.raises(ValueError, match="at least one above-ground parcel"):
        _solver_for_case(no_air_ps, case["level"]).solve(pt, pressure, no_air_ps, phis=case["phis"])


def test_reference_state_gaussian_grid_preserves_flat_limit_contract():
    time, level, latitude, longitude = reference_case_coords(grid="gaussian", ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 875.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        reference_case_theta_profile(level)[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)
    pi = solution.reference_pressure(pt, pressure=pressure)
    ape_tolerance = (
        1.0e-16
        * float(solution.total_mass.isel(time=0))
        * MARS.cp
        * float(reference_case_theta_profile(level).max())
    )

    np.testing.assert_allclose(
        pi.isel(level=slice(1, None)).values,
        pressure.isel(level=slice(1, None)).values,
        atol=1.0e-12,
    )
    assert np.all(
        np.asarray(pi.isel(level=0).values, dtype=float)
        <= float(solution.reference_bottom_pressure.isel(time=0)) + 1.0e-12
    )
    assert np.all(np.asarray(pi.isel(level=0).values, dtype=float) > np.asarray(pressure.isel(level=1).values, dtype=float))
    np.testing.assert_allclose(
        solution.pi_s.values,
        np.broadcast_to(solution.reference_surface_pressure.values[:, None, None], solution.pi_s.shape),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(solution.pi_sZ.values, solution.pi_s.values, atol=1.0e-12)
    np.testing.assert_allclose(solution.reference_bottom_pressure.values, solution.reference_surface_pressure.values, atol=1.0e-12)
    np.testing.assert_allclose(
        A(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
        ).values,
        0.0,
        atol=ape_tolerance,
    )
    np.testing.assert_allclose(
        A_Z(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
        ).values,
        0.0,
        atol=ape_tolerance,
    )
    np.testing.assert_allclose(
        A_E(
            temperature,
            pressure,
            theta_mask,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
        ).values,
        0.0,
        atol=ape_tolerance,
    )
    np.testing.assert_allclose(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            integrator.cell_area,
        ).values,
        solution.total_mass.values,
        atol=1.0e-12,
    )
