from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.reservoirs import A, A1, A2, A_E, A_E1, A_E2, A_Z, A_Z1, A_Z2
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import Koehler1986ReferenceState, ReferenceStateSolution, potential_temperature

from .helpers import (
    make_coords,
    pressure_field,
    reference_case_surface_geopotential_values,
    surface_geopotential,
    surface_mass_from_pi_s,
    surface_pressure,
    temperature_from_theta_values,
)


def _surface_theta(ps: xr.DataArray, value: float) -> xr.DataArray:
    field = xr.full_like(ps, float(value), dtype=float)
    field.name = "surface_potential_temperature"
    field.attrs["units"] = "K"
    return field


def _build_flat_k86_case(*, phis_value: float = 0.0, solver_strategy: str = "koehler_iteration"):
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[900.0, 700.0, 500.0])
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 950.0)
    phis = surface_geopotential(time, latitude, longitude, phis_value)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([190.0, 210.0, 230.0])[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    surface_theta = _surface_theta(ps, 180.0)
    solver = Koehler1986ReferenceState(
        theta_levels=[180.0, 190.0, 210.0, 230.0, 250.0],
        pressure_tolerance=1.0e-8,
        max_iterations=64,
        solver_strategy=solver_strategy,
    )
    solution = solver.solve(
        pt,
        pressure,
        ps,
        phis=phis,
        surface_potential_temperature=surface_theta,
    )
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
        "theta_mask": theta_mask,
        "integrator": integrator,
        "measure": measure,
        "solver": solver,
        "solution": solution,
    }


def _build_asymmetric_k86_case(*, solver_strategy: str = "koehler_iteration", pressure_tolerance: float = 1.0e-5):
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[900.0, 700.0, 500.0, 300.0])
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [950.0, 900.0, 850.0, 800.0],
            [920.0, 870.0, 820.0, 770.0],
            [900.0, 850.0, 800.0, 750.0],
            [880.0, 830.0, 780.0, 730.0],
        ],
        dtype=float,
    )
    ps = surface_pressure(time, latitude, longitude, ps_values)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(
            latitude,
            longitude,
            lat_range=120.0,
            lon_range=240.0,
        ),
    )
    theta_values = (
        np.asarray([190.0, 210.0, 230.0, 250.0])[None, :, None, None]
        + np.linspace(0.0, 8.0, latitude.size)[None, None, :, None]
        + np.linspace(0.0, 4.0, longitude.size)[None, None, None, :]
    )
    temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    pt = potential_temperature(temperature, pressure)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    surface_theta = _surface_theta(ps, 170.0)
    solver = Koehler1986ReferenceState(
        theta_levels=[170.0, 190.0, 210.0, 230.0, 250.0, 270.0, 290.0],
        pressure_tolerance=pressure_tolerance,
        max_iterations=80,
        solver_strategy=solver_strategy,
    )
    solution = solver.solve(
        pt,
        pressure,
        ps,
        phis=phis,
        surface_potential_temperature=surface_theta,
    )
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
        "theta_mask": theta_mask,
        "integrator": integrator,
        "measure": measure,
        "solver": solver,
        "solution": solution,
    }


def test_koehler1986_flat_reference_returns_public_solution_and_pressure_curve():
    case = _build_flat_k86_case()
    solution = case["solution"]

    assert isinstance(solution, ReferenceStateSolution)
    assert solution.method == "koehler1986_isentropic_surface_iteration"
    assert bool(solution.converged.values.all())
    assert bool(solution.converged_zonal.values.all())
    np.testing.assert_allclose(
        solution.reference_pressure(case["potential_temperature"], pressure=case["pressure"]).values,
        case["pressure"].values,
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(solution.pi_s.values, case["ps"].values, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(solution.pi_sZ.values, case["ps"].values, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(
        solution.mass_reference.sum(dim="reference_sample").values,
        solution.total_mass.values,
        rtol=1.0e-12,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            case["integrator"].cell_area,
        ).values,
        solution.total_mass.values,
        rtol=1.0e-12,
        atol=1.0e-4,
    )


def test_koehler1986_iteration_conserves_observed_isentropic_layer_mass():
    case = _build_asymmetric_k86_case()
    solver = case["solver"]
    state = solver._last_geometry_state
    assert state is not None
    full_family = state.full_family
    zonal_family = state.zonal_family

    total_area = float(case["integrator"].cell_area.sum())
    full_pressure_residual = abs(full_family.mass_residual) * solver.constants.g / total_area
    zonal_pressure_residual = abs(zonal_family.mass_residual) * solver.constants.g / total_area
    assert float(full_pressure_residual.max()) <= 10.0 * solver.pressure_tolerance
    assert float(zonal_pressure_residual.max()) <= 10.0 * solver.pressure_tolerance
    assert bool(full_family.converged.values.all())
    assert bool(zonal_family.converged.values.all())
    assert np.isfinite(case["solution"].pi_s.values).all()
    assert np.isfinite(case["solution"].pi_sZ.values).all()
    assert float(abs(case["solution"].pi_s - case["solution"].pi_sZ).max()) > 1.0e-6


def test_koehler1986_surface_terms_match_explicit_solution_overrides():
    case = _build_asymmetric_k86_case()
    ps = case["ps"]
    phis = case["phis"]
    integrator = case["integrator"]
    measure = case["measure"]
    solution = case["solution"]

    implicit_a2 = A2(ps, phis, integrator, reference_state=solution, measure=measure)
    explicit_a2 = A2(ps, phis, integrator, measure=measure, pi_s=solution.pi_s)
    implicit_az2 = A_Z2(ps, phis, integrator, reference_state=solution, measure=measure)
    explicit_az2 = A_Z2(ps, phis, integrator, measure=measure, pi_sZ=solution.pi_sZ)
    implicit_ae2 = A_E2(ps, phis, integrator, reference_state=solution, measure=measure)
    explicit_ae2 = A_E2(
        ps,
        phis,
        integrator,
        measure=measure,
        pi_s=solution.pi_s,
        pi_sZ=solution.pi_sZ,
    )

    np.testing.assert_allclose(implicit_a2.values, explicit_a2.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose(implicit_az2.values, explicit_az2.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose(implicit_ae2.values, explicit_ae2.values, rtol=1.0e-12, atol=1.0e-10)


def test_koehler1986_requires_top_fixed_isentrope_above_model_top():
    case = _build_flat_k86_case()
    solver = Koehler1986ReferenceState(
        theta_levels=[180.0, 190.0, 210.0, 230.0],
        pressure_tolerance=1.0e-8,
        max_iterations=64,
    )

    with pytest.raises(ValueError, match="highest fixed isentropic level"):
        solver.solve(
            case["potential_temperature"],
            case["pressure"],
            case["ps"],
            phis=case["phis"],
            surface_potential_temperature=_surface_theta(case["ps"], 180.0),
        )


def test_koehler1986_root_and_iteration_are_consistent():
    iteration_case = _build_asymmetric_k86_case(solver_strategy="koehler_iteration", pressure_tolerance=1.0e-5)
    root_case = _build_asymmetric_k86_case(solver_strategy="root", pressure_tolerance=1.0e-5)
    iteration = iteration_case["solution"]
    root = root_case["solution"]
    atol = max(20.0 * iteration_case["solver"].pressure_tolerance, 1.0e-6)

    assert bool(iteration.converged.values.all())
    assert bool(iteration.converged_zonal.values.all())
    assert bool(root.converged.values.all())
    assert bool(root.converged_zonal.values.all())
    for name in ("pi_reference", "pi_s", "pi_sZ", "reference_surface_pressure", "reference_bottom_pressure"):
        np.testing.assert_allclose(
            getattr(iteration, name).values,
            getattr(root, name).values,
            rtol=0.0,
            atol=atol,
        )


def test_koehler1986_zero_ape_reference_atmosphere_with_constant_nonzero_phis():
    case = _build_flat_k86_case(phis_value=2000.0)
    pressure = case["pressure"]
    temperature = case["temperature"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    measure = case["measure"]
    ps = case["ps"]
    phis = case["phis"]
    solution = case["solution"]
    ape_tolerance = (
        1.0e-14
        * float(solution.total_mass.max())
        * solution.constants.cp
        * float(case["potential_temperature"].max())
    )

    for term in (
        A(temperature, pressure, theta_mask, integrator, reference_state=solution, measure=measure, ps=ps, phis=phis),
        A_Z(temperature, pressure, theta_mask, integrator, reference_state=solution, measure=measure, ps=ps, phis=phis),
        A_E(temperature, pressure, theta_mask, integrator, reference_state=solution, measure=measure, ps=ps, phis=phis),
        A1(temperature, pressure, theta_mask, integrator, reference_state=solution, measure=measure, ps=ps),
        A_Z1(temperature, pressure, theta_mask, integrator, reference_state=solution, measure=measure, ps=ps),
        A_E1(temperature, pressure, theta_mask, integrator, reference_state=solution, measure=measure, ps=ps),
        A2(ps, phis, integrator, reference_state=solution, measure=measure),
        A_Z2(ps, phis, integrator, reference_state=solution, measure=measure),
        A_E2(ps, phis, integrator, reference_state=solution, measure=measure),
    ):
        np.testing.assert_allclose(term.values, 0.0, rtol=0.0, atol=ape_tolerance)


def test_koehler1986_full_and_zonal_layer_mass_closure():
    case = _build_asymmetric_k86_case()
    solver = case["solver"]
    state = solver._last_geometry_state
    assert state is not None
    total_area = float(case["integrator"].cell_area.sum())

    for family, pi_s in (
        (state.full_family, case["solution"].pi_s),
        (state.zonal_family, case["solution"].pi_sZ),
    ):
        pressure_residual = abs(family.mass_residual) * solver.constants.g / total_area
        assert float(pressure_residual.max()) <= 10.0 * solver.pressure_tolerance
        np.testing.assert_allclose(
            family.layer_mass_reference.sum(dim="isentropic_layer").values,
            case["solution"].total_mass.values,
            rtol=1.0e-8,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            surface_mass_from_pi_s(
                pi_s,
                case["solution"].reference_top_pressure,
                case["integrator"].cell_area,
            ).values,
            case["solution"].total_mass.values,
            rtol=1.0e-8,
            atol=1.0e-4,
        )


def test_koehler1986_flat_limit_is_insensitive_to_zero_vs_constant_phis():
    zero = _build_flat_k86_case(phis_value=0.0)["solution"]
    shifted_case = _build_flat_k86_case(phis_value=2000.0)
    shifted = shifted_case["solution"]
    pt = shifted_case["potential_temperature"]
    pressure = shifted_case["pressure"]

    for name in ("pi_reference", "pi_s", "pi_sZ", "reference_surface_pressure", "reference_bottom_pressure"):
        np.testing.assert_allclose(getattr(zero, name).values, getattr(shifted, name).values, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(
        zero.reference_pressure(pt, pressure=pressure).values,
        shifted.reference_pressure(pt, pressure=pressure).values,
        rtol=0.0,
        atol=1.0e-8,
    )
