from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.reservoirs import A, A1, A2, A_E, A_E1, A_E2, A_Z, A_Z1, A_Z2

from .helpers_exact import build_k86_exact_topographic_reference_case


def _full_and_zonal_pressure_residual_max(case: dict) -> tuple[float, float]:
    solver = case["solver"]
    state = solver._last_geometry_state
    assert state is not None
    total_area = float(case["integrator"].cell_area.sum())
    full = float(np.nanmax(np.abs(state.full_family.mass_residual.values) * solver.constants.g / total_area))
    zonal = float(np.nanmax(np.abs(state.zonal_family.mass_residual.values) * solver.constants.g / total_area))
    return full, zonal


def _masked_allclose(actual: xr.DataArray, expected: xr.DataArray, mask: xr.DataArray, *, atol: float) -> None:
    selector = np.asarray(mask.values, dtype=bool)
    np.testing.assert_allclose(
        np.asarray(actual.values, dtype=float)[selector],
        np.asarray(expected.values, dtype=float)[selector],
        rtol=0.0,
        atol=atol,
    )


def _ape_tolerance(case: dict) -> float:
    solution = case["solution"]
    assert solution is not None
    return (
        1.0e-12
        * float(solution.total_mass.max())
        * solution.constants.cp
        * float(case["potential_temperature"].max())
    )


def test_koehler1986_exact_topographic_reference_observed_state_matches_analytic_interfaces():
    case = build_k86_exact_topographic_reference_case(solver_strategy="root")
    observed = case["solver"]._last_observed_state
    assert observed is not None

    xr.testing.assert_equal(observed["is_below_surface"], case["exact_is_below_surface"])
    xr.testing.assert_equal(observed["is_above_model_top"], case["exact_is_above_model_top"])
    xr.testing.assert_equal(observed["is_free_atmosphere"], case["exact_is_free_atmosphere"])

    _masked_allclose(
        observed["pressure_on_theta"],
        case["exact_pressure_on_theta"],
        observed["is_free_atmosphere"],
        atol=1.0e-10,
    )
    _masked_allclose(
        observed["pressure_on_theta"],
        case["ps"].broadcast_like(observed["pressure_on_theta"]),
        observed["is_below_surface"],
        atol=1.0e-10,
    )
    assert np.isnan(
        observed["pressure_on_theta"].where(observed["is_above_model_top"]).values[
            np.asarray(observed["is_above_model_top"].values, dtype=bool)
        ]
    ).all()

    np.testing.assert_allclose(
        observed["interface_pressure"].values,
        case["exact_interface_pressure"].values,
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        observed["mean_pressure_on_theta"].values,
        case["exact_mean_pressure_on_theta"].values,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_koehler1986_exact_topographic_reference_geometry_matches_analytic_profile():
    case = build_k86_exact_topographic_reference_case(solver_strategy="root")
    solution = case["solution"]
    assert solution is not None
    state = case["solver"]._last_geometry_state
    assert state is not None

    full = state.full_family.geometry
    zonal = state.zonal_family.geometry

    np.testing.assert_allclose(solution.pi_reference.values, case["exact_pi_levels"].values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        solution.reference_interface_geopotential.values,
        case["exact_phi_levels"].values,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(full.profile.pi_levels.values, case["exact_pi_levels"].values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(full.profile.phi_levels.values, case["exact_phi_levels"].values, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(full.surface.pi_s.values, case["exact_pi_s"].values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(full.surface.theta_s_ref.values, case["exact_theta_s"].values, rtol=0.0, atol=1.0e-9)

    np.testing.assert_allclose(zonal.profile.pi_levels.values, full.profile.pi_levels.values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(zonal.profile.phi_levels.values, full.profile.phi_levels.values, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(zonal.surface.pi_s.values, full.surface.pi_s.values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(zonal.surface.theta_s_ref.values, full.surface.theta_s_ref.values, rtol=0.0, atol=1.0e-9)

    np.testing.assert_allclose(solution.pi_s.values, case["exact_pi_s"].values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(solution.pi_sZ.values, case["exact_pi_s"].values, rtol=0.0, atol=1.0e-6)


def test_koehler1986_exact_topographic_reference_public_curve_is_identity():
    case = build_k86_exact_topographic_reference_case(solver_strategy="root")
    solution = case["solution"]
    assert solution is not None

    pi = solution.reference_pressure(case["potential_temperature"], pressure=case["pressure"])
    _masked_allclose(pi, case["pressure"], case["theta_mask"], atol=1.0e-4)

    pi_z = solution.zonal_reference_pressure(
        case["representative_theta"],
        pressure=case["representative_pressure"],
    )
    np.testing.assert_allclose(
        pi_z.values,
        case["representative_pressure"].values,
        rtol=0.0,
        atol=1.0e-4,
    )


@pytest.mark.parametrize("solver_strategy", ["root", "koehler_iteration"])
def test_koehler1986_zero_ape_reference_atmosphere_with_topography_and_linear_theta_exner(solver_strategy: str):
    case = build_k86_exact_topographic_reference_case(
        solver_strategy=solver_strategy,
        pressure_tolerance=1.0e-9,
        max_iterations=400,
    )
    solution = case["solution"]
    assert solution is not None
    full_pressure_residual, zonal_pressure_residual = _full_and_zonal_pressure_residual_max(case)

    assert bool(solution.converged.values.all())
    assert bool(solution.converged_zonal.values.all())
    assert full_pressure_residual <= 10.0 * case["solver"].pressure_tolerance
    assert zonal_pressure_residual <= 10.0 * case["solver"].pressure_tolerance

    pi = solution.reference_pressure(case["potential_temperature"], pressure=case["pressure"])
    _masked_allclose(pi, case["pressure"], case["theta_mask"], atol=1.0e-4)
    np.testing.assert_allclose(solution.pi_s.values, case["ps"].values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(solution.pi_sZ.values, case["ps"].values, rtol=0.0, atol=1.0e-6)

    temperature = case["temperature"]
    pressure = case["pressure"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    measure = case["measure"]
    ps = case["ps"]
    phis = case["phis"]
    ape_tolerance = _ape_tolerance(case)

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
