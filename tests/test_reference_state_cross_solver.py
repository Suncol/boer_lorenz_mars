from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.reservoirs import A, A_E, A_Z
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.reference_state import (
    FiniteVolumeReferenceState,
    Koehler1986ReferenceState,
    potential_temperature,
)

from .helpers import (
    make_coords,
    pressure_field,
    reference_case_surface_geopotential_values,
    surface_geopotential,
    surface_mass_from_pi_s,
    surface_pressure,
    temperature_from_theta_values,
)
from .helpers_exact import build_asymmetric_reference_case, build_flat_reference_case


_FLAT_ATOL = 1.0e-8
_WEAK_TERRAIN_TOPOGRAPHY_MIN_DELTA = 0.1
_THETA_RESOLUTION_PI_S_MAX_DELTA = 0.05
_THETA_RESOLUTION_RELATIVE_BOUND = 0.5


def _build_flat_cross_solver_pair():
    common_kwargs = {
        "ntime": 1,
        "level_values": (900.0, 700.0, 500.0),
        "ps_value": 1000.0,
        "phis_value": 0.0,
        "theta_profile": (190.0, 210.0, 230.0),
    }
    fv_case = build_flat_reference_case(
        solver_kind="fv",
        pressure_tolerance=1.0e-8,
        max_iterations=80,
        **common_kwargs,
    )
    k86_case = build_flat_reference_case(
        solver_kind="k86",
        surface_theta_value=180.0,
        theta_levels=[180.0, 190.0, 210.0, 230.0, 250.0],
        pressure_tolerance=1.0e-8,
        max_iterations=80,
        **common_kwargs,
    )
    return fv_case, k86_case


def _build_common_weak_terrain_pair():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=(900.0, 700.0, 500.0, 300.0))
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [950.0, 920.0, 890.0, 860.0],
            [930.0, 900.0, 870.0, 840.0],
            [910.0, 880.0, 850.0, 820.0],
            [890.0, 860.0, 830.0, 800.0],
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
            base=100.0,
            lat_range=80.0,
            lon_range=120.0,
        ),
    )
    phis_flat = surface_geopotential(time, latitude, longitude, 0.0)
    theta_values = (
        np.asarray([190.0, 210.0, 230.0, 250.0])[None, :, None, None]
        + np.linspace(0.0, 4.0, latitude.size)[None, None, :, None]
        + np.linspace(0.0, 2.0, longitude.size)[None, None, None, :]
    )
    temperature = temperature_from_theta_values(time, level, latitude, longitude, theta_values)
    pt = potential_temperature(temperature, pressure)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    surface_theta = xr.full_like(ps, 170.0, dtype=float)
    surface_theta.name = "surface_potential_temperature"
    surface_theta.attrs["units"] = "K"

    fv_solver = FiniteVolumeReferenceState(pressure_tolerance=1.0e-6, max_iterations=64)
    fv_solution = fv_solver.solve(pt, pressure, ps, phis=phis)
    fv_flat_solution = fv_solver.solve(pt, pressure, ps, phis=phis_flat)

    k86_solver = Koehler1986ReferenceState(
        theta_levels=[170.0, 190.0, 210.0, 230.0, 250.0, 270.0, 290.0],
        pressure_tolerance=1.0e-5,
        max_iterations=80,
        solver_strategy="koehler_iteration",
    )
    k86_solution = k86_solver.solve(
        pt,
        pressure,
        ps,
        phis=phis,
        surface_potential_temperature=surface_theta,
    )
    k86_flat_solution = k86_solver.solve(
        pt,
        pressure,
        ps,
        phis=phis_flat,
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
        "phis_flat": phis_flat,
        "temperature": temperature,
        "potential_temperature": pt,
        "pt": pt,
        "integrator": integrator,
        "measure": measure,
        "fv_solver": fv_solver,
        "fv_solution": fv_solution,
        "fv_flat_solution": fv_flat_solution,
        "k86_solver": k86_solver,
        "k86_solution": k86_solution,
        "k86_flat_solution": k86_flat_solution,
    }


def test_fv_and_k86_agree_in_flat_limit():
    fv_case, k86_case = _build_flat_cross_solver_pair()
    fv_solution = fv_case["solution"]
    k86_solution = k86_case["solution"]
    pt = fv_case["potential_temperature"]
    pressure = fv_case["pressure"]

    assert bool(fv_solution.converged.values.all())
    assert bool(fv_solution.converged_zonal.values.all())
    assert bool(k86_solution.converged.values.all())
    assert bool(k86_solution.converged_zonal.values.all())

    np.testing.assert_allclose(
        fv_solution.reference_pressure(pt, pressure=pressure).values,
        k86_solution.reference_pressure(pt, pressure=pressure).values,
        rtol=0.0,
        atol=_FLAT_ATOL,
    )
    for name in (
        "pi_s",
        "pi_sZ",
        "reference_surface_pressure",
        "reference_bottom_pressure",
        "reference_top_pressure",
        "total_mass",
    ):
        np.testing.assert_allclose(
            getattr(fv_solution, name).values,
            getattr(k86_solution, name).values,
            rtol=0.0,
            atol=_FLAT_ATOL,
        )


def test_fv_and_k86_both_produce_mass_closed_finite_solutions_on_weak_terrain_case():
    case = _build_common_weak_terrain_pair()
    integrator = case["integrator"]

    for solver, solution, flat_solution in (
        (case["fv_solver"], case["fv_solution"], case["fv_flat_solution"]),
        (case["k86_solver"], case["k86_solution"], case["k86_flat_solution"]),
    ):
        assert np.isfinite(solution.pi_reference.values).any()
        assert np.isfinite(solution.pi_s.values).all()
        assert np.isfinite(solution.pi_sZ.values).all()
        assert float(solution.pi_s.min()) > 0.0
        assert float(solution.pi_sZ.min()) > 0.0
        assert np.isfinite(solution.reference_surface_pressure.values).all()
        assert np.isfinite(solution.reference_bottom_pressure.values).all()
        assert float(solution.reference_surface_pressure.min()) > 0.0
        assert float(solution.reference_bottom_pressure.min()) > 0.0

        np.testing.assert_allclose(
            solution.mass_reference.sum(dim="reference_sample").values,
            solution.total_mass.values,
            rtol=1.0e-8,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            surface_mass_from_pi_s(
                solution.pi_s,
                solution.reference_top_pressure,
                integrator.cell_area,
            ).values,
            solution.total_mass.values,
            rtol=max(10.0 * solver.pressure_tolerance, 1.0e-6),
            atol=0.0,
        )
        assert float(np.max(np.abs(solution.pi_s.values - flat_solution.pi_s.values))) > _WEAK_TERRAIN_TOPOGRAPHY_MIN_DELTA

    np.testing.assert_allclose(
        case["fv_solution"].total_mass.values,
        case["k86_solution"].total_mass.values,
        rtol=0.0,
        atol=10.0,
    )


@pytest.mark.slow_reference
def test_koehler1986_theta_resolution_convergence():
    # Use the default iterative branch here: in the current implementation it is the
    # stable way to isolate fixed-isentrope resolution effects without root-solver
    # convergence false negatives on the coarse ladder.
    coarse = build_asymmetric_reference_case(
        solver_kind="k86",
        ntime=1,
        surface_theta_value=173.0,
        solver_strategy="koehler_iteration",
        pressure_tolerance=1.0e-5,
        max_iterations=120,
        theta_levels=np.arange(170.0, 291.0, 20.0),
    )
    fine = build_asymmetric_reference_case(
        solver_kind="k86",
        ntime=1,
        surface_theta_value=173.0,
        solver_strategy="koehler_iteration",
        pressure_tolerance=1.0e-5,
        max_iterations=120,
        theta_levels=np.arange(170.0, 291.0, 10.0),
    )

    coarse_solution = coarse["solution"]
    fine_solution = fine["solution"]
    assert bool(coarse_solution.converged.values.all())
    assert bool(coarse_solution.converged_zonal.values.all())
    assert bool(fine_solution.converged.values.all())
    assert bool(fine_solution.converged_zonal.values.all())

    coarse_state = coarse["solver"]._last_geometry_state
    fine_state = fine["solver"]._last_geometry_state
    assert coarse_state is not None
    assert fine_state is not None
    total_area = float(coarse["integrator"].cell_area.sum())

    def _joint_pressure_residual(state, solver):
        full = abs(state.full_family.mass_residual) * solver.constants.g / total_area
        zonal = abs(state.zonal_family.mass_residual) * solver.constants.g / total_area
        return max(float(full.max()), float(zonal.max()))

    coarse_joint_residual = _joint_pressure_residual(coarse_state, coarse["solver"])
    fine_joint_residual = _joint_pressure_residual(fine_state, fine["solver"])
    assert fine_joint_residual <= coarse_joint_residual + 1.0e-12

    max_pi_s_delta = float(np.max(np.abs(fine_solution.pi_s.values - coarse_solution.pi_s.values)))
    assert np.isfinite(max_pi_s_delta)
    assert max_pi_s_delta <= _THETA_RESOLUTION_PI_S_MAX_DELTA

    for diagnostic in (A, A_Z, A_E):
        coarse_term = diagnostic(
            coarse["temperature"],
            coarse["pressure"],
            coarse["theta_mask"],
            coarse["integrator"],
            reference_state=coarse_solution,
            measure=coarse["measure"],
            ps=coarse["ps"],
            phis=coarse["phis"],
        )
        fine_term = diagnostic(
            fine["temperature"],
            fine["pressure"],
            fine["theta_mask"],
            fine["integrator"],
            reference_state=fine_solution,
            measure=fine["measure"],
            ps=fine["ps"],
            phis=fine["phis"],
        )
        relative_delta = np.abs(coarse_term.values - fine_term.values) / np.maximum(np.abs(fine_term.values), 1.0)
        assert float(np.max(relative_delta)) <= _THETA_RESOLUTION_RELATIVE_BOUND
