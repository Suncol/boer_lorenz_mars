from __future__ import annotations

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

import mars_exact_lec.reference_state as reference_state_module
import mars_exact_lec.reference_state.koehler_solver as koehler_solver_module
from mars_exact_lec.reference_state import (
    FiniteVolumeReferenceState,
    Koehler1986ReferenceState,
    ReferenceStateSolution,
    potential_temperature,
)
from mars_exact_lec.reference_state.koehler_solver import _solve_reference_family

from .helpers import make_coords, pressure_field, surface_geopotential, surface_pressure, temperature_from_theta_values


def _build_reference_inputs():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[900.0, 700.0, 500.0])
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
    return pt, pressure, ps, phis


def _build_nonmonotonic_reference_inputs():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[900.0, 700.0, 500.0])
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 950.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([190.0, 185.0, 230.0])[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    surface_theta = xr.full_like(ps, 170.0, dtype=float)
    surface_theta.name = "surface_potential_temperature"
    surface_theta.attrs["units"] = "K"
    return pt, pressure, ps, phis, surface_theta


def test_reference_state_public_exports_include_phase01_classes():
    assert reference_state_module.FiniteVolumeReferenceState is FiniteVolumeReferenceState
    assert reference_state_module.Koehler1986ReferenceState is Koehler1986ReferenceState
    assert "FiniteVolumeReferenceState" in reference_state_module.__all__
    assert "Koehler1986ReferenceState" in reference_state_module.__all__
    assert "KoehlerReferenceState" not in reference_state_module.__all__
    assert ReferenceStateSolution is not None


def test_reference_state_legacy_alias_access_warns_from_package_and_shim():
    with pytest.warns(DeprecationWarning, match="deprecated legacy alias"):
        package_alias = reference_state_module.KoehlerReferenceState
    with pytest.warns(DeprecationWarning, match="deprecated legacy alias"):
        shim_alias = koehler_solver_module.KoehlerReferenceState

    assert package_alias is FiniteVolumeReferenceState
    assert shim_alias is FiniteVolumeReferenceState
    assert "KoehlerReferenceState" not in koehler_solver_module.__all__


def test_reference_state_koehler_solver_shim_reexports_legacy_private_helper():
    assert _solve_reference_family is not None


def test_finite_volume_reference_state_matches_legacy_alias_and_sets_metadata():
    pt, pressure, ps, phis = _build_reference_inputs()

    with pytest.warns(DeprecationWarning, match="deprecated legacy alias"):
        legacy_cls = reference_state_module.KoehlerReferenceState
    legacy_solution = legacy_cls().solve(pt, pressure, ps, phis=phis)
    fv_solution = FiniteVolumeReferenceState().solve(pt, pressure, ps, phis=phis)

    assert isinstance(legacy_solution, ReferenceStateSolution)
    assert isinstance(fv_solution, ReferenceStateSolution)
    assert legacy_solution.method == "finite_volume_parcel_sorted"
    assert fv_solution.method == "finite_volume_parcel_sorted"
    assert (
        fv_solution.theta_reference.attrs["reference_coordinate_semantics"]
        == "finite_volume_theta_groups"
    )
    assert (
        fv_solution.pi_reference.attrs["reference_pressure_sampling"]
        == "half_mass_pressure_sample"
    )

    for name in (
        "theta_reference",
        "pi_reference",
        "mass_reference",
        "reference_interface_pressure",
        "reference_interface_geopotential",
        "total_mass",
        "reference_surface_pressure",
        "reference_bottom_pressure",
        "reference_top_pressure",
        "pi_s",
        "pi_sZ",
    ):
        np.testing.assert_allclose(
            getattr(legacy_solution, name).values,
            getattr(fv_solution, name).values,
            equal_nan=True,
        )


def test_koehler1986_reference_state_minimal_smoke_solve():
    pt, pressure, ps, phis = _build_reference_inputs()
    surface_theta = xr.full_like(ps, 170.0, dtype=float)

    solver = Koehler1986ReferenceState()
    assert solver.monotonic_policy == "reject"
    solution = solver.solve(
        pt,
        pressure,
        ps,
        phis=phis,
        surface_potential_temperature=surface_theta,
        theta_levels=[160.0, 180.0, 200.0, 220.0, 240.0],
    )

    assert isinstance(solution, ReferenceStateSolution)
    assert solution.method == "koehler1986_isentropic_surface_iteration"
    assert solver._last_observed_state is not None
    assert solver._last_geometry_state is not None
    assert solution.pi_s is not None
    assert solution.pi_sZ is not None
    assert np.isfinite(solution.pi_s.values).all()
    assert np.isfinite(solution.pi_sZ.values).all()
    assert bool(solution.converged.values.all())
    assert bool(solution.converged_zonal.values.all())
    assert solver._last_observed_state.attrs["monotonic_policy"] == "reject"
    assert solution.monotonic_violations is not None
    assert solution.monotonic_repairs is not None
    assert int(solution.monotonic_violations.max()) == 0
    assert int(solution.monotonic_repairs.max()) == 0


def test_reference_state_solution_rejects_nonconverged_full_curve_evaluation():
    pt, pressure, ps, phis = _build_reference_inputs()
    solution = FiniteVolumeReferenceState().solve(pt, pressure, ps, phis=phis)
    bad_status = xr.DataArray(
        np.asarray([False], dtype=bool),
        dims=("time",),
        coords={"time": solution.converged.coords["time"].values},
        name="reference_state_converged",
    )
    nonconverged = replace(solution, converged=bad_status)

    with pytest.raises(ValueError, match="non-converged reference state"):
        nonconverged.reference_pressure(pt)
    with pytest.raises(ValueError, match="non-converged reference state"):
        nonconverged.efficiency(pt, pressure)


def test_reference_state_solution_rejects_missing_zonal_convergence():
    pt, pressure, ps, phis = _build_reference_inputs()
    solution = FiniteVolumeReferenceState().solve(pt, pressure, ps, phis=phis)
    missing_status = xr.DataArray(
        np.asarray([np.nan], dtype=float),
        dims=("time",),
        coords={"time": solution.converged_zonal.coords["time"].values},
        name="reference_state_converged_zonal",
    )
    nonconverged = replace(solution, converged_zonal=missing_status)
    representative_theta = pt.mean(dim="longitude")
    representative_pressure = pressure.mean(dim="longitude")

    with pytest.raises(ValueError, match="non-converged reference state"):
        nonconverged.zonal_reference_pressure(representative_theta)
    with pytest.raises(ValueError, match="non-converged reference state"):
        nonconverged.zonal_efficiency(representative_theta, representative_pressure)


def test_koehler1986_reference_state_defaults_to_reject_nonmonotonic_profiles():
    pt, pressure, ps, phis, surface_theta = _build_nonmonotonic_reference_inputs()
    solver = Koehler1986ReferenceState(theta_levels=[170.0, 190.0, 210.0, 230.0, 250.0])

    with pytest.raises(ValueError, match="monotonic_policy='reject'"):
        solver.solve(
            pt,
            pressure,
            ps,
            phis=phis,
            surface_potential_temperature=surface_theta,
        )


def test_koehler1986_reference_state_repair_is_explicit_and_recorded():
    pt, pressure, ps, phis, surface_theta = _build_nonmonotonic_reference_inputs()
    solver = Koehler1986ReferenceState(
        theta_levels=[170.0, 190.0, 210.0, 230.0, 250.0],
        monotonic_policy="repair",
        pressure_tolerance=1.0e-6,
        max_iterations=64,
    )

    solution = solver.solve(
        pt,
        pressure,
        ps,
        phis=phis,
        surface_potential_temperature=surface_theta,
    )

    assert solver._last_observed_state.attrs["monotonic_policy"] == "repair"
    assert solution.monotonic_violations is not None
    assert solution.monotonic_repairs is not None
    assert solution.monotonic_violations_zonal is not None
    assert solution.monotonic_repairs_zonal is not None
    assert int(solution.monotonic_violations.max()) == 1
    assert int(solution.monotonic_repairs.max()) == 1
    assert int(solution.monotonic_violations_zonal.max()) == 1
    assert int(solution.monotonic_repairs_zonal.max()) == 1
