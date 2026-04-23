from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.reference_state import (
    FiniteVolumeReferenceState,
    Koehler1986ReferenceState,
    KoehlerReferenceState,
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


def test_reference_state_public_exports_include_phase01_classes():
    assert KoehlerReferenceState is FiniteVolumeReferenceState
    assert FiniteVolumeReferenceState is not None
    assert Koehler1986ReferenceState is not None
    assert ReferenceStateSolution is not None


def test_reference_state_koehler_solver_shim_reexports_legacy_private_helper():
    assert _solve_reference_family is not None


def test_finite_volume_reference_state_matches_legacy_alias_and_sets_metadata():
    pt, pressure, ps, phis = _build_reference_inputs()

    legacy_solution = KoehlerReferenceState().solve(pt, pressure, ps, phis=phis)
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
