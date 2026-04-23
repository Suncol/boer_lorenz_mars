from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.constants_mars import MARS

from .helpers import surface_geopotential, surface_mass_from_pi_s
from .helpers_exact import build_flat_reference_case


EXPECTED_CONTRACT = {
    "fv": {
        "method": "finite_volume_parcel_sorted",
        "reference_coordinate_semantics": "finite_volume_theta_groups",
        "reference_pressure_sampling": "half_mass_pressure_sample",
        "reference_curve_interpolation_space": "pressure",
        "flat_pressure_atol": 1.0e-10,
        "surface_atol": 1.0e-10,
        "solve_kwargs": {},
    },
    "k86": {
        "method": "koehler1986_isentropic_surface_iteration",
        "reference_coordinate_semantics": "fixed_isentropic_levels",
        "reference_pressure_sampling": "fixed_isentropic_level_pressure",
        "reference_curve_interpolation_space": "exner",
        "flat_pressure_atol": 1.0e-8,
        "surface_atol": 1.0e-8,
        "solve_kwargs": {
            "surface_theta_value": 180.0,
            "theta_levels": [180.0, 190.0, 210.0, 230.0, 250.0],
            "pressure_tolerance": 1.0e-8,
            "max_iterations": 80,
        },
    },
}


def _flat_case(solver_kind: str, *, phis_value: float = 0.0):
    contract = EXPECTED_CONTRACT[solver_kind]
    return build_flat_reference_case(
        solver_kind=solver_kind,
        ntime=1,
        level_values=(900.0, 700.0, 500.0),
        ps_value=1000.0,
        phis_value=phis_value,
        theta_profile=(190.0, 210.0, 230.0),
        **contract["solve_kwargs"],
    )


@pytest.mark.parametrize("solver_kind", ["fv", "k86"])
def test_reference_state_public_flat_reference_contract(solver_kind):
    case = _flat_case(solver_kind)
    contract = EXPECTED_CONTRACT[solver_kind]
    solution = case["solution"]
    pressure = case["pressure"]
    pt = case["potential_temperature"]

    np.testing.assert_allclose(
        solution.reference_pressure(pt, pressure=pressure).values,
        pressure.values,
        rtol=0.0,
        atol=contract["flat_pressure_atol"],
    )
    assert np.isfinite(solution.pi_s.values).all()
    assert np.isfinite(solution.pi_sZ.values).all()
    assert float(solution.pi_s.min()) > 0.0
    assert float(solution.pi_sZ.min()) > 0.0
    for name in ("reference_surface_pressure", "reference_bottom_pressure", "reference_top_pressure"):
        field = getattr(solution, name)
        assert np.isfinite(field.values).all()
        assert float(field.min()) > 0.0
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


@pytest.mark.parametrize("solver_kind", ["fv", "k86"])
def test_reference_state_public_efficiency_formula_consistency(solver_kind):
    case = _flat_case(solver_kind)
    solution = case["solution"]
    pressure = case["pressure"]
    pt = case["potential_temperature"]
    representative_theta = case["representative_theta"]
    representative_pressure = case["representative_pressure"]

    pi = solution.reference_pressure(pt, pressure=pressure)
    n = solution.efficiency(pt, pressure)
    np.testing.assert_allclose(n.values, (1.0 - (pi / pressure) ** MARS.kappa).values, atol=1.0e-12)

    pi_z = solution.zonal_reference_pressure(representative_theta, pressure=representative_pressure)
    n_z = solution.zonal_efficiency(representative_theta, representative_pressure)
    np.testing.assert_allclose(
        n_z.values,
        (1.0 - (pi_z / representative_pressure) ** MARS.kappa).values,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("solver_kind", ["fv", "k86"])
def test_reference_state_public_coordinate_guards(solver_kind):
    case = _flat_case(solver_kind)
    solution = case["solution"]
    pressure = case["pressure"]
    pt = case["potential_temperature"]
    representative_theta = case["representative_theta"]
    representative_pressure = case["representative_pressure"]
    bad_full_pressure = pressure.assign_coords(longitude=pressure.coords["longitude"].values + 0.5)
    bad_zonal_pressure = representative_pressure.assign_coords(
        latitude=representative_pressure.coords["latitude"].values + 0.5
    )

    with pytest.raises(ValueError, match="share the same dims"):
        solution.reference_pressure(pt, pressure=representative_pressure)
    with pytest.raises(ValueError, match="Coordinate 'longitude'"):
        solution.reference_pressure(pt, pressure=bad_full_pressure)
    with pytest.raises(ValueError, match="Coordinate 'latitude'"):
        solution.zonal_reference_pressure(representative_theta, pressure=bad_zonal_pressure)
    with pytest.raises(ValueError, match="Coordinate 'latitude'"):
        solution.zonal_efficiency(representative_theta, bad_zonal_pressure)


@pytest.mark.parametrize("solver_kind", ["fv", "k86"])
def test_reference_state_public_metadata_and_domain_attrs(solver_kind):
    case = _flat_case(solver_kind)
    contract = EXPECTED_CONTRACT[solver_kind]
    solution = case["solution"]

    assert solution.method == contract["method"]
    assert solution.theta_reference.attrs["reference_coordinate_semantics"] == contract["reference_coordinate_semantics"]
    assert solution.pi_reference.attrs["reference_pressure_sampling"] == contract["reference_pressure_sampling"]
    assert (
        solution.pi_reference.attrs.get("reference_curve_interpolation_space", "pressure")
        == contract["reference_curve_interpolation_space"]
    )
    for name in ("ps_effective", "pi_s", "pi_sZ", "reference_surface_pressure", "reference_bottom_pressure"):
        field = getattr(solution, name)
        assert field.attrs["surface_pressure_policy"] == case["policy"]
        assert "domain" in field.attrs


@pytest.mark.parametrize("solver_kind", ["fv", "k86"])
def test_reference_state_public_flat_topography_constant_offset_invariance(solver_kind):
    contract = EXPECTED_CONTRACT[solver_kind]
    flat = _flat_case(solver_kind, phis_value=0.0)
    shifted = _flat_case(solver_kind, phis_value=2000.0)
    pt = flat["potential_temperature"]
    pressure = flat["pressure"]

    for name in ("pi_reference", "pi_s", "pi_sZ", "reference_surface_pressure", "reference_bottom_pressure"):
        np.testing.assert_allclose(
            getattr(flat["solution"], name).values,
            getattr(shifted["solution"], name).values,
            rtol=0.0,
            atol=contract["surface_atol"],
        )
    np.testing.assert_allclose(
        flat["solution"].reference_pressure(pt, pressure=pressure).values,
        shifted["solution"].reference_pressure(pt, pressure=pressure).values,
        rtol=0.0,
        atol=contract["flat_pressure_atol"],
    )
