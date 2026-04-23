from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.conversions import C_A, C_E, C_K1
from mars_exact_lec.boer.reservoirs import A, A1, A2, A_E, A_E1, A_E2, A_Z, A_Z1, A_Z2, kinetic_energy_eddy
from mars_exact_lec.common.zonal_ops import weighted_representative_zonal_mean

from .helpers import full_field
from .helpers_exact import build_asymmetric_exact_case


def test_available_potential_energy_partition_closes_with_explicit_supplied_solution_fields():
    case = build_asymmetric_exact_case(include_reference=True, ntime=2)
    pressure = case["pressure"]
    temperature = case["temperature"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    measure = case["measure"]
    ps = case["ps"]
    phis = case["phis"]
    pt = case["potential_temperature"]
    solution = case["solution"]

    representative_theta = weighted_representative_zonal_mean(pt, measure.cell_fraction)
    representative_pressure = weighted_representative_zonal_mean(pressure, measure.cell_fraction)
    n = solution.efficiency(pt, pressure)
    n_z = solution.zonal_efficiency(representative_theta, representative_pressure)

    a_total = A(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, phis=phis, n=n, pi_s=solution.pi_s)
    a_z = A_Z(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, phis=phis, n_z=n_z, pi_sZ=solution.pi_sZ)
    a_e = A_E(
        temperature,
        pressure,
        theta_mask,
        integrator,
        measure=measure,
        ps=ps,
        phis=phis,
        n=n,
        n_z=n_z,
        pi_s=solution.pi_s,
        pi_sZ=solution.pi_sZ,
    )
    a1 = A1(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, n=n)
    az1 = A_Z1(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, n_z=n_z)
    ae1 = A_E1(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, n=n, n_z=n_z)
    a2 = A2(ps, phis, integrator, measure=measure, pi_s=solution.pi_s)
    az2 = A_Z2(ps, phis, integrator, measure=measure, pi_sZ=solution.pi_sZ)
    ae2 = A_E2(ps, phis, integrator, measure=measure, pi_s=solution.pi_s, pi_sZ=solution.pi_sZ)

    np.testing.assert_allclose((a_z + a_e).values, a_total.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((az1 + ae1).values, a1.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((az2 + ae2).values, a2.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((a1 + a2).values, a_total.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((az1 + az2).values, a_z.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((ae1 + ae2).values, a_e.values, rtol=1.0e-12, atol=1.0e-10)
    assert not np.allclose(a2.values, 0.0, atol=1.0e-12)


def test_no_eddy_limit_zeroes_ke_ce_ca_ck1_on_same_case():
    case = build_asymmetric_exact_case(include_reference=True, ntime=2)
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    measure = case["measure"]
    pt = case["potential_temperature"]
    solution = case["solution"]

    representative_theta = weighted_representative_zonal_mean(pt, measure.cell_fraction)
    representative_pressure = weighted_representative_zonal_mean(pressure, measure.cell_fraction)
    n_z = solution.zonal_efficiency(representative_theta, representative_pressure)

    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(5.0, 20.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
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
            time.size,
            level.size,
            latitude.size,
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
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="omega",
        units="Pa s-1",
    )
    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(180.0, 240.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="temperature",
        units="K",
    )
    alpha = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.7, 1.1, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="alpha",
        units="m3 kg-1",
    )

    ke = kinetic_energy_eddy(u, v, theta_mask, integrator, measure=measure)
    ce = C_E(omega, alpha, theta_mask, integrator, measure=measure)
    ca = C_A(temperature, u, v, omega, n_z, theta_mask, integrator, measure=measure)
    ck1 = C_K1(u, v, omega, theta_mask, integrator, measure=measure)

    np.testing.assert_allclose(ke.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ce.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ca.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck1.values, 0.0, rtol=0.0, atol=1.0e-12)


def test_exact_ape_body_terms_reject_non_coordinate_pressure_fields():
    case = build_asymmetric_exact_case(include_reference=True, ntime=1)
    pressure = case["pressure"].copy(deep=True)
    pressure.loc[dict(longitude=pressure.coords["longitude"].values[0])] = (
        pressure.sel(longitude=pressure.coords["longitude"].values[0]) + 5.0
    )

    with pytest.raises(ValueError, match="pressure-coordinate level broadcast"):
        A1(
            case["temperature"],
            pressure,
            case["theta_mask"],
            case["integrator"],
            measure=case["measure"],
            ps=case["ps"],
            reference_state=case["solution"],
        )
    with pytest.raises(ValueError, match="pressure-coordinate level broadcast"):
        A_Z1(
            case["temperature"],
            pressure,
            case["theta_mask"],
            case["integrator"],
            measure=case["measure"],
            ps=case["ps"],
            reference_state=case["solution"],
        )
    with pytest.raises(ValueError, match="pressure-coordinate level broadcast"):
        A_E1(
            case["temperature"],
            pressure,
            case["theta_mask"],
            case["integrator"],
            measure=case["measure"],
            ps=case["ps"],
            reference_state=case["solution"],
        )
