from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

import xarray as xr

from mars_exact_lec.boer.closure import four_box_residual_generation_dissipation
from mars_exact_lec.boer.conversions import C_A, C_E, C_K, C_K1, C_K2, C_Z, C_Z1, C_Z2
from mars_exact_lec.boer.reservoirs import (
    A,
    A1,
    A2,
    A_E,
    A_E1,
    A_E2,
    A_Z,
    A_Z1,
    A_Z2,
    kinetic_energy_eddy,
    kinetic_energy_zonal,
    total_horizontal_ke,
)
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.time_derivatives import time_derivative
from mars_exact_lec.common.zonal_ops import weighted_representative_zonal_mean

from .helpers import full_field, make_coords, pressure_field, surface_geopotential, surface_pressure, zonal_field
from .helpers_exact import build_asymmetric_exact_case, build_flat_reference_case


def _closure_series(name, values, units):
    time = xr.DataArray(
        np.asarray([0.0, 1.0, 2.0], dtype=float),
        dims=("time",),
        coords={"time": np.asarray([0.0, 1.0, 2.0], dtype=float)},
        name="time",
        attrs={"units": "hours"},
    )
    term = xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("time",),
        coords={"time": time.values},
        name=name,
        attrs={
            "units": units,
            "normalization": "global_integral",
            "surface_pressure_policy": "raise",
            "domain": "full_model_pressure_domain",
            "not_exact_full_atmosphere": False,
        },
    )
    term.coords["time"].attrs.update(time.attrs)
    return term


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


def test_topographic_reservoir_and_conversion_invariants_close_on_same_case():
    case = build_asymmetric_exact_case(include_reference=True, ntime=3)
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
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

    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(4.0, 18.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-5.0, 8.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.2, 0.3, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="omega",
        units="Pa s-1",
    )
    alpha = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.6, 1.2, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="alpha",
        units="m3 kg-1",
    )
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        900.0
        + 0.2 * pressure.values
        + 20.0 * np.sin(np.deg2rad(longitude.values))[None, None, None, :]
        + 5.0 * np.cos(np.deg2rad(latitude.values))[None, None, :, None],
        name="geopotential",
        units="m2 s-2",
    )

    k = total_horizontal_ke(u, v, theta_mask, integrator, measure=measure)
    kz = kinetic_energy_zonal(u, v, theta_mask, integrator, measure=measure)
    ke = kinetic_energy_eddy(u, v, theta_mask, integrator, measure=measure)
    a = A(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, phis=phis, n=n, pi_s=solution.pi_s)
    az = A_Z(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, phis=phis, n_z=n_z, pi_sZ=solution.pi_sZ)
    ae = A_E(
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
    cz = C_Z(omega, alpha, theta_mask, integrator, measure=measure, ps=ps, phis=phis)
    cz1 = C_Z1(omega, alpha, theta_mask, integrator, measure=measure)
    cz2 = C_Z2(ps, phis, integrator, measure=measure)
    ck = C_K(u, v, omega, theta_mask, integrator, measure=measure, ps=ps, geopotential=geopotential)
    ck1 = C_K1(u, v, omega, theta_mask, integrator, measure=measure)
    ck2 = C_K2(u, v, omega, theta_mask, integrator, measure=measure, ps=ps, geopotential=geopotential)

    np.testing.assert_allclose((kz + ke).values, k.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((az + ae).values, a.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((az1 + ae1).values, a1.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((az2 + ae2).values, a2.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((cz1 + cz2).values, cz.values, rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose((ck1 + ck2).values, ck.values, rtol=1.0e-12, atol=1.0e-10)


def test_available_potential_energy_components_match_independent_small_grid_oracle():
    time, level, latitude, longitude = make_coords(
        ntime=1,
        level_values=[800.0, 400.0],
        nlat=2,
        nlon=4,
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    pressure = pressure_field(time, level, latitude, longitude)
    theta = full_field(time, level, latitude, longitude, 1.0, name="theta", units="1")
    ps_body = surface_pressure(time, latitude, longitude, 1000.0)

    temperature_values = np.asarray(
        [
            [
                [[180.0, 185.0, 195.0, 205.0], [210.0, 220.0, 235.0, 250.0]],
                [[160.0, 170.0, 175.0, 190.0], [200.0, 215.0, 225.0, 240.0]],
            ]
        ],
        dtype=float,
    )
    n_values = np.asarray(
        [
            [
                [[0.050, 0.060, 0.070, 0.080], [0.090, 0.100, 0.110, 0.120]],
                [[0.040, 0.050, 0.060, 0.070], [0.080, 0.090, 0.100, 0.110]],
            ]
        ],
        dtype=float,
    )
    n_z_values = np.asarray([[[0.055, 0.095], [0.045, 0.085]]], dtype=float)
    temperature = full_field(time, level, latitude, longitude, temperature_values, name="temperature", units="K")
    n = full_field(time, level, latitude, longitude, n_values, name="N", units="1")
    n_z = zonal_field(time, level, latitude, n_z_values, name="N_Z")

    ps_surface_values = np.asarray(
        [[[900.0, 910.0, 920.0, 930.0], [880.0, 890.0, 900.0, 910.0]]],
        dtype=float,
    )
    phis_values = np.asarray(
        [[[100.0, 150.0, 200.0, 250.0], [300.0, 350.0, 400.0, 450.0]]],
        dtype=float,
    )
    pi_s_values = np.asarray(
        [[[760.0, 765.0, 770.0, 775.0], [780.0, 785.0, 790.0, 795.0]]],
        dtype=float,
    )
    pi_sZ_values = np.asarray(
        [[[740.0, 740.0, 740.0, 740.0], [770.0, 770.0, 770.0, 770.0]]],
        dtype=float,
    )
    ps_surface = surface_pressure(time, latitude, longitude, ps_surface_values)
    phis = surface_geopotential(time, latitude, longitude, phis_values)
    pi_s = surface_pressure(time, latitude, longitude, pi_s_values)
    pi_sZ = surface_pressure(time, latitude, longitude, pi_sZ_values)

    az1 = A_Z1(temperature, pressure, theta, integrator, ps=ps_body, n_z=n_z)
    ae1 = A_E1(temperature, pressure, theta, integrator, ps=ps_body, n=n, n_z=n_z)
    az2 = A_Z2(ps_surface, phis, integrator, pi_sZ=pi_sZ)
    ae2 = A_E2(ps_surface, phis, integrator, pi_s=pi_s, pi_sZ=pi_sZ)

    mass_zonal = np.asarray(integrator.zonal_mass_weights.values, dtype=float)[None, :, :]
    surface_weights = np.asarray(integrator.surface_weights.values, dtype=float)[None, :, :]
    temperature_r = temperature_values.mean(axis=-1)
    expected_az1 = integrator.constants.cp * np.sum(
        mass_zonal * n_z_values * temperature_r,
        axis=(1, 2),
    )
    expected_ae1 = integrator.constants.cp * np.sum(
        mass_zonal * np.mean((n_values - n_z_values[..., None]) * temperature_values, axis=-1),
        axis=(1, 2),
    )
    expected_az2 = np.sum(
        surface_weights * (ps_surface_values - pi_sZ_values) * phis_values,
        axis=(1, 2),
    )
    expected_ae2 = np.sum(
        surface_weights * (pi_sZ_values - pi_s_values) * phis_values,
        axis=(1, 2),
    )

    for expected in (expected_az1, expected_ae1, expected_az2, expected_ae2):
        assert not np.allclose(expected, 0.0, atol=1.0e-12)

    np.testing.assert_allclose(az1.values, expected_az1, rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(ae1.values, expected_ae1, rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(az2.values, expected_az2, rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(ae2.values, expected_ae2, rtol=1.0e-12, atol=1.0e-8)


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
    ps = case["ps"]
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
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        1024.0,
        name="geopotential",
        units="m2 s-2",
    )

    ke = kinetic_energy_eddy(u, v, theta_mask, integrator, measure=measure)
    ce = C_E(omega, alpha, theta_mask, integrator, measure=measure)
    ca = C_A(temperature, u, v, omega, n_z, theta_mask, integrator, measure=measure)
    ck1 = C_K1(u, v, omega, theta_mask, integrator, measure=measure)
    ck2 = C_K2(u, v, omega, theta_mask, integrator, measure=measure, geopotential=geopotential, ps=ps)
    ck = C_K(u, v, omega, theta_mask, integrator, measure=measure, geopotential=geopotential, ps=ps)

    np.testing.assert_allclose(ke.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ce.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ca.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck1.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck.values, 0.0, rtol=0.0, atol=1.0e-12)


def test_no_eddy_limit_zeroes_all_eddy_and_zonal_to_eddy_terms_on_flat_case():
    case = build_flat_reference_case(
        ntime=3,
        ps_value=1000.0,
        phis_value=2000.0,
        theta_profile=(210.0, 230.0, 250.0),
    )
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    measure = case["measure"]
    ps = case["ps"]
    phis = case["phis"]
    temperature = case["temperature"]
    solution = case["solution"]

    shape = (time.size, level.size, latitude.size)
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(4.0, 16.0, np.prod(shape)).reshape(shape)[..., None],
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-2.0, 5.0, np.prod(shape)).reshape(shape)[..., None],
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.2, 0.3, np.prod(shape)).reshape(shape)[..., None],
        name="omega",
        units="Pa s-1",
    )
    alpha = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.7, 1.1, np.prod(shape)).reshape(shape)[..., None],
        name="alpha",
        units="m3 kg-1",
    )
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(100.0, 1000.0, np.prod(shape)).reshape(shape)[..., None],
        name="geopotential",
        units="m2 s-2",
    )
    n_z = solution.zonal_efficiency(case["representative_theta"], case["representative_pressure"])

    ke = kinetic_energy_eddy(u, v, theta_mask, integrator, measure=measure)
    ae = A_E(
        temperature,
        pressure,
        theta_mask,
        integrator,
        measure=measure,
        reference_state=solution,
        ps=ps,
        phis=phis,
    )
    ce = C_E(omega, alpha, theta_mask, integrator, measure=measure)
    ca = C_A(temperature, u, v, omega, n_z, theta_mask, integrator, measure=measure)
    ck2 = C_K2(u, v, omega, theta_mask, integrator, measure=measure, geopotential=geopotential, ps=ps)
    ck = C_K(u, v, omega, theta_mask, integrator, measure=measure, geopotential=geopotential, ps=ps)

    np.testing.assert_allclose(ke.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ae.values, 0.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(ce.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ca.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck.values, 0.0, rtol=0.0, atol=1.0e-12)


def test_flat_surface_limit_zeroes_surface_and_ck2_topographic_terms():
    case = build_flat_reference_case(
        ntime=3,
        ps_value=1000.0,
        phis_value=0.0,
        theta_profile=(210.0, 230.0, 250.0),
    )
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    temperature = case["temperature"]
    theta_mask = case["theta_mask"]
    integrator = case["integrator"]
    measure = case["measure"]
    ps = case["ps"]
    phis = case["phis"]
    solution = case["solution"]

    u = full_field(time, level, latitude, longitude, 3.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, -2.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.1, name="omega", units="Pa s-1")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha", units="m3 kg-1")
    geopotential = full_field(time, level, latitude, longitude, 500.0, name="geopotential", units="m2 s-2")

    az2 = A_Z2(ps, phis, integrator, measure=measure, pi_sZ=solution.pi_sZ)
    ae2 = A_E2(ps, phis, integrator, measure=measure, pi_s=solution.pi_s, pi_sZ=solution.pi_sZ)
    a2 = A2(ps, phis, integrator, measure=measure, pi_s=solution.pi_s)
    cz2 = C_Z2(ps, phis, integrator, measure=measure)
    ck2 = C_K2(u, v, omega, theta_mask, integrator, measure=measure, ps=ps, geopotential=geopotential)
    ck = C_K(u, v, omega, theta_mask, integrator, measure=measure, ps=ps, geopotential=geopotential)
    ck1 = C_K1(u, v, omega, theta_mask, integrator, measure=measure)

    np.testing.assert_allclose(az2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ae2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(a2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(cz2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ck2.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose((ck - ck1).values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        (
            A_Z(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, phis=phis, reference_state=solution)
            - A_Z1(temperature, pressure, theta_mask, integrator, measure=measure, ps=ps, reference_state=solution)
        ).values,
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        (C_Z(omega, alpha, theta_mask, integrator, measure=measure, ps=ps, phis=phis) - C_Z1(omega, alpha, theta_mask, integrator, measure=measure)).values,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_four_box_residual_identity_matches_total_storage_tendency():
    az = _closure_series("A_Z", [10.0, 13.0, 18.0], "J")
    ae = _closure_series("A_E", [6.0, 8.0, 11.0], "J")
    kz = _closure_series("K_Z", [20.0, 25.0, 31.0], "J")
    ke = _closure_series("K_E", [5.0, 4.0, 6.0], "J")
    cz = _closure_series("C_Z", [2.0, 2.5, 3.0], "W")
    ca = _closure_series("C_A", [0.5, 1.0, 1.5], "W")
    ce = _closure_series("C_E", [1.5, 1.0, 0.5], "W")
    ck = _closure_series("C_K", [0.25, 0.5, 0.75], "W")

    diagnostics = four_box_residual_generation_dissipation(az, ae, kz, ke, cz, ca, ce, ck)
    total_storage_tendency = time_derivative(az + ae + kz + ke)

    np.testing.assert_allclose(
        (diagnostics["G_Z"] + diagnostics["G_E"] - diagnostics["F_Z"] - diagnostics["F_E"]).values,
        total_storage_tendency.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(diagnostics["G_Z"].values, (diagnostics["dA_Z_dt"] + cz + ca).values)
    np.testing.assert_allclose(diagnostics["G_E"].values, (diagnostics["dA_E_dt"] + ce - ca).values)
    np.testing.assert_allclose(diagnostics["F_Z"].values, (cz - ck - diagnostics["dK_Z_dt"]).values)
    np.testing.assert_allclose(diagnostics["F_E"].values, (ce + ck - diagnostics["dK_E_dt"]).values)


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
