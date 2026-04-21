from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.conversions import C_A, C_K1
from mars_exact_lec.boer.reservoirs import A, A_E, A_Z
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.zonal_ops import representative_zonal_mean
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import KoehlerReferenceState, potential_temperature

from .helpers import full_field, make_coords, pressure_field, surface_pressure


def test_reference_state_reproduces_stable_flat_reference_and_zero_ape():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    theta_profile = np.asarray([180.0, 200.0, 220.0])
    temperature_profile = theta_profile * (level.values / MARS.p00) ** MARS.kappa
    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        temperature_profile[None, :, None, None],
        name="temperature",
        units="K",
    )

    pt = potential_temperature(temperature, pressure)
    solution = KoehlerReferenceState().solve(pt, pressure, ps)
    pi = solution.reference_pressure(pt)

    np.testing.assert_allclose(pi.values, pressure.values, atol=1e-12)
    np.testing.assert_allclose(A(temperature, pressure, theta_mask, integrator, reference_state=solution).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(A_Z(temperature, pressure, theta_mask, integrator, reference_state=solution).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(A_E(temperature, pressure, theta_mask, integrator, reference_state=solution).values, 0.0, atol=1e-12)


def test_reference_state_preserves_total_mass_and_monotonic_reference_curve():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [900.0, 650.0, 450.0, 250.0],
            [900.0, 650.0, 450.0, 250.0],
            [900.0, 650.0, 450.0, 250.0],
            [900.0, 650.0, 450.0, 250.0],
        ]
    )
    ps = surface_pressure(time, latitude, longitude, ps_values)
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
    solution = KoehlerReferenceState().solve(pt, pressure, ps)

    direct_mass = integrator.integrate_full(theta_mask)
    np.testing.assert_allclose(solution.total_mass.values, direct_mass.values)

    planetary_area = float(integrator.cell_area.sum())
    reconstructed_mass = (
        (solution.reference_surface_pressure - solution.reference_top_pressure) * planetary_area / MARS.g
    )
    np.testing.assert_allclose(reconstructed_mass.values, solution.total_mass.values)

    for time_index in range(time.size):
        theta_ref = solution.theta_reference.isel(time=time_index).values
        pi_ref = solution.pi_reference.isel(time=time_index).values
        valid = np.isfinite(theta_ref) & np.isfinite(pi_ref)
        assert np.all(np.diff(theta_ref[valid]) > 0.0)
        assert np.all(np.diff(pi_ref[valid]) < 0.0)


def test_available_potential_energy_body_partition_closes():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [900.0, 700.0, 500.0, 350.0],
            [900.0, 700.0, 500.0, 350.0],
            [900.0, 700.0, 500.0, 350.0],
            [900.0, 700.0, 500.0, 350.0],
        ]
    )
    ps = surface_pressure(time, latitude, longitude, ps_values)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

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
    solution = KoehlerReferenceState().solve(pt, pressure, ps)

    a_total = A(temperature, pressure, theta_mask, integrator, reference_state=solution)
    a_z = A_Z(temperature, pressure, theta_mask, integrator, reference_state=solution)
    a_e = A_E(temperature, pressure, theta_mask, integrator, reference_state=solution)

    np.testing.assert_allclose((a_z + a_e).values, a_total.values)


def test_ca_and_ck1_vanish_without_longitudinal_eddies():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(180.0, 230.0, time.size * level.size * latitude.size).reshape(
            time.size, level.size, latitude.size
        )[..., None],
        name="temperature",
        units="K",
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
    solution = KoehlerReferenceState().solve(pt, pressure, ps)
    n_z = solution.zonal_efficiency(
        representative_zonal_mean(pt, theta_mask),
        representative_zonal_mean(pressure, theta_mask),
    )

    np.testing.assert_allclose(
        C_A(temperature, u, v, omega, n_z, theta_mask, integrator).values,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        C_K1(u, v, omega, theta_mask, integrator).values,
        0.0,
        atol=1e-12,
    )
