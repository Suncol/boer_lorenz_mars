from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.conversions import (
    conversion_eddy_ape_to_ke,
    conversion_zonal_ape_to_eddy_ape,
    conversion_zonal_ape_to_ke_part1,
)
from mars_exact_lec.boer.reservoirs import kinetic_energy_eddy
from mars_exact_lec.common.integrals import build_mass_integrator, integrate_mass_full
from mars_exact_lec.io.mask_below_ground import make_theta

from .helpers import full_field, make_coords, pressure_field, surface_pressure, temperature_from_theta_values, zonal_field


def test_eddy_free_fields_give_zero_ke_and_ce():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, 900.0))
    integrator = build_mass_integrator(level, latitude, longitude)

    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size).reshape(time.size, level.size, latitude.size)[..., None],
        name="u",
    )
    v = full_field(time, level, latitude, longitude, 2.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha")

    np.testing.assert_allclose(kinetic_energy_eddy(u, v, theta, integrator).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        conversion_eddy_ape_to_ke(omega, alpha, theta, integrator).values,
        0.0,
        atol=1e-12,
    )


def test_conversion_partition_closes_total_theta_omega_alpha_term():
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
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, ps_values))
    integrator = build_mass_integrator(level, latitude, longitude)

    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.5, 0.5, time.size * level.size * latitude.size * longitude.size).reshape(
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
        np.linspace(0.5, 1.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="alpha",
        units="m3 kg-1",
    )

    ce = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator)
    cz1 = conversion_zonal_ape_to_ke_part1(omega, alpha, theta, integrator)
    total = -integrate_mass_full(theta * omega * alpha, integrator=integrator)

    np.testing.assert_allclose((ce + cz1).values, total.values)


def test_flat_surface_theta_and_zonal_mean_reduce_cleanly():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, 900.0))
    np.testing.assert_allclose(theta.values, 1.0)


def test_conversion_zonal_ape_to_eddy_ape_rejects_longitude_varying_full_nz():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, 900.0))
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    n_z = zonal_field(time, level, latitude, 1.0, name="N_Z")
    n_z_broadcast = n_z.expand_dims(longitude=longitude).transpose("time", "level", "latitude", "longitude")
    n_z_bad = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size * longitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
            longitude.size,
        ),
        name="N_Z_bad",
    )

    zonal_result = conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z, theta, integrator)
    broadcast_result = conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z_broadcast, theta, integrator)

    np.testing.assert_allclose(zonal_result.values, broadcast_result.values)
    with pytest.raises(ValueError, match="longitude-varying"):
        conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z_bad, theta, integrator)
