from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.common.integrals import build_mass_integrator, integrate_mass_zonal
from mars_exact_lec.common.zonal_ops import representative_eddy, representative_zonal_mean, theta_coverage, zonal_mean
from mars_exact_lec.io.mask_below_ground import make_theta

from .helpers import full_field, make_coords, pressure_field, surface_pressure


def test_representative_mean_reduces_to_plain_zonal_mean_when_fully_above_ground():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, 900.0))
    field = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="field",
    )

    np.testing.assert_allclose(
        representative_zonal_mean(field, theta).values,
        zonal_mean(field).values,
    )


def test_zero_coverage_ring_uses_safe_fill_and_integrates_to_zero():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.full((time.size, latitude.size, longitude.size), 900.0)
    ps_values[:, -1, :] = 100.0
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, ps_values))
    field = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="field",
    )
    coverage = theta_coverage(theta)
    rep_mean = representative_zonal_mean(field, theta)
    integrator = build_mass_integrator(level, latitude, longitude)

    assert np.allclose(coverage.isel(latitude=-1).values, 0.0)
    assert np.all(np.isfinite(rep_mean.isel(latitude=-1).values))
    zero_ring = (coverage * rep_mean).where(coverage == 0.0, 0.0)
    assert np.allclose(integrate_mass_zonal(zero_ring, integrator=integrator).values, 0.0)


def test_theta_weighted_representative_eddy_has_zero_zonal_mean():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [900.0, 700.0, 500.0, 300.0],
            [900.0, 700.0, 500.0, 300.0],
            [900.0, 700.0, 500.0, 300.0],
            [900.0, 700.0, 500.0, 300.0],
        ]
    )
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, ps_values))
    field = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="field",
    )
    eddy = representative_eddy(field, theta)
    np.testing.assert_allclose(zonal_mean(theta * eddy).values, 0.0, atol=1e-12)
