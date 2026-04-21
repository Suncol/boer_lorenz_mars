from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.io.mask_below_ground import apply_below_ground_mask, make_below_ground_mask, make_theta

from .helpers import full_field, make_coords, pressure_field, surface_pressure


def test_theta_full_above_ground_is_one():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta = make_theta(pressure, ps)
    np.testing.assert_allclose(theta.values, 1.0)


def test_theta_full_below_ground_is_zero():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 100.0)
    theta = make_theta(pressure, ps)
    np.testing.assert_allclose(theta.values, 0.0)


def test_theta_partial_truncation_and_equal_pressure_exclusion():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [800.0, 500.0, 300.0, 250.0],
            [800.0, 500.0, 300.0, 250.0],
            [800.0, 500.0, 300.0, 250.0],
            [800.0, 500.0, 300.0, 250.0],
        ]
    )
    ps = surface_pressure(time, latitude, longitude, ps_values)
    theta = make_theta(pressure, ps)

    assert float(theta.isel(time=0, level=0, latitude=0, longitude=0)) == 1.0
    assert float(theta.isel(time=0, level=0, latitude=0, longitude=1)) == 0.0
    assert float(theta.isel(time=0, level=1, latitude=0, longitude=1)) == 0.0
    assert float(theta.isel(time=0, level=2, latitude=0, longitude=2)) == 0.0


def test_below_ground_mask_and_apply_mask():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 450.0)
    theta = make_theta(pressure, ps)
    mask = make_below_ground_mask(pressure, ps)
    field = full_field(time, level, latitude, longitude, 3.0, name="field")
    masked = apply_below_ground_mask(field, theta)

    np.testing.assert_array_equal(mask.values, theta.values == 0.0)
    assert masked.isnull().isel(time=0, level=0).all()
    assert masked.notnull().isel(time=0, level=2).all()
