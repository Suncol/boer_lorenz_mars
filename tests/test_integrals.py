from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.integrals import (
    build_mass_integrator,
    delta_p,
    integrate_mass_full,
    integrate_surface,
    integrate_mass_zonal,
    pressure_level_edges,
)
from mars_exact_lec.constants_mars import MARS

from .helpers import full_field, make_coords, pressure_field, surface_geopotential, zonal_field


def test_integrate_mass_full_of_one_matches_implied_mass():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    integrator = build_mass_integrator(level, latitude, longitude)
    ones = xr.ones_like(pressure)

    result = integrate_mass_full(ones, integrator=integrator)
    expected = 4.0 * np.pi * MARS.a**2 * float(delta_p(level).sum()) / MARS.g

    np.testing.assert_allclose(result.values, expected)


def test_full_and_zonal_mass_integrals_agree_for_zonally_symmetric_field():
    time, level, latitude, longitude = make_coords()
    integrator = build_mass_integrator(level, latitude, longitude)

    zonal = zonal_field(
        time,
        level,
        latitude,
        np.arange(time.size * level.size * latitude.size).reshape(time.size, level.size, latitude.size),
        name="zonal",
    )
    full = zonal.expand_dims(longitude=longitude).transpose("time", "level", "latitude", "longitude")

    full_result = integrate_mass_full(full, integrator=integrator)
    zonal_result = integrate_mass_zonal(zonal, integrator=integrator)

    np.testing.assert_allclose(full_result.values, zonal_result.values)


def test_pressure_level_edges_and_delta_p_use_explicit_bounds_when_available():
    level = xr.DataArray(
        np.asarray([820.0, 560.0, 240.0]),
        dims=("level",),
        coords={"level": [820.0, 560.0, 240.0]},
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
        name="level",
    )
    bounds = xr.DataArray(
        np.asarray([[980.0, 700.0], [700.0, 420.0], [420.0, 60.0]]),
        dims=("level", "bounds"),
        coords={"level": level.values, "bounds": [0, 1]},
    )
    np.testing.assert_allclose(
        pressure_level_edges(level, bounds=bounds).values,
        np.asarray([980.0, 700.0, 420.0, 60.0]),
    )
    np.testing.assert_allclose(delta_p(level, bounds=bounds).values, np.asarray([280.0, 280.0, 360.0]))

    _, _, latitude, longitude = make_coords()
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=bounds)
    np.testing.assert_allclose(integrator.delta_p.values, np.asarray([280.0, 280.0, 360.0]))


def test_integrate_surface_of_one_matches_global_area_over_g():
    time, _, latitude, longitude = make_coords()
    integrator = build_mass_integrator(
        xr.DataArray(
            np.asarray([700.0, 500.0, 300.0]),
            dims=("level",),
            coords={"level": [700.0, 500.0, 300.0]},
        ),
        latitude,
        longitude,
    )
    ones = surface_geopotential(time, latitude, longitude, 1.0, name="ones")

    result = integrate_surface(ones, integrator=integrator)
    expected = 4.0 * np.pi * MARS.a**2 / MARS.g

    np.testing.assert_allclose(result.values, expected)


def test_mass_integrator_rejects_same_shape_coordinate_mismatch():
    time, level, latitude, longitude = make_coords()
    integrator = build_mass_integrator(level, latitude, longitude)
    bad_full = full_field(time, level, latitude, longitude, 1.0, name="bad_full").assign_coords(
        longitude=longitude.values + 0.5
    )
    bad_surface = surface_geopotential(time, latitude, longitude, 1.0, name="bad_surface").assign_coords(
        longitude=longitude.values + 0.5
    )

    with pytest.raises(ValueError, match="longitude"):
        integrator.integrate_full(bad_full)
    with pytest.raises(ValueError, match="longitude"):
        integrator.integrate_surface(bad_surface)
