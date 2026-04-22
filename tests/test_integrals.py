from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

import mars_exact_lec.common.integrals as mass_integrals
from mars_exact_lec.common.integrals import (
    build_mass_integrator,
    delta_p,
    integrate_mass_full,
    integrate_mass_zonal,
    pressure_level_edges,
)
from mars_exact_lec.constants_mars import MARS

from .helpers import full_field, make_coords, pressure_field, zonal_field


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


def test_pressure_level_edges_and_delta_p_use_explicit_bounds_when_available(monkeypatch):
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
    monkeypatch.setattr(mass_integrals, "_get_explicit_level_bounds", lambda _: bounds)

    np.testing.assert_allclose(pressure_level_edges(level).values, np.asarray([980.0, 700.0, 420.0, 60.0]))
    np.testing.assert_allclose(delta_p(level).values, np.asarray([280.0, 280.0, 360.0]))
