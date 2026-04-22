from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.reservoirs import total_horizontal_ke
from mars_exact_lec.common.integrals import build_mass_integrator, delta_p
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.validation import _resolve_live_seba_energy_budget, seba_total_hke_per_level

from .helpers import full_field, make_coords, pressure_field, seba_dataset, surface_pressure, surface_pressure_policy_for_case

pytestmark = pytest.mark.live_seba


@pytest.fixture(scope="module")
def live_seba_runtime():
    try:
        return _resolve_live_seba_energy_budget()
    except ImportError as exc:
        pytest.skip(str(exc))


def test_seba_hke_matches_direct_grid_hke_and_column_integral(live_seba_runtime):
    assert live_seba_runtime.__name__ == "EnergyBudget"

    time, level, latitude, longitude = make_coords(grid="regular")
    nlat = 32
    nlon = 64
    latitude = xr.DataArray(
        np.linspace(90.0 - 90.0 / nlat, -90.0 + 90.0 / nlat, nlat),
        dims=("latitude",),
        coords={"latitude": np.linspace(90.0 - 90.0 / nlat, -90.0 + 90.0 / nlat, nlat)},
        name="latitude",
        attrs={"units": "degrees_north", "axis": "Y", "standard_name": "latitude"},
    )
    longitude = xr.DataArray(
        np.arange(0.0, 360.0, 360.0 / nlon),
        dims=("longitude",),
        coords={"longitude": np.arange(0.0, 360.0, 360.0 / nlon)},
        name="longitude",
        attrs={"units": "degrees_east", "axis": "X", "standard_name": "longitude"},
    )
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta = make_theta(pressure, ps)

    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(1.0, 4.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="u_wind",
        units="m/s",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.5, 2.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="v_wind",
        units="m/s",
    )
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega", units="Pa/s")
    temperature = full_field(time, level, latitude, longitude, 210.0, name="temperature", units="K")
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        0.0,
        name="geopotential",
        units="m**2 s**-2",
    )
    surface_temperature = xr.DataArray(
        np.full((time.size, latitude.size, longitude.size), 210.0),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time.values,
            "latitude": latitude.values,
            "longitude": longitude.values,
        },
        name="ts",
        attrs={"units": "K", "standard_name": "surface_temperature"},
    )

    dataset = seba_dataset(time, level, latitude, longitude, u, v, omega, temperature)
    dataset["geopotential"] = geopotential
    dataset["ts"] = surface_temperature
    hke_level = seba_total_hke_per_level(dataset, p_levels=level.values, ps=ps, rsphere=MARS.a)

    areas = 4.0 * np.pi * MARS.a**2
    direct_hke_level = 0.5 * (u**2 + v**2).mean(dim="longitude")
    direct_hke_level = direct_hke_level.weighted(np.cos(np.deg2rad(latitude))).mean(dim="latitude")
    np.testing.assert_allclose(hke_level.values, direct_hke_level.values, rtol=5e-4, atol=1e-6)

    integrator = build_mass_integrator(level, latitude, longitude)
    policy = surface_pressure_policy_for_case(ps, level)
    direct_total = total_horizontal_ke(u, v, theta, integrator, ps=ps, surface_pressure_policy=policy)
    seba_column_total = (hke_level * delta_p(level) / MARS.g).sum(dim="level") * areas

    np.testing.assert_allclose(seba_column_total.values, direct_total.values, rtol=5e-4, atol=1e-4)
