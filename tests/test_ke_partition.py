from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.reservoirs import kinetic_energy_eddy, kinetic_energy_zonal, total_horizontal_ke
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.io.mask_below_ground import make_theta

from .helpers import full_field, make_coords, pressure_field, surface_pressure


@pytest.mark.parametrize("ps_value", [900.0, None])
def test_ke_partition_closes_with_and_without_topographic_truncation(ps_value):
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    if ps_value is None:
        ps_values = np.asarray(
            [
                [900.0, 650.0, 450.0, 250.0],
                [900.0, 650.0, 450.0, 250.0],
                [900.0, 650.0, 450.0, 250.0],
                [900.0, 650.0, 450.0, 250.0],
            ]
        )
    else:
        ps_values = ps_value

    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, ps_values))
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(1.0, 4.0, time.size * level.size * latitude.size * longitude.size).reshape(
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
        np.linspace(0.5, 2.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="v",
        units="m s-1",
    )
    integrator = build_mass_integrator(level, latitude, longitude)

    total = total_horizontal_ke(u, v, theta, integrator)
    kz = kinetic_energy_zonal(u, v, theta, integrator)
    ke = kinetic_energy_eddy(u, v, theta, integrator)

    np.testing.assert_allclose((kz + ke).values, total.values)
