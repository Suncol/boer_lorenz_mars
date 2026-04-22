from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import (
    interpolate_pressure_to_isentropes_metadata,
    isentropic_interfaces,
    isentropic_layer_mass_statistics,
    potential_temperature,
)

from .helpers import full_field, make_coords, pressure_field, surface_pressure


def _stable_temperature(time, level, latitude, longitude, theta_profile: np.ndarray) -> xr.DataArray:
    return full_field(
        time,
        level,
        latitude,
        longitude,
        theta_profile[None, :, None, None] * (level.values / MARS.p00)[None, :, None, None] ** MARS.kappa,
        name="temperature",
        units="K",
    )


def _column_sample(field: xr.DataArray) -> xr.DataArray:
    sample = field.isel(time=0, latitude=0, longitude=0)
    if "isentropic_layer" in sample.dims:
        return sample.transpose("isentropic_layer")
    if "isentropic_level" in sample.dims:
        return sample.transpose("isentropic_level")
    return sample


def test_isentropic_metadata_distinguishes_pressure_centers_and_edges():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    temperature = _stable_temperature(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0]),
    )
    pt = potential_temperature(temperature, pressure)
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, 900.0))
    interfaces = isentropic_interfaces([180.0, 200.0, 220.0])

    metadata = interpolate_pressure_to_isentropes_metadata(pt, pressure, interfaces, theta_mask=theta)

    assert float(metadata["column_top_pressure"].isel(time=0, latitude=0, longitude=0)) == 300.0
    assert float(metadata["column_bottom_pressure"].isel(time=0, latitude=0, longitude=0)) == 700.0
    assert float(metadata["column_top_edge_pressure"].isel(time=0, latitude=0, longitude=0)) == 200.0
    assert float(metadata["column_bottom_edge_pressure"].isel(time=0, latitude=0, longitude=0)) == 800.0
    np.testing.assert_allclose(
        _column_sample(metadata["pressure_on_isentrope"]).values,
        np.asarray([np.nan, 600.0, 400.0, np.nan]),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("ps_value", "expected_interfaces", "expected_thickness"),
    [
        (450.0, np.asarray([400.0, 400.0, 400.0, 200.0]), np.asarray([0.0, 0.0, 200.0])),
        (650.0, np.asarray([600.0, 600.0, 400.0, 200.0]), np.asarray([0.0, 200.0, 200.0])),
        (900.0, np.asarray([800.0, 600.0, 400.0, 200.0]), np.asarray([200.0, 200.0, 200.0])),
    ],
)
def test_isentropic_layer_mass_statistics_match_discrete_whole_cell_mass(
    ps_value: float,
    expected_interfaces: np.ndarray,
    expected_thickness: np.ndarray,
):
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    integrator = build_mass_integrator(level, latitude, longitude)
    temperature = _stable_temperature(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0]),
    )
    ps = surface_pressure(time, latitude, longitude, ps_value)
    theta = make_theta(pressure, ps)
    pt = potential_temperature(temperature, pressure)
    interfaces = isentropic_interfaces([180.0, 200.0, 220.0])
    metadata = interpolate_pressure_to_isentropes_metadata(pt, pressure, interfaces, theta_mask=theta)

    stats = isentropic_layer_mass_statistics(metadata, integrator=integrator)
    stats_with_surface_pressure = isentropic_layer_mass_statistics(
        metadata,
        surface_pressure=ps,
        integrator=integrator,
    )
    direct_mass = integrator.integrate_full(theta)

    for name in (
        "interface_pressure",
        "layer_pressure_thickness",
        "layer_mass_per_area",
        "layer_mass",
        "column_mass_per_area",
        "column_mass",
        "cumulative_mass_above_per_area",
        "cumulative_mass_above",
    ):
        np.testing.assert_allclose(stats[name].values, stats_with_surface_pressure[name].values)

    np.testing.assert_allclose(_column_sample(stats["interface_pressure"]).values, expected_interfaces)
    np.testing.assert_allclose(_column_sample(stats["layer_pressure_thickness"]).values, expected_thickness)
    np.testing.assert_allclose(stats["layer_mass"].sum(dim="isentropic_layer").values, stats["column_mass"].values)
    np.testing.assert_allclose(stats["column_mass"].values, direct_mass.values)
    assert stats.attrs["mass_mode"] == "discrete_whole_cell_phase2"
    assert stats.attrs["surface_pressure_behavior"] == "ignored_for_main_outputs"


def test_isentropic_layer_mass_statistics_support_irregular_pressure_spacing():
    time, _, latitude, longitude = make_coords()
    level = xr.DataArray(
        np.asarray([850.0, 600.0, 250.0]),
        dims=("level",),
        coords={"level": [850.0, 600.0, 250.0]},
        name="level",
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )
    pressure = pressure_field(time, level, latitude, longitude)
    integrator = build_mass_integrator(level, latitude, longitude)
    temperature = _stable_temperature(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0]),
    )
    ps = surface_pressure(time, latitude, longitude, 1100.0)
    theta = make_theta(pressure, ps)
    pt = potential_temperature(temperature, pressure)
    interfaces = isentropic_interfaces([180.0, 200.0, 220.0])

    metadata = interpolate_pressure_to_isentropes_metadata(pt, pressure, interfaces, theta_mask=theta)
    stats = isentropic_layer_mass_statistics(metadata, integrator=integrator)
    direct_mass = integrator.integrate_full(theta)

    np.testing.assert_allclose(
        _column_sample(stats["interface_pressure"]).values,
        np.asarray([975.0, 725.0, 425.0, 75.0]),
    )
    np.testing.assert_allclose(stats["layer_mass"].sum(dim="isentropic_layer").values, stats["column_mass"].values)
    np.testing.assert_allclose(stats["column_mass"].values, direct_mass.values)
