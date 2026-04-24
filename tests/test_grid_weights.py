from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec._validation import normalize_coordinate
from mars_exact_lec.common.grid_weights import (
    cell_area,
    infer_grid,
    latitude_weights,
    longitude_bounds,
    zonal_band_area,
)
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.constants_mars import MARS

from .helpers import make_coords


@pytest.mark.parametrize(("grid", "expected"), [("regular", "regular"), ("gaussian", "gaussian")])
def test_grid_weights_sum_to_global_area(grid, expected):
    _, _, latitude, longitude = make_coords(grid=grid)

    assert infer_grid(latitude) == expected

    areas = cell_area(latitude, longitude)
    bands = zonal_band_area(latitude)
    total_area = 4.0 * np.pi * MARS.a**2

    np.testing.assert_allclose(float(areas.sum()), total_area, rtol=0.0, atol=1e-6 * total_area)
    np.testing.assert_allclose(float(bands.sum()), total_area, rtol=0.0, atol=1e-6 * total_area)
    np.testing.assert_allclose(float(latitude_weights(latitude, normalize=True).sum()), 1.0)


def test_cell_area_prefers_explicit_bounds_over_midpoint_geometry():
    _, _, latitude, longitude = make_coords()
    latitude_bounds = xr.DataArray(
        np.asarray([[55.0, 90.0], [5.0, 55.0], [-35.0, 5.0], [-90.0, -35.0]]),
        dims=("latitude", "bounds"),
        coords={"latitude": latitude.values, "bounds": [0, 1]},
    )
    longitude_bounds = xr.DataArray(
        np.asarray([[-45.0, 30.0], [30.0, 120.0], [120.0, 210.0], [210.0, 315.0]]),
        dims=("longitude", "bounds"),
        coords={"longitude": longitude.values, "bounds": [0, 1]},
    )

    explicit = cell_area(
        latitude,
        longitude,
        latitude_cell_bounds=latitude_bounds,
        longitude_cell_bounds=longitude_bounds,
    )
    midpoint = cell_area(latitude, longitude)

    assert not np.allclose(explicit.values, midpoint.values)
    np.testing.assert_allclose(float(explicit.sum()), 4.0 * np.pi * MARS.a**2, rtol=0.0, atol=1.0e-6 * 4.0 * np.pi * MARS.a**2)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 90.0, 180.0, 270.0],
        [45.0, 135.0, 225.0, 315.0],
        [-180.0, -90.0, 0.0, 90.0],
    ],
)
def test_longitude_validation_accepts_regular_global_rings(values):
    longitude = xr.DataArray(
        np.asarray(values),
        dims=("longitude",),
        coords={"longitude": values},
        name="longitude",
    )

    normalized = normalize_coordinate(longitude, "longitude")

    np.testing.assert_allclose(normalized.values, np.asarray(values))


def test_longitude_validation_rejects_regional_ring_without_bounds():
    _, level, latitude, _ = make_coords()
    longitude = xr.DataArray(
        np.asarray([0.0, 10.0, 20.0, 30.0]),
        dims=("longitude",),
        coords={"longitude": [0.0, 10.0, 20.0, 30.0]},
        name="longitude",
    )

    with pytest.raises(ValueError, match="equally spaced full global ring"):
        normalize_coordinate(longitude, "longitude")
    with pytest.raises(ValueError, match="equally spaced full global ring"):
        longitude_bounds(longitude)
    with pytest.raises(ValueError, match="equally spaced full global ring"):
        build_mass_integrator(level, latitude, longitude)


def test_explicit_longitude_bounds_may_wrap_the_dateline():
    _, _, _, longitude = make_coords()
    wrapped_bounds = xr.DataArray(
        np.asarray([[315.0, 45.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]]),
        dims=("longitude", "bounds"),
        coords={"longitude": longitude.values, "bounds": [0, 1]},
    )

    resolved = longitude_bounds(longitude, bounds=wrapped_bounds)

    np.testing.assert_allclose(
        resolved.values,
        np.asarray([[-45.0, 45.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]]),
    )


def test_explicit_longitude_bounds_must_be_contiguous_and_non_overlapping():
    _, _, _, longitude = make_coords()
    overlapping_bounds = xr.DataArray(
        np.asarray([[-45.0, 45.0], [30.0, 90.0], [90.0, 210.0], [210.0, 300.0]]),
        dims=("longitude", "bounds"),
        coords={"longitude": longitude.values, "bounds": [0, 1]},
    )

    with pytest.raises(ValueError, match="contiguous, non-overlapping"):
        longitude_bounds(longitude, bounds=overlapping_bounds)


def test_infer_grid_rejects_evenly_spaced_latitudes_that_do_not_cover_the_full_sphere():
    latitude = xr.DataArray(
        np.asarray([-60.0, -20.0, 20.0, 60.0]),
        dims=("latitude",),
        coords={"latitude": [-60.0, -20.0, 20.0, 60.0]},
    )

    with pytest.raises(ValueError, match="does not tile the full sphere"):
        infer_grid(latitude)


def test_explicit_latitude_bounds_must_tile_the_full_sphere_without_gaps_or_overlaps():
    _, _, latitude, _ = make_coords()
    bad_bounds = xr.DataArray(
        np.asarray([[-90.0, -10.0], [-20.0, 20.0], [30.0, 60.0], [60.0, 90.0]]),
        dims=("latitude", "bounds"),
        coords={"latitude": latitude.values, "bounds": [0, 1]},
    )

    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        zonal_band_area(latitude, latitude_cell_bounds=bad_bounds)
