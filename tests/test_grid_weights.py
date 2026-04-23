from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.grid_weights import cell_area, infer_grid, latitude_weights, zonal_band_area
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
