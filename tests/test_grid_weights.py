from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

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
