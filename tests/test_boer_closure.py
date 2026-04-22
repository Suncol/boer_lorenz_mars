from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.closure import four_box_residual_generation_dissipation, four_box_storage_tendencies
from mars_exact_lec.common.normalization import normalize_dataset_per_area, planetary_area, to_per_area


def _time_coord():
    values = np.asarray([0.0, 1.0, 2.0], dtype=float)
    return xr.DataArray(values, dims=("time",), coords={"time": values}, attrs={"units": "hours"})


def _series(name: str, values, units: str) -> xr.DataArray:
    time = _time_coord()
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("time",),
        coords={"time": time.values},
        name=name,
        attrs={"units": units},
    ).assign_coords(time=time)


def test_four_box_storage_tendencies_produce_power_units():
    storage = four_box_storage_tendencies(
        _series("A_Z", [0.0, 3600.0, 7200.0], "J"),
        _series("A_E", [0.0, 7200.0, 14400.0], "J"),
        _series("K_Z", [0.0, -3600.0, -7200.0], "J"),
        _series("K_E", [0.0, 1800.0, 3600.0], "J"),
    )

    np.testing.assert_allclose(storage["dA_Z_dt"].values, 1.0)
    np.testing.assert_allclose(storage["dA_E_dt"].values, 2.0)
    np.testing.assert_allclose(storage["dK_Z_dt"].values, -1.0)
    np.testing.assert_allclose(storage["dK_E_dt"].values, 0.5)
    assert storage["dA_Z_dt"].attrs["units"] == "W"
    assert storage["dA_Z_dt"].attrs["base_quantity"] == "power"


def test_four_box_residual_generation_and_dissipation_follow_budget_identities():
    A_Z = _series("A_Z", [0.0, 3600.0, 7200.0], "J")
    A_E = _series("A_E", [0.0, 7200.0, 14400.0], "J")
    K_Z = _series("K_Z", [0.0, -3600.0, -7200.0], "J")
    K_E = _series("K_E", [0.0, 1800.0, 3600.0], "J")
    C_Z = _series("C_Z", [2.0, 2.0, 2.0], "W")
    C_A = _series("C_A", [3.0, 3.0, 3.0], "W")
    C_E = _series("C_E", [5.0, 5.0, 5.0], "W")
    C_K = _series("C_K", [7.0, 7.0, 7.0], "W")

    diagnostics = four_box_residual_generation_dissipation(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)

    np.testing.assert_allclose(diagnostics["G_Z"].values, 6.0)
    np.testing.assert_allclose(diagnostics["G_E"].values, 4.0)
    np.testing.assert_allclose(diagnostics["F_Z"].values, -4.0)
    np.testing.assert_allclose(diagnostics["F_E"].values, 11.5)
    assert diagnostics["G_Z"].attrs["units"] == "W"
    assert diagnostics["F_E"].attrs["normalization"] == "global_integral"


def test_per_area_normalization_converts_units_for_scalars_and_datasets():
    area = planetary_area()
    az = _series("A_Z", [area, 2.0 * area, 3.0 * area], "J")
    cz = _series("C_Z", [area, 2.0 * area, 3.0 * area], "W")

    az_per_area = to_per_area(az)
    diagnostics = normalize_dataset_per_area(xr.Dataset({"A_Z": az, "C_Z": cz}))

    np.testing.assert_allclose(az_per_area.values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(diagnostics["A_Z"].values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(diagnostics["C_Z"].values, [1.0, 2.0, 3.0])
    assert az_per_area.attrs["units"] == "J m-2"
    assert diagnostics["C_Z"].attrs["units"] == "W m-2"
    assert diagnostics.attrs["normalization"] == "planetary_mean_per_area"
