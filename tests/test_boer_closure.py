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
        attrs={
            "units": units,
            "surface_pressure_policy": "raise",
            "domain": "full_model_pressure_domain",
            "not_exact_full_atmosphere": False,
        },
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
    assert storage["dA_Z_dt"].attrs["domain"] == "full_model_pressure_domain"
    assert storage.attrs["not_exact_full_atmosphere"] is False


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
    assert diagnostics["G_Z"].attrs["domain"] == "full_model_pressure_domain"
    assert diagnostics.attrs["surface_pressure_policy"] == "raise"


def test_four_box_residual_generation_and_dissipation_close_total_budget_identity():
    A_Z = _series("A_Z", [0.0, 3600.0, 7200.0], "J")
    A_E = _series("A_E", [0.0, 7200.0, 14400.0], "J")
    K_Z = _series("K_Z", [0.0, -3600.0, -7200.0], "J")
    K_E = _series("K_E", [0.0, 1800.0, 3600.0], "J")
    C_Z = _series("C_Z", [2.0, 2.0, 2.0], "W")
    C_A = _series("C_A", [3.0, 3.0, 3.0], "W")
    C_E = _series("C_E", [5.0, 5.0, 5.0], "W")
    C_K = _series("C_K", [7.0, 7.0, 7.0], "W")

    storage = four_box_storage_tendencies(A_Z, A_E, K_Z, K_E)
    diagnostics = four_box_residual_generation_dissipation(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)

    lhs = storage["dA_Z_dt"] + storage["dA_E_dt"] + storage["dK_Z_dt"] + storage["dK_E_dt"]
    rhs = diagnostics["G_Z"] + diagnostics["G_E"] - diagnostics["F_Z"] - diagnostics["F_E"]

    np.testing.assert_allclose(lhs.values, rhs.values, rtol=0.0, atol=1.0e-12)
    assert lhs.attrs["units"] == "W"


def test_per_area_normalization_converts_units_for_scalars_and_datasets():
    area = planetary_area()
    az = _series("A_Z", [area, 2.0 * area, 3.0 * area], "J")
    cz = _series("C_Z", [area, 2.0 * area, 3.0 * area], "W")

    az_per_area = to_per_area(az)
    diagnostics = normalize_dataset_per_area(
        xr.Dataset(
            {"A_Z": az, "C_Z": cz},
            attrs={
                "surface_pressure_policy": "raise",
                "domain": "full_model_pressure_domain",
                "not_exact_full_atmosphere": False,
            },
        )
    )

    np.testing.assert_allclose(az_per_area.values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(diagnostics["A_Z"].values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(diagnostics["C_Z"].values, [1.0, 2.0, 3.0])
    assert az_per_area.attrs["units"] == "J m-2"
    assert diagnostics["C_Z"].attrs["units"] == "W m-2"
    assert diagnostics.attrs["normalization"] == "planetary_mean_per_area"
    assert az_per_area.attrs["domain"] == "full_model_pressure_domain"
    assert diagnostics["A_Z"].attrs["not_exact_full_atmosphere"] is False
    assert diagnostics.attrs["domain"] == "full_model_pressure_domain"


def test_four_box_closure_propagates_clip_domain_metadata():
    A_Z = _series("A_Z", [0.0, 3600.0, 7200.0], "J")
    A_E = _series("A_E", [0.0, 7200.0, 14400.0], "J")
    K_Z = _series("K_Z", [0.0, -3600.0, -7200.0], "J")
    K_E = _series("K_E", [0.0, 1800.0, 3600.0], "J")
    C_Z = _series("C_Z", [2.0, 2.0, 2.0], "W")
    C_A = _series("C_A", [3.0, 3.0, 3.0], "W")
    C_E = _series("C_E", [5.0, 5.0, 5.0], "W")
    C_K = _series("C_K", [7.0, 7.0, 7.0], "W")

    for term in (A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K):
        term.attrs["surface_pressure_policy"] = "clip"
        term.attrs["domain"] = "truncated_to_model_pressure_domain"
        term.attrs["not_exact_full_atmosphere"] = True

    diagnostics = four_box_residual_generation_dissipation(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)

    assert diagnostics["G_Z"].attrs["surface_pressure_policy"] == "clip"
    assert diagnostics["F_E"].attrs["domain"] == "truncated_to_model_pressure_domain"
    assert diagnostics["dA_Z_dt"].attrs["not_exact_full_atmosphere"] is True
    assert diagnostics.attrs["domain"] == "truncated_to_model_pressure_domain"


def test_four_box_closure_rejects_mixed_pressure_domain_metadata():
    A_Z = _series("A_Z", [0.0, 3600.0, 7200.0], "J")
    A_E = _series("A_E", [0.0, 7200.0, 14400.0], "J")
    K_Z = _series("K_Z", [0.0, -3600.0, -7200.0], "J")
    K_E = _series("K_E", [0.0, 1800.0, 3600.0], "J")
    C_Z = _series("C_Z", [2.0, 2.0, 2.0], "W")
    C_A = _series("C_A", [3.0, 3.0, 3.0], "W")
    C_E = _series("C_E", [5.0, 5.0, 5.0], "W")
    C_K = _series("C_K", [7.0, 7.0, 7.0], "W")

    C_Z.attrs["surface_pressure_policy"] = "clip"
    C_Z.attrs["domain"] = "truncated_to_model_pressure_domain"
    C_Z.attrs["not_exact_full_atmosphere"] = True

    with pytest.raises(ValueError, match="same pressure-domain definition"):
        four_box_residual_generation_dissipation(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)
