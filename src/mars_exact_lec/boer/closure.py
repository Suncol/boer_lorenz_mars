"""Residual four-box closure diagnostics for the Mars exact Boer cycle."""

from __future__ import annotations

import xarray as xr

from ..common.time_derivatives import time_derivative


def _annotate_power(term: xr.DataArray, name: str, long_name: str) -> xr.DataArray:
    term = term.rename(name)
    term.attrs["normalization"] = "global_integral"
    term.attrs["base_quantity"] = "power"
    term.attrs.setdefault("units", "W")
    term.attrs["long_name"] = long_name
    return term


def four_box_storage_tendencies(
    A_Z: xr.DataArray,
    A_E: xr.DataArray,
    K_Z: xr.DataArray,
    K_E: xr.DataArray,
) -> xr.Dataset:
    """Return the four storage tendencies of the exact Lorenz/Boer reservoirs."""

    return xr.Dataset(
        data_vars={
            "dA_Z_dt": _annotate_power(time_derivative(A_Z), "dA_Z_dt", "time tendency of zonal APE"),
            "dA_E_dt": _annotate_power(time_derivative(A_E), "dA_E_dt", "time tendency of eddy APE"),
            "dK_Z_dt": _annotate_power(time_derivative(K_Z), "dK_Z_dt", "time tendency of zonal KE"),
            "dK_E_dt": _annotate_power(time_derivative(K_E), "dK_E_dt", "time tendency of eddy KE"),
        }
    )


def four_box_residual_generation_dissipation(
    A_Z: xr.DataArray,
    A_E: xr.DataArray,
    K_Z: xr.DataArray,
    K_E: xr.DataArray,
    C_Z: xr.DataArray,
    C_A: xr.DataArray,
    C_E: xr.DataArray,
    C_K: xr.DataArray,
) -> xr.Dataset:
    """Return residual four-box generation and dissipation diagnostics."""

    storage = four_box_storage_tendencies(A_Z, A_E, K_Z, K_E)
    result = storage.copy()
    result["G_Z"] = _annotate_power(
        storage["dA_Z_dt"] + C_Z + C_A,
        "G_Z",
        "residual zonal diabatic generation",
    )
    result["G_E"] = _annotate_power(
        storage["dA_E_dt"] + C_E - C_A,
        "G_E",
        "residual eddy diabatic generation",
    )
    result["F_Z"] = _annotate_power(
        C_Z - C_K - storage["dK_Z_dt"],
        "F_Z",
        "residual zonal frictional dissipation",
    )
    result["F_E"] = _annotate_power(
        C_E + C_K - storage["dK_E_dt"],
        "F_E",
        "residual eddy frictional dissipation",
    )
    return result


__all__ = ["four_box_storage_tendencies", "four_box_residual_generation_dissipation"]
