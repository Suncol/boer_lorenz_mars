"""Residual four-box closure diagnostics for the Mars exact Boer cycle."""

from __future__ import annotations

import xarray as xr

from ..common.time_derivatives import time_derivative


_DOMAIN_KEYS = ("surface_pressure_policy", "domain", "not_exact_full_atmosphere")


def _extract_domain_metadata(term: xr.DataArray) -> dict[str, str | bool]:
    missing = [key for key in _DOMAIN_KEYS if key not in term.attrs]
    if missing:
        raise ValueError(
            "four_box closure requires inputs computed on a declared pressure-domain definition; "
            f"missing attrs: {missing!r} on {term.name!r}."
        )
    return {key: term.attrs[key] for key in _DOMAIN_KEYS}


def _resolve_shared_domain_metadata(*terms: xr.DataArray) -> dict[str, str | bool]:
    shared_metadata: dict[str, str | bool] | None = None
    for term in terms:
        metadata = _extract_domain_metadata(term)
        if shared_metadata is None:
            shared_metadata = metadata
            continue
        if metadata != shared_metadata:
            raise ValueError(
                "four_box_residual_generation_dissipation requires inputs computed on the same "
                "pressure-domain definition."
            )
    return {} if shared_metadata is None else dict(shared_metadata)


def _annotate_power(
    term: xr.DataArray,
    name: str,
    long_name: str,
    *,
    domain_metadata: dict[str, str | bool] | None = None,
) -> xr.DataArray:
    term = term.rename(name)
    term.attrs["normalization"] = "global_integral"
    term.attrs["base_quantity"] = "power"
    term.attrs.setdefault("units", "W")
    term.attrs["long_name"] = long_name
    if domain_metadata is not None:
        term.attrs.update(domain_metadata)
    return term


def four_box_storage_tendencies(
    A_Z: xr.DataArray,
    A_E: xr.DataArray,
    K_Z: xr.DataArray,
    K_E: xr.DataArray,
) -> xr.Dataset:
    """Return the four storage tendencies of the exact Lorenz/Boer reservoirs."""

    domain_metadata = _resolve_shared_domain_metadata(A_Z, A_E, K_Z, K_E)
    result = xr.Dataset(
        data_vars={
            "dA_Z_dt": _annotate_power(
                time_derivative(A_Z),
                "dA_Z_dt",
                "time tendency of zonal APE",
                domain_metadata=domain_metadata,
            ),
            "dA_E_dt": _annotate_power(
                time_derivative(A_E),
                "dA_E_dt",
                "time tendency of eddy APE",
                domain_metadata=domain_metadata,
            ),
            "dK_Z_dt": _annotate_power(
                time_derivative(K_Z),
                "dK_Z_dt",
                "time tendency of zonal KE",
                domain_metadata=domain_metadata,
            ),
            "dK_E_dt": _annotate_power(
                time_derivative(K_E),
                "dK_E_dt",
                "time tendency of eddy KE",
                domain_metadata=domain_metadata,
            ),
        }
    )
    result.attrs.update(domain_metadata)
    return result


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

    domain_metadata = _resolve_shared_domain_metadata(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)
    storage = four_box_storage_tendencies(A_Z, A_E, K_Z, K_E)
    result = storage.copy()
    result["G_Z"] = _annotate_power(
        storage["dA_Z_dt"] + C_Z + C_A,
        "G_Z",
        "residual zonal diabatic generation",
        domain_metadata=domain_metadata,
    )
    result["G_E"] = _annotate_power(
        storage["dA_E_dt"] + C_E - C_A,
        "G_E",
        "residual eddy diabatic generation",
        domain_metadata=domain_metadata,
    )
    result["F_Z"] = _annotate_power(
        C_Z - C_K - storage["dK_Z_dt"],
        "F_Z",
        "residual zonal frictional dissipation",
        domain_metadata=domain_metadata,
    )
    result["F_E"] = _annotate_power(
        C_E + C_K - storage["dK_E_dt"],
        "F_E",
        "residual eddy frictional dissipation",
        domain_metadata=domain_metadata,
    )
    result.attrs.update(domain_metadata)
    return result


__all__ = ["four_box_storage_tendencies", "four_box_residual_generation_dissipation"]
