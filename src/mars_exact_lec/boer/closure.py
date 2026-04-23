"""Residual four-box closure diagnostics for the Mars exact Boer cycle."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..common.time_derivatives import time_derivative


_DOMAIN_KEYS = ("surface_pressure_policy", "domain", "not_exact_full_atmosphere")
_SUPPORTED_NORMALIZATIONS = {"global_integral", "planetary_mean_per_area"}


def _require_time_coordinate(term: xr.DataArray) -> xr.DataArray:
    if "time" not in term.dims:
        raise ValueError(f"four_box closure terms must contain a 'time' dimension; {term.name!r} does not.")
    return xr.DataArray(term.coords["time"])


def _ensure_matching_time_coordinates(*terms: xr.DataArray) -> None:
    reference_time: xr.DataArray | None = None
    for term in terms:
        time_coord = _require_time_coordinate(term)
        if reference_time is None:
            reference_time = time_coord
            continue
        if not np.array_equal(reference_time.values, time_coord.values):
            raise ValueError("four_box closure requires all inputs to share the same time coordinate.")


def _infer_normalization(term: xr.DataArray) -> str:
    normalization = term.attrs.get("normalization")
    if normalization is not None:
        normalized = str(normalization).strip()
        if normalized not in _SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization {normalized!r} on {term.name!r}; "
                f"expected one of {sorted(_SUPPORTED_NORMALIZATIONS)!r}."
            )
        return normalized

    units = str(term.attrs.get("units", "")).strip()
    return "planetary_mean_per_area" if units.endswith("m-2") else "global_integral"


def _resolve_shared_normalization(*terms: xr.DataArray) -> str:
    shared: str | None = None
    for term in terms:
        normalization = _infer_normalization(term)
        if shared is None:
            shared = normalization
            continue
        if normalization != shared:
            raise ValueError(
                "four_box closure requires all inputs to share the same normalization "
                "(global-integral or planetary-mean per-area)."
            )
    return "global_integral" if shared is None else shared


def _validate_units(term: xr.DataArray, *, expected_prefix: str) -> None:
    units = term.attrs.get("units")
    if not isinstance(units, str) or not units.startswith(expected_prefix):
        raise ValueError(
            f"{term.name!r} must declare units starting with {expected_prefix!r}; got {units!r}."
        )


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
    normalization: str,
    domain_metadata: dict[str, str | bool] | None = None,
) -> xr.DataArray:
    term = term.rename(name)
    term.attrs["normalization"] = normalization
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

    _ensure_matching_time_coordinates(A_Z, A_E, K_Z, K_E)
    for term in (A_Z, A_E, K_Z, K_E):
        _validate_units(term, expected_prefix="J")
    domain_metadata = _resolve_shared_domain_metadata(A_Z, A_E, K_Z, K_E)
    normalization = _resolve_shared_normalization(A_Z, A_E, K_Z, K_E)
    result = xr.Dataset(
        data_vars={
            "dA_Z_dt": _annotate_power(
                time_derivative(A_Z),
                "dA_Z_dt",
                "time tendency of zonal APE",
                normalization=normalization,
                domain_metadata=domain_metadata,
            ),
            "dA_E_dt": _annotate_power(
                time_derivative(A_E),
                "dA_E_dt",
                "time tendency of eddy APE",
                normalization=normalization,
                domain_metadata=domain_metadata,
            ),
            "dK_Z_dt": _annotate_power(
                time_derivative(K_Z),
                "dK_Z_dt",
                "time tendency of zonal KE",
                normalization=normalization,
                domain_metadata=domain_metadata,
            ),
            "dK_E_dt": _annotate_power(
                time_derivative(K_E),
                "dK_E_dt",
                "time tendency of eddy KE",
                normalization=normalization,
                domain_metadata=domain_metadata,
            ),
        }
    )
    result.attrs["normalization"] = normalization
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

    _ensure_matching_time_coordinates(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)
    for term in (A_Z, A_E, K_Z, K_E):
        _validate_units(term, expected_prefix="J")
    for term in (C_Z, C_A, C_E, C_K):
        _validate_units(term, expected_prefix="W")
    domain_metadata = _resolve_shared_domain_metadata(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)
    normalization = _resolve_shared_normalization(A_Z, A_E, K_Z, K_E, C_Z, C_A, C_E, C_K)
    storage = four_box_storage_tendencies(A_Z, A_E, K_Z, K_E)
    result = storage.copy()
    result["G_Z"] = _annotate_power(
        storage["dA_Z_dt"] + C_Z + C_A,
        "G_Z",
        "residual zonal diabatic generation",
        normalization=normalization,
        domain_metadata=domain_metadata,
    )
    result["G_E"] = _annotate_power(
        storage["dA_E_dt"] + C_E - C_A,
        "G_E",
        "residual eddy diabatic generation",
        normalization=normalization,
        domain_metadata=domain_metadata,
    )
    result["F_Z"] = _annotate_power(
        C_Z - C_K - storage["dK_Z_dt"],
        "F_Z",
        "residual zonal frictional dissipation",
        normalization=normalization,
        domain_metadata=domain_metadata,
    )
    result["F_E"] = _annotate_power(
        C_E + C_K - storage["dK_E_dt"],
        "F_E",
        "residual eddy frictional dissipation",
        normalization=normalization,
        domain_metadata=domain_metadata,
    )
    result.attrs["normalization"] = normalization
    result.attrs.update(domain_metadata)
    return result


__all__ = ["four_box_storage_tendencies", "four_box_residual_generation_dissipation"]
