"""Normalization helpers for exact Mars energy diagnostics."""

from __future__ import annotations

import xarray as xr

from ..constants_mars import MARS, MarsConstants


_PER_AREA_UNITS = {
    "J": "J m-2",
    "W": "W m-2",
}


def planetary_area(*, constants: MarsConstants = MARS) -> float:
    """Return the total planetary surface area in square metres."""

    return 4.0 * float(constants.a) ** 2 * 3.141592653589793


def to_per_area(
    term: xr.DataArray,
    *,
    constants: MarsConstants = MARS,
    area: float | None = None,
    output_name: str | None = None,
) -> xr.DataArray:
    """Convert a global-integral diagnostic into a planetary-mean per-area quantity."""

    if not isinstance(term, xr.DataArray):
        raise TypeError("'term' must be an xarray.DataArray.")
    resolved_area = float(area) if area is not None else planetary_area(constants=constants)
    result = term / resolved_area
    result.name = output_name if output_name is not None else (f"{term.name}_per_area" if term.name else None)
    result.attrs = dict(term.attrs)
    result.attrs["normalization"] = "planetary_mean_per_area"
    result.attrs["planetary_area"] = resolved_area
    units = result.attrs.get("units")
    if isinstance(units, str):
        result.attrs["units"] = _PER_AREA_UNITS.get(units, f"{units} m-2")
    return result


def normalize_dataset_per_area(
    ds: xr.Dataset,
    *,
    constants: MarsConstants = MARS,
    area: float | None = None,
) -> xr.Dataset:
    """Return a dataset whose data variables have been normalized to per-area quantities."""

    if not isinstance(ds, xr.Dataset):
        raise TypeError("'ds' must be an xarray.Dataset.")

    resolved_area = float(area) if area is not None else planetary_area(constants=constants)
    normalized = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    for name, value in ds.data_vars.items():
        normalized[name] = to_per_area(value, constants=constants, area=resolved_area, output_name=name)
    normalized.attrs["normalization"] = "planetary_mean_per_area"
    normalized.attrs["planetary_area"] = resolved_area
    return normalized


__all__ = ["planetary_area", "to_per_area", "normalize_dataset_per_area"]
