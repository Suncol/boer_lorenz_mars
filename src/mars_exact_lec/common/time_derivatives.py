"""Coordinate-derivative helpers shared by the stage-3 exact topographic terms."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._validation import require_dataarray


_MARS_SOL_SECONDS = 88_775.244
_NUMERIC_TIME_UNIT_SCALE = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "sol": _MARS_SOL_SECONDS,
    "sols": _MARS_SOL_SECONDS,
}


def _validate_coordinate_samples(coordinate: np.ndarray, dim: str) -> np.ndarray:
    values = np.asarray(coordinate, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Coordinate {dim!r} must be one-dimensional.")
    if values.size < 2:
        raise ValueError(f"At least two {dim!r} samples are required to compute a derivative.")

    diffs = np.diff(values)
    if np.any(diffs == 0.0):
        raise ValueError(f"Coordinate {dim!r} must be strictly monotonic.")
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError(f"Coordinate {dim!r} must be strictly monotonic.")
    return values


def _parse_numeric_time_units(time_coord: xr.DataArray) -> float:
    units = time_coord.attrs.get("units")
    if units is None:
        raise ValueError(
            "Numeric 'time' coordinates must declare units in time.attrs['units']; "
            "supported units are seconds, minutes, hours, days, and sols."
        )

    normalized = str(units).strip().lower()
    if " since " in normalized:
        normalized = normalized.split(" since ", 1)[0].strip()
    scale = _NUMERIC_TIME_UNIT_SCALE.get(normalized)
    if scale is None:
        raise ValueError(
            f"Unsupported numeric time unit {units!r}; supported units are "
            "'s', 'sec', 'second(s)', 'min', 'minute(s)', 'h', 'hr', 'hour(s)', "
            "'day(s)', and 'sol(s)'."
        )
    return scale


def _time_coordinate_seconds(time_coord: xr.DataArray) -> np.ndarray:
    values = np.asarray(time_coord.values)
    if np.issubdtype(values.dtype, np.datetime64):
        numeric = values.astype("datetime64[ns]").astype(np.int64).astype(float) * 1.0e-9
    else:
        numeric = values.astype(float) * _parse_numeric_time_units(time_coord)
    return _validate_coordinate_samples(numeric, "time")


def _segment_slices(valid: np.ndarray) -> list[slice]:
    valid_index = np.flatnonzero(valid)
    if valid_index.size == 0:
        return []

    splits = np.where(np.diff(valid_index) > 1)[0] + 1
    return [slice(int(segment[0]), int(segment[-1]) + 1) for segment in np.split(valid_index, splits)]


def _segmented_first_derivative_1d(
    values: np.ndarray,
    coordinate: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    coordinate = _validate_coordinate_samples(coordinate, "coordinate")
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=float)

    for segment in _segment_slices(valid):
        segment_values = values[segment]
        segment_coordinate = coordinate[segment]
        if segment_values.size == 1:
            continue
        edge_order = 2 if segment_values.size > 2 else 1
        result[segment] = np.gradient(segment_values, segment_coordinate, edge_order=edge_order)
    return result


def coordinate_derivative(
    field: xr.DataArray,
    dim: str,
    *,
    coordinate: xr.DataArray | None = None,
    valid_mask: xr.DataArray | None = None,
    name: str | None = None,
    derivative_units: str | None = None,
) -> xr.DataArray:
    """Return the first derivative of ``field`` along ``dim``.

    When ``valid_mask`` is provided, derivatives are evaluated independently on
    each contiguous valid segment and remain ``NaN`` on isolated single-sample
    segments.
    """

    field = require_dataarray(field, "field")
    if dim not in field.dims:
        raise ValueError(f"'field' must contain the dimension {dim!r}.")

    if coordinate is None:
        coordinate = xr.DataArray(field.coords[dim])
    else:
        coordinate = require_dataarray(coordinate, "coordinate")
        if coordinate.ndim != 1 or coordinate.dims != (dim,):
            raise ValueError(f"'coordinate' must be one-dimensional with dim {dim!r}.")

    _validate_coordinate_samples(np.asarray(coordinate.values, dtype=float), dim)

    if valid_mask is None:
        valid_mask = xr.apply_ufunc(np.isfinite, field, dask="parallelized", output_dtypes=[bool])
    else:
        valid_mask = require_dataarray(valid_mask, "valid_mask")
        if valid_mask.dims != field.dims:
            raise ValueError("'valid_mask' must have the same dims as 'field'.")
        for coord_name in field.dims:
            reference = field.coords[coord_name].values
            current = valid_mask.coords[coord_name].values
            equal = np.array_equal(reference, current) if coord_name == "time" else np.allclose(reference, current)
            if not equal:
                raise ValueError(
                    f"Coordinate {coord_name!r} of 'valid_mask' does not match the field being differentiated."
                )
        valid_mask = valid_mask.astype(bool)

    derivative = xr.apply_ufunc(
        _segmented_first_derivative_1d,
        field.astype(float),
        coordinate.astype(float),
        valid_mask,
        input_core_dims=[[dim], [dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    derivative = derivative.transpose(*field.dims)
    derivative.name = name if name is not None else (f"d{field.name}_d{dim}" if field.name else f"dfield_d{dim}")
    derivative.attrs = dict(field.attrs)
    if derivative_units is not None:
        derivative.attrs["derivative_units"] = derivative_units
    derivative.attrs["derivative_dimension"] = dim
    return derivative


def time_derivative(field: xr.DataArray, *, valid_mask: xr.DataArray | None = None) -> xr.DataArray:
    """Return the finite-difference time derivative of ``field``.

    The implementation uses second-order centered differences in the interior,
    first-order one-sided differences at contiguous valid-segment boundaries,
    and requires numeric time coordinates to declare explicit units.
    """

    field = require_dataarray(field, "field")
    if "time" not in field.dims:
        raise ValueError("'field' must contain a 'time' dimension.")

    time_coord = xr.DataArray(field.coords["time"])
    time_values = xr.DataArray(
        _time_coordinate_seconds(time_coord),
        dims=("time",),
        coords={"time": field.coords["time"].values},
        name="time_seconds",
    )
    result = coordinate_derivative(
        field,
        "time",
        coordinate=time_values,
        valid_mask=valid_mask,
        name=f"d{field.name}_dt" if field.name else "time_derivative",
        derivative_units="per_second",
    )
    result.attrs["time_derivative"] = True
    return result


__all__ = ["coordinate_derivative", "time_derivative"]
