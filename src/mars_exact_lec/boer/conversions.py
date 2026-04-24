"""Boer exact conversion terms used by the Mars exact Lorenz cycle branch."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._validation import (
    ensure_matching_coordinates,
    ensure_matching_surface_coordinates,
    normalize_field,
    normalize_surface_field,
    normalize_theta_mask,
    normalize_zonal_field,
    require_dataarray,
    resolve_deprecated_theta_mask,
)
from ..common.geopotential import (
    broadcast_surface_field,
    resolve_deprecated_geopotential_mode,
    resolve_geopotential,
)
from ..common.integrals import MassIntegrator
from ..common.topography_measure import TopographyAwareMeasure, resolve_exact_measure
from ..common.time_derivatives import coordinate_derivative, time_derivative
from ..common.zonal_ops import (
    representative_eddy,
    representative_zonal_mean,
    theta_coverage,
    weighted_coverage,
    weighted_representative_eddy,
    weighted_representative_zonal_mean,
    zonal_mean,
)
from ..constants_mars import MARS, MarsConstants


_METRIC_TOL = 1.0e-12


def _require_integrator(integrator: MassIntegrator | None) -> MassIntegrator:
    if integrator is None:
        raise TypeError("'integrator' is required.")
    return integrator


def _weight_field(theta_mask: xr.DataArray, measure: TopographyAwareMeasure | None) -> xr.DataArray:
    theta_mask = normalize_theta_mask(theta_mask)
    if measure is None:
        return theta_mask
    ensure_matching_coordinates(theta_mask, [measure.cell_fraction])
    return measure.cell_fraction


def _coverage_field(theta_mask: xr.DataArray, measure: TopographyAwareMeasure | None) -> xr.DataArray:
    weight = _weight_field(theta_mask, measure)
    return weighted_coverage(weight) if measure is not None else theta_coverage(weight)


def _representative_mean(
    field: xr.DataArray,
    theta_mask: xr.DataArray,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    weight = _weight_field(theta_mask, measure)
    if measure is None:
        return representative_zonal_mean(field, weight)
    return weighted_representative_zonal_mean(field, weight)


def _representative_eddy_component(
    field: xr.DataArray,
    theta_mask: xr.DataArray,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    weight = _weight_field(theta_mask, measure)
    if measure is None:
        return representative_eddy(field, weight)
    return weighted_representative_eddy(field, weight)


def _safe_mass_ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    denominator = denominator.broadcast_like(numerator)
    return xr.where(denominator > 0.0, numerator / denominator, 0.0)


def _integrate_zonal_mass_aware(
    integrand: xr.DataArray,
    coverage: xr.DataArray,
    integrator: MassIntegrator,
    measure: TopographyAwareMeasure | None,
) -> xr.DataArray:
    if measure is None:
        return integrator.integrate_zonal(integrand)
    return measure.integrate_zonal(_safe_mass_ratio(integrand, coverage))


def _effective_surface_pressure(ps: xr.DataArray, measure: TopographyAwareMeasure | None) -> xr.DataArray:
    ps = normalize_surface_field(ps, "ps")
    if measure is None:
        return ps
    ensure_matching_surface_coordinates(ps, [measure.effective_surface_pressure])
    return measure.effective_surface_pressure


def _annotate_quantity(
    result: xr.DataArray,
    *,
    units: str,
    base_quantity: str,
    measure: TopographyAwareMeasure | None = None,
    extra_attrs: dict[str, object] | None = None,
) -> xr.DataArray:
    result.attrs["units"] = units
    result.attrs["normalization"] = "global_integral"
    result.attrs["base_quantity"] = base_quantity
    if measure is not None:
        result.attrs.update(measure.domain_metadata)
    if extra_attrs is not None:
        result.attrs.update(extra_attrs)
    return result


def _ck2_boundary_attrs(
    *,
    reconstruction: str = "linear_pressure_phi_star_to_layer_faces",
    geopotential_source: str = "level_center_geopotential",
    geopotential_mode: str = "strict",
    geopotential_reconstruction_allowed: bool = False,
    geopotential_reconstruction_approximate: bool = False,
) -> dict[str, object]:
    """Return machine-readable provenance for the C_K2 finite-volume scheme."""

    return {
        "ck2_discretization": "cut_cell_finite_volume_leibniz_corrected",
        "ck2_geopotential_source": geopotential_source,
        "ck2_geopotential_mode": geopotential_mode,
        "ck2_geopotential_reconstruction_allowed": geopotential_reconstruction_allowed,
        "ck2_geopotential_reconstruction_approximate": geopotential_reconstruction_approximate,
        "ck2_vertical_integral": "trapezoidal_phi_star_dp",
        "ck2_reconstruction": reconstruction,
        "ck2_bottom_pressure": "min(layer_lower_edge,effective_surface_pressure)_clamped_to_layer",
        "ck2_horizontal_boundary_correction": "subtract_phi_bottom_grad_p_bottom",
        "ck2_pressure_term": "phi_bottom_minus_phi_top_over_full_delta_p",
        "ck2_zonal_mean": "cell_area_weighted_full_longitude_ring",
        "ck2_derivative_mask": "finite_volume_above_ground_pressure_thickness_positive",
    }


def _ck2_level_center_geopotential_source(phi: xr.DataArray) -> str:
    if bool(phi.attrs.get("reconstructed_hydrostatically", False)) or (
        phi.attrs.get("geopotential_source") == "hydrostatically_reconstructed"
    ):
        return "hydrostatically_reconstructed_geopotential"
    if bool(phi.attrs.get("filled_hydrostatically", False)):
        return "level_center_geopotential_with_hydrostatic_fill"
    if bool(phi.attrs.get("filled_below_ground", False)):
        return "level_center_geopotential_with_log_pressure_fill"
    return "level_center_geopotential"


def _ck2_geopotential_reconstruction_approximate(phi: xr.DataArray) -> bool:
    return any(
        bool(phi.attrs.get(key, False))
        for key in (
            "geopotential_reconstruction_approximate",
            "reconstructed_hydrostatically",
            "filled_hydrostatically",
            "filled_below_ground",
        )
    )


def _supported_directional_zonal_term(derivative: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    """Return ``[w D]`` while explicitly excluding unsupported derivative points.

    This helper intentionally does not renormalize over the supported subset. It
    computes the geometric zonal mean of the weighted derivative, with
    unsupported points contributing zero weight and zero tendency.
    """

    derivative = normalize_field(derivative, "derivative")
    weight = normalize_field(weight, "weight")
    ensure_matching_coordinates(derivative, [weight])
    support_mask = xr.apply_ufunc(np.isfinite, derivative, dask="allowed")
    derivative_eff = xr.where(support_mask, derivative, 0.0)
    weight_eff = xr.where(support_mask, weight, 0.0)
    return zonal_mean(weight_eff * derivative_eff)


def _resolved_measure(
    integrator: MassIntegrator,
    *,
    theta_mask: xr.DataArray | None = None,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    diagnostic_name: str,
) -> TopographyAwareMeasure:
    return resolve_exact_measure(
        integrator,
        measure=measure,
        ps=ps,
        theta_mask=theta_mask,
        theta=theta,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name=diagnostic_name,
    )


def _coerce_to_zonal(field: xr.DataArray, name: str) -> xr.DataArray:
    """Accept either a zonal field or a longitude-constant full field."""

    try:
        return normalize_zonal_field(field, name)
    except ValueError:
        full = normalize_field(field, name)
        reference = full.isel(longitude=0, drop=True)
        reconstructed = reference.expand_dims(longitude=full.coords["longitude"]).transpose(*full.dims)
        if not np.allclose(
            np.asarray(full.values, dtype=float),
            np.asarray(reconstructed.values, dtype=float),
            equal_nan=True,
        ):
            raise ValueError(
                f"{name!r} must already be zonal or longitude-constant on the canonical full grid; "
                "longitude-varying full fields are not valid zonal diagnostics."
            )
        return reference


def _ensure_matching_zonal_coordinates(reference: xr.DataArray, field: xr.DataArray, name: str) -> None:
    """Require a zonal field to share the canonical time/level/latitude coordinates."""

    for coord_name in ("time", "level", "latitude"):
        reference_coord = reference.coords[coord_name].values
        field_coord = field.coords[coord_name].values
        if coord_name == "time":
            equal = np.array_equal(reference_coord, field_coord)
        else:
            equal = np.allclose(reference_coord, field_coord)
        if not equal:
            raise ValueError(f"Coordinate {coord_name!r} of {name!r} does not match the reference field.")


def _pressure_derivative(field: xr.DataArray, *, valid_mask: xr.DataArray | None = None) -> xr.DataArray:
    """Return ``∂field/∂p`` using the canonical pressure coordinate."""

    return coordinate_derivative(
        field,
        "level",
        valid_mask=valid_mask,
        name=f"d{field.name}_dp" if field.name else "pressure_derivative",
        derivative_units="per_pascal",
    )


def _interp_column_to_pressure(
    values: np.ndarray,
    source_pressure: np.ndarray,
    target_pressure: np.ndarray,
    source_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    source_pressure = np.asarray(source_pressure, dtype=float)
    target_pressure = np.asarray(target_pressure, dtype=float)
    source_valid = (
        np.asarray(source_valid_mask, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(source_pressure)
        & (source_pressure > 0.0)
    )
    target_valid = (
        np.asarray(target_valid_mask, dtype=bool)
        & np.isfinite(target_pressure)
        & (target_pressure > 0.0)
    )
    result = np.full(target_pressure.shape, np.nan, dtype=float)
    if not np.any(target_valid) or not np.any(source_valid):
        return result

    x = source_pressure[source_valid]
    y = values[source_valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    target = target_pressure[target_valid]
    if x.size == 1:
        result[target_valid] = y[0]
        return result

    interpolated = np.interp(target, x, y)
    left = target < x[0]
    right = target > x[-1]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        interpolated[left] = y[0] + slope * (target[left] - x[0])
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        interpolated[right] = y[-1] + slope * (target[right] - x[-1])

    result[target_valid] = interpolated
    return result


def _interp_field_to_pressure(
    field: xr.DataArray,
    target_pressure: xr.DataArray,
    *,
    valid_mask: xr.DataArray | None = None,
    target_valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Linearly reconstruct a full field to target pressures in each column."""

    field = normalize_field(field, "field")
    target_pressure = normalize_field(target_pressure, "target_pressure")
    ensure_matching_coordinates(field, [target_pressure])
    if valid_mask is None:
        valid_mask = xr.apply_ufunc(np.isfinite, field, dask="parallelized", output_dtypes=[bool])
    else:
        valid_mask = normalize_field(valid_mask, "valid_mask").astype(bool)
        ensure_matching_coordinates(field, [valid_mask])
    if target_valid_mask is None:
        target_valid_mask = xr.apply_ufunc(
            np.isfinite,
            target_pressure,
            dask="parallelized",
            output_dtypes=[bool],
        )
    else:
        target_valid_mask = normalize_field(target_valid_mask, "target_valid_mask").astype(bool)
        ensure_matching_coordinates(field, [target_valid_mask])

    source_pressure = xr.DataArray(
        np.asarray(field.coords["level"].values, dtype=float),
        dims=("level",),
        coords={"level": field.coords["level"].values},
        name="source_pressure",
    )
    reconstructed = xr.apply_ufunc(
        _interp_column_to_pressure,
        field.astype(float),
        source_pressure.astype(float),
        target_pressure.astype(float),
        valid_mask,
        target_valid_mask,
        input_core_dims=[["level"], ["level"], ["level"], ["level"], ["level"]],
        output_core_dims=[["level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).transpose(*field.dims)
    reconstructed.name = f"{field.name}_at_pressure" if field.name else "field_at_pressure"
    reconstructed.attrs = dict(field.attrs)
    reconstructed.attrs["vertical_reconstruction"] = "linear_pressure"
    return reconstructed


def _normalize_interface_geopotential(
    interface_geopotential: xr.DataArray,
    reference: xr.DataArray,
    measure: TopographyAwareMeasure,
) -> xr.DataArray:
    """Return interface geopotential on ``(time, level_edge, latitude, longitude)``."""

    interface_geopotential = require_dataarray(interface_geopotential, "interface_geopotential")
    interface_dims = ("time", "level_edge", "latitude", "longitude")
    if set(interface_geopotential.dims) != set(interface_dims):
        raise ValueError(
            "'interface_geopotential' must contain exactly the dims "
            f"{interface_dims}; got {interface_geopotential.dims!r}."
        )

    interface_geopotential = interface_geopotential.transpose(*interface_dims).astype(float)
    expected_edges = measure.level_edges
    if interface_geopotential.sizes["level_edge"] != expected_edges.sizes["level_edge"]:
        raise ValueError(
            "'interface_geopotential' must have one more level_edge than the pressure-level field."
        )

    if "level_edge" in interface_geopotential.coords:
        edge_coord = np.asarray(interface_geopotential.coords["level_edge"].values, dtype=float)
        expected_pressure = np.asarray(expected_edges.values, dtype=float)
        expected_index = np.arange(expected_pressure.size, dtype=float)
        if not (
            np.allclose(edge_coord, expected_pressure, rtol=1.0e-8, atol=1.0e-6)
            or np.allclose(edge_coord, expected_index, rtol=0.0, atol=0.0)
        ):
            raise ValueError(
                "'interface_geopotential' level_edge coordinate must either match the "
                "pressure interfaces used by the integrator or be a zero-based edge index."
            )

    reference = normalize_field(reference, "reference")
    for coord_name in ("time", "latitude", "longitude"):
        reference_coord = reference.coords[coord_name].values
        current_coord = interface_geopotential.coords[coord_name].values
        equal = (
            np.array_equal(reference_coord, current_coord)
            if coord_name == "time"
            else np.allclose(reference_coord, current_coord)
        )
        if not equal:
            raise ValueError(
                f"Coordinate {coord_name!r} of 'interface_geopotential' does not match the reference field."
            )
    return interface_geopotential


def _interface_values_to_layer_faces(
    interface_values: xr.DataArray,
    level: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    lower = interface_values.isel(level_edge=slice(0, -1)).rename({"level_edge": "level"})
    upper = interface_values.isel(level_edge=slice(1, None)).rename({"level_edge": "level"})
    lower = lower.assign_coords(level=level.values).transpose("time", "level", "latitude", "longitude")
    upper = upper.assign_coords(level=level.values).transpose("time", "level", "latitude", "longitude")
    lower.name = f"{interface_values.name}_lower_face" if interface_values.name else "interface_lower_face"
    upper.name = f"{interface_values.name}_upper_face" if interface_values.name else "interface_upper_face"
    lower.attrs = dict(interface_values.attrs)
    upper.attrs = dict(interface_values.attrs)
    return lower, upper


def _weighted_representative_eddy_on_faces(field: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    field = normalize_field(field, "field")
    weight = normalize_field(weight, "weight")
    ensure_matching_coordinates(field, [weight])
    support = (weight > 0.0) & xr.apply_ufunc(np.isfinite, field, dask="parallelized", output_dtypes=[bool])
    weight_eff = xr.where(support, weight, 0.0)
    field_eff = xr.where(support, field, 0.0)
    coverage = weighted_coverage(weight_eff)
    weighted_mean = zonal_mean(weight_eff * field_eff)
    mean = _safe_mass_ratio(weighted_mean, coverage)
    return xr.where(support, field - mean.broadcast_like(field), np.nan)


def _ck2_interface_geopotential_face_stars(
    interface_geopotential: xr.DataArray,
    theta_mask: xr.DataArray,
    measure: TopographyAwareMeasure,
    *,
    center_geopotential: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray, str]:
    """Return top and effective-bottom ``Phi*`` from interface geopotential."""

    theta_mask = normalize_theta_mask(theta_mask)
    interface_geopotential = _normalize_interface_geopotential(interface_geopotential, theta_mask, measure)
    lower_face, upper_face = _interface_values_to_layer_faces(
        interface_geopotential,
        theta_mask.coords["level"],
    )
    dp_above = measure.above_ground_dp
    p_top = measure.upper_edge.broadcast_like(dp_above)
    p_lower = measure.lower_edge.broadcast_like(dp_above)
    p_bottom = measure.effective_bottom_pressure
    layer_fraction = xr.where(p_lower > p_top, (p_bottom - p_top) / (p_lower - p_top), 0.0)
    linear_bottom = upper_face + layer_fraction * (lower_face - upper_face)
    reconstruction = "interface_geopotential_faces_pressure_linear_partial_bottom"

    if phis is not None:
        surface_geopotential = broadcast_surface_field(phis, theta_mask, "phis")
        surface_geopotential = surface_geopotential.expand_dims(level=theta_mask.coords["level"]).transpose(
            "time",
            "level",
            "latitude",
            "longitude",
        )
        partial_bottom = (dp_above > 0.0) & (p_bottom < p_lower - 1.0e-9)
        bottom_face = xr.where(partial_bottom, surface_geopotential, lower_face)
        reconstruction = "interface_geopotential_faces_surface_partial_bottom"
    elif center_geopotential is not None:
        center_geopotential = normalize_field(center_geopotential, "center_geopotential")
        ensure_matching_coordinates(theta_mask, [center_geopotential])
        p_center = xr.DataArray(
            np.asarray(theta_mask.coords["level"].values, dtype=float),
            dims=("level",),
            coords={"level": theta_mask.coords["level"].values},
            name="level_center_pressure",
        ).broadcast_like(dp_above)
        upper_to_center = xr.where(
            p_center > p_top,
            upper_face + ((p_bottom - p_top) / (p_center - p_top)) * (center_geopotential - upper_face),
            linear_bottom,
        )
        center_to_lower = xr.where(
            p_lower > p_center,
            center_geopotential
            + ((p_bottom - p_center) / (p_lower - p_center)) * (lower_face - center_geopotential),
            linear_bottom,
        )
        bottom_face = xr.where(p_bottom <= p_center, upper_to_center, center_to_lower)
        reconstruction = "interface_geopotential_faces_center_linear_partial_bottom"
    else:
        bottom_face = linear_bottom

    top_star = _weighted_representative_eddy_on_faces(upper_face, measure.cell_fraction)
    bottom_star = _weighted_representative_eddy_on_faces(bottom_face, measure.cell_fraction)
    top_star.name = "phi_top_star_from_interface"
    bottom_star.name = "phi_bottom_star_from_interface"
    return top_star, bottom_star, reconstruction


def _meridional_gradient_zonal(field: xr.DataArray, *, radius: float) -> xr.DataArray:
    """Return the meridional component ``(1 / a) ∂field/∂φ`` for a zonal field."""

    latitude_deg = field.coords["latitude"]
    field_rad = field.assign_coords(latitude=np.deg2rad(latitude_deg.values))
    edge_order = 2 if field.sizes["latitude"] > 2 else 1
    derivative = field_rad.differentiate("latitude", edge_order=edge_order)
    return (derivative / radius).assign_coords(latitude=latitude_deg.values)


def _metric_factors(latitude: xr.DataArray, *, constants: MarsConstants) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return ``cos(phi)``, ``tan(phi)``, and ``a cos(phi)`` on the latitude grid."""

    lat_radians = np.deg2rad(latitude.values.astype(float))
    cosphi = xr.DataArray(
        np.cos(lat_radians),
        dims=("latitude",),
        coords={"latitude": latitude.values},
        name="cosphi",
    )
    if np.any(np.abs(cosphi.values) <= _METRIC_TOL):
        raise ValueError("C_K terms require latitude points away from exact poles because a*cos(phi) appears in the denominator.")

    sinphi = xr.DataArray(
        np.sin(lat_radians),
        dims=("latitude",),
        coords={"latitude": latitude.values},
        name="sinphi",
    )
    tanphi = sinphi / cosphi
    metric = constants.a * cosphi
    return cosphi, tanphi, metric


def _inverse_exner_from_level(level: xr.DataArray, *, constants: MarsConstants) -> xr.DataArray:
    """Return ``theta / T = (p00 / p)^kappa`` on pressure levels."""

    return (constants.p00 / level) ** constants.kappa


def _exner_from_level(level: xr.DataArray, *, constants: MarsConstants) -> xr.DataArray:
    """Return ``T / theta = (p / p00)^kappa`` on pressure levels."""

    return (level / constants.p00) ** constants.kappa


def _zonal_advective_operator(
    meridional_flux: xr.DataArray,
    vertical_flux: xr.DataArray,
    scalar: xr.DataArray,
    *,
    constants: MarsConstants,
) -> xr.DataArray:
    """Apply ``F_y (1/a)∂_phi + F_p ∂_p`` to a zonal scalar field."""

    meridional_term = meridional_flux * _meridional_gradient_zonal(scalar, radius=constants.a)
    vertical_term = vertical_flux * _pressure_derivative(scalar)
    return meridional_term + vertical_term


def _normalize_valid_mask(field: xr.DataArray, valid_mask: xr.DataArray | None) -> xr.DataArray:
    field = normalize_field(field, "field")
    if valid_mask is None:
        return xr.ones_like(field, dtype=bool)

    valid_mask = normalize_field(valid_mask, "valid_mask")
    ensure_matching_coordinates(field, [valid_mask])
    return valid_mask.astype(bool)


def _periodic_segment_indices(valid: np.ndarray) -> list[np.ndarray]:
    valid_index = np.flatnonzero(valid)
    if valid_index.size == 0:
        return []

    splits = np.where(np.diff(valid_index) > 1)[0] + 1
    segments = [np.asarray(segment, dtype=int) for segment in np.split(valid_index, splits)]
    if len(segments) > 1 and segments[0][0] == 0 and segments[-1][-1] == valid.size - 1:
        segments[0] = np.concatenate([segments[-1], segments[0]])
        segments.pop()
    return segments


def _unwrap_periodic_coordinate(coordinate: np.ndarray, indices: np.ndarray, *, period: float) -> np.ndarray:
    unwrapped = np.asarray(coordinate[indices], dtype=float).copy()
    for idx in range(1, unwrapped.size):
        while unwrapped[idx] <= unwrapped[idx - 1]:
            unwrapped[idx] += period
    return unwrapped


def _periodic_derivative_all_valid(values: np.ndarray, coordinate: np.ndarray, *, period: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size == 1:
        return result
    if np.all(values == values[0]):
        result[:] = 0.0
        return result
    if values.size == 2:
        slope = (values[1] - values[0]) / (coordinate[1] - coordinate[0])
        result[:] = slope
        return result

    for idx in range(values.size):
        prev_idx = (idx - 1) % values.size
        next_idx = (idx + 1) % values.size
        x_prev = float(coordinate[prev_idx])
        x_curr = float(coordinate[idx])
        x_next = float(coordinate[next_idx])
        if prev_idx > idx:
            x_prev -= period
        if next_idx < idx:
            x_next += period
        result[idx] = np.gradient(
            np.asarray([values[prev_idx], values[idx], values[next_idx]], dtype=float),
            np.asarray([x_prev, x_curr, x_next], dtype=float),
            edge_order=2,
        )[1]
    return result


def _segmented_periodic_derivative_1d(
    values: np.ndarray,
    coordinate: np.ndarray,
    valid_mask: np.ndarray,
    *,
    period: float,
) -> np.ndarray:
    """Differentiate a masked cyclic 1D field on contiguous valid segments.

    Fully valid rings use a periodic centered stencil. Open segments use a
    centered interior with one-sided boundaries, two-point segments reduce to a
    first-order slope, and singleton segments are left unsupported as ``NaN``.
    """

    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if not np.any(valid):
        return result

    if np.all(valid):
        return _periodic_derivative_all_valid(values, coordinate, period=period)

    for indices in _periodic_segment_indices(valid):
        if indices.size == 1:
            continue
        segment_values = values[indices]
        segment_coordinate = _unwrap_periodic_coordinate(coordinate, indices, period=period)
        if np.all(segment_values == segment_values[0]):
            result[indices] = 0.0
            continue
        edge_order = 2 if segment_values.size > 2 else 1
        result[indices] = np.gradient(segment_values, segment_coordinate, edge_order=edge_order)
    return result


def _longitude_gradient(
    field: xr.DataArray,
    *,
    constants: MarsConstants,
    valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return ``(1 / a cos(phi)) ∂field/∂lambda`` on the masked above-ground domain."""

    latitude = field.coords["latitude"]
    longitude_deg = field.coords["longitude"]
    _, _, metric = _metric_factors(latitude, constants=constants)
    valid_mask = _normalize_valid_mask(field, valid_mask)
    longitude_rad = xr.DataArray(
        np.deg2rad(longitude_deg.values),
        dims=("longitude",),
        coords={"longitude": longitude_deg.values},
        name="longitude_radians",
    )
    derivative = xr.apply_ufunc(
        _segmented_periodic_derivative_1d,
        field.astype(float),
        longitude_rad.astype(float),
        valid_mask,
        kwargs={"period": 2.0 * np.pi},
        input_core_dims=[["longitude"], ["longitude"], ["longitude"]],
        output_core_dims=[["longitude"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).transpose(*field.dims)
    return (derivative / metric).assign_coords(longitude=longitude_deg.values)


def _longitude_derivative(field: xr.DataArray) -> xr.DataArray:
    """Return ``∂field/∂lambda`` on the full periodic longitude ring."""

    field = normalize_field(field, "field")
    longitude_deg = field.coords["longitude"]
    longitude_rad = xr.DataArray(
        np.deg2rad(longitude_deg.values),
        dims=("longitude",),
        coords={"longitude": longitude_deg.values},
        name="longitude_radians",
    )
    derivative = xr.apply_ufunc(
        _segmented_periodic_derivative_1d,
        field.astype(float),
        longitude_rad.astype(float),
        xr.ones_like(field, dtype=bool),
        kwargs={"period": 2.0 * np.pi},
        input_core_dims=[["longitude"], ["longitude"], ["longitude"]],
        output_core_dims=[["longitude"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).transpose(*field.dims)
    derivative.name = f"d{field.name}_dlambda" if field.name else "longitude_derivative"
    return derivative.assign_coords(longitude=longitude_deg.values)


def _meridional_gradient(
    field: xr.DataArray,
    *,
    constants: MarsConstants,
    valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return ``(1 / a) ∂field/∂phi`` on contiguous valid latitude segments."""

    latitude_deg = field.coords["latitude"]
    valid_mask = _normalize_valid_mask(field, valid_mask)
    latitude_rad = xr.DataArray(
        np.deg2rad(latitude_deg.values),
        dims=("latitude",),
        coords={"latitude": latitude_deg.values},
        name="latitude_radians",
    )
    derivative = coordinate_derivative(
        field,
        "latitude",
        coordinate=latitude_rad,
        valid_mask=valid_mask,
        name=f"d{field.name}_dlat" if field.name else "meridional_gradient",
    )
    return (derivative / constants.a).assign_coords(latitude=latitude_deg.values)


def _latitude_derivative(field: xr.DataArray) -> xr.DataArray:
    """Return ``∂field/∂phi`` on the full latitude coordinate."""

    field = normalize_field(field, "field")
    latitude_deg = field.coords["latitude"]
    latitude_rad = xr.DataArray(
        np.deg2rad(latitude_deg.values),
        dims=("latitude",),
        coords={"latitude": latitude_deg.values},
        name="latitude_radians",
    )
    return coordinate_derivative(
        field,
        "latitude",
        coordinate=latitude_rad,
        name=f"d{field.name}_dphi" if field.name else "latitude_derivative",
    ).assign_coords(latitude=latitude_deg.values)


def _area_weighted_zonal_mean(field: xr.DataArray, integrator: MassIntegrator) -> xr.DataArray:
    """Return the finite-volume area-weighted zonal mean of a full-grid field."""

    field = normalize_field(field, "field")
    integrator._ensure_full_grid_matches(field)
    area = integrator.cell_area.broadcast_like(field)
    numerator = (field * area).sum(dim="longitude")
    band_area = integrator.zonal_band_area.broadcast_like(numerator)
    return xr.where(band_area > 0.0, numerator / band_area, 0.0)


def _ck2_finite_volume_terms(
    phi_star: xr.DataArray | None,
    measure: TopographyAwareMeasure,
    integrator: MassIntegrator,
    *,
    constants: MarsConstants,
    phi_top_star: xr.DataArray | None = None,
    phi_bottom_star: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return cut-cell finite-volume CK2 directional terms."""

    dp_above = measure.above_ground_dp
    layer_valid = (dp_above > 0.0).rename("ck2_layer_valid")
    delta_p = integrator.delta_p.broadcast_like(dp_above)
    p_top = measure.upper_edge.broadcast_like(dp_above)
    p_bottom = measure.effective_bottom_pressure

    if (phi_top_star is None) != (phi_bottom_star is None):
        raise ValueError("'phi_top_star' and 'phi_bottom_star' must be provided together.")

    if phi_top_star is None:
        if phi_star is None:
            raise ValueError("Either 'phi_star' or explicit CK2 face-star fields must be provided.")
        phi_star = normalize_field(phi_star, "phi_star")
        ensure_matching_coordinates(phi_star, [measure.cell_fraction])
        source_valid = xr.apply_ufunc(
            np.isfinite,
            phi_star,
            dask="parallelized",
            output_dtypes=[bool],
        )
        phi_top = _interp_field_to_pressure(
            phi_star,
            p_top,
            valid_mask=source_valid,
            target_valid_mask=layer_valid,
        )
        phi_bottom = _interp_field_to_pressure(
            phi_star,
            p_bottom,
            valid_mask=source_valid,
            target_valid_mask=layer_valid,
        )
        reference = phi_star
    else:
        phi_top = normalize_field(phi_top_star, "phi_top_star")
        phi_bottom = normalize_field(phi_bottom_star, "phi_bottom_star")
        ensure_matching_coordinates(phi_top, [phi_bottom, measure.cell_fraction])
        reference = phi_top

    support = layer_valid & xr.apply_ufunc(np.isfinite, phi_top, dask="parallelized", output_dtypes=[bool])
    support = support & xr.apply_ufunc(np.isfinite, phi_bottom, dask="parallelized", output_dtypes=[bool])

    phi_bottom_eff = xr.where(support, phi_bottom, 0.0)
    phi_top_eff = xr.where(support, phi_top, 0.0)
    layer_integral = xr.where(support, 0.5 * (phi_top_eff + phi_bottom_eff) * dp_above, 0.0)
    layer_integral.name = "ck2_phi_star_layer_integral"

    dI_dt = time_derivative(layer_integral)
    dpb_dt = time_derivative(p_bottom)
    dI_dlambda = _longitude_derivative(layer_integral)
    dpb_dlambda = _longitude_derivative(p_bottom)
    dI_dphi = _latitude_derivative(layer_integral)
    dpb_dphi = _latitude_derivative(p_bottom)

    _, _, metric = _metric_factors(reference.coords["latitude"], constants=constants)
    tendency_full = (dI_dt - phi_bottom_eff * dpb_dt) / delta_p
    gradient_x_full = (dI_dlambda - phi_bottom_eff * dpb_dlambda) / (metric * delta_p)
    gradient_y_full = (dI_dphi - phi_bottom_eff * dpb_dphi) / (constants.a * delta_p)
    pressure_full = (phi_bottom_eff - phi_top_eff) / delta_p

    tendency_full = xr.where(support, tendency_full, 0.0)
    gradient_x_full = xr.where(support, gradient_x_full, 0.0)
    gradient_y_full = xr.where(support, gradient_y_full, 0.0)
    pressure_full = xr.where(support, pressure_full, 0.0)

    tendency_term = _area_weighted_zonal_mean(tendency_full, integrator)
    gradient_x_term = _area_weighted_zonal_mean(gradient_x_full, integrator)
    gradient_y_term = _area_weighted_zonal_mean(gradient_y_full, integrator)
    pressure_term = _area_weighted_zonal_mean(pressure_full, integrator)
    tendency_term.name = "ck2_tendency_term"
    gradient_x_term.name = "ck2_gradient_x_term"
    gradient_y_term.name = "ck2_gradient_y_term"
    pressure_term.name = "ck2_pressure_term"
    return tendency_term, gradient_x_term, gradient_y_term, pressure_term


def conversion_zonal_ape_to_ke_part1(
    omega: xr.DataArray,
    alpha: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return ``C_Z1 = - ∫_M [Theta] [omega]_R [alpha]_R dm`` in Watts."""

    integrator = _require_integrator(integrator)
    omega = normalize_field(omega, "omega")
    alpha = normalize_field(alpha, "alpha")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(omega, [alpha, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_Z1",
    )

    coverage = _coverage_field(theta_mask, measure)
    omega_r = _representative_mean(omega, theta_mask, measure)
    alpha_r = _representative_mean(alpha, theta_mask, measure)
    integrand = -coverage * omega_r * alpha_r
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "C_Z1"
    return _annotate_quantity(result, units="W", base_quantity="power", measure=measure)


def conversion_zonal_ape_to_ke_part2(
    ps: xr.DataArray,
    phis: xr.DataArray,
    integrator: MassIntegrator,
    *,
    measure: TopographyAwareMeasure | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return ``C_Z2 = - ∫_S (dps/dt * Phi_s) dσ / g`` in Watts."""

    ps = normalize_surface_field(ps, "ps")
    phis = normalize_surface_field(phis, "phis")
    ensure_matching_surface_coordinates(ps, [phis])
    measure = _resolved_measure(
        integrator,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_Z2",
    )

    ps_effective = _effective_surface_pressure(ps, measure)
    integrand = -time_derivative(ps_effective) * phis
    result = integrator.integrate_surface(integrand)
    result.name = "C_Z2"
    return _annotate_quantity(result, units="W", base_quantity="power", measure=measure)


def conversion_zonal_ape_to_ke(
    omega: xr.DataArray,
    alpha: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return the total ``C_Z = C_Z1 + C_Z2`` in Watts."""

    integrator = _require_integrator(integrator)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    if ps is None or phis is None:
        raise ValueError("Total C_Z requires both 'ps' and 'phis'.")
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_Z",
    )
    result = conversion_zonal_ape_to_ke_part1(
        omega,
        alpha,
        theta_mask,
        integrator,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
    ) + conversion_zonal_ape_to_ke_part2(
        ps,
        phis,
        integrator,
        measure=measure,
        surface_pressure_policy=surface_pressure_policy,
    )
    result.name = "C_Z"
    return _annotate_quantity(result, units="W", base_quantity="power", measure=measure)


def conversion_zonal_ape_to_eddy_ape(
    temperature: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    n_z: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return ``C_A`` in Watts."""

    integrator = _require_integrator(integrator)
    temperature = normalize_field(temperature, "temperature")
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    omega = normalize_field(omega, "omega")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    n_z = _coerce_to_zonal(n_z, "n_z")
    ensure_matching_coordinates(temperature, [u, v, omega, theta_mask])
    _ensure_matching_zonal_coordinates(temperature, n_z, "n_z")
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_A",
    )

    weight = _weight_field(theta_mask, measure)
    coverage = _coverage_field(theta_mask, measure)
    temperature_star = _representative_eddy_component(temperature, theta_mask, measure)
    v_star = _representative_eddy_component(v, theta_mask, measure)
    omega_star = _representative_eddy_component(omega, theta_mask, measure)

    meridional_flux = zonal_mean(weight * temperature_star * v_star)
    vertical_flux = zonal_mean(weight * temperature_star * omega_star)

    pressure = n_z.coords["level"]
    scalar = _exner_from_level(pressure, constants=constants) * n_z
    inverse_exner = _inverse_exner_from_level(pressure, constants=constants)
    advective_tendency = _zonal_advective_operator(
        meridional_flux,
        vertical_flux,
        scalar,
        constants=constants,
    )

    integrand = -constants.cp * inverse_exner * advective_tendency
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "C_A"
    return _annotate_quantity(result, units="W", base_quantity="power", measure=measure)


def conversion_eddy_ape_to_ke(
    omega: xr.DataArray,
    alpha: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    """Return ``C_E = - ∫_M [Theta omega* alpha*] dm`` in Watts."""

    integrator = _require_integrator(integrator)
    omega = normalize_field(omega, "omega")
    alpha = normalize_field(alpha, "alpha")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(omega, [alpha, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_E",
    )

    weight = _weight_field(theta_mask, measure)
    coverage = _coverage_field(theta_mask, measure)
    omega_star = _representative_eddy_component(omega, theta_mask, measure)
    alpha_star = _representative_eddy_component(alpha, theta_mask, measure)
    integrand = -zonal_mean(weight * omega_star * alpha_star)
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "C_E"
    return _annotate_quantity(result, units="W", base_quantity="power", measure=measure)


def conversion_zonal_ke_to_eddy_ke_part1(
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return ``C_K1`` in Watts."""

    integrator = _require_integrator(integrator)
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    omega = normalize_field(omega, "omega")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(u, [v, omega, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_K1",
    )

    _, tanphi, metric = _metric_factors(u.coords["latitude"], constants=constants)

    weight = _weight_field(theta_mask, measure)
    coverage = _coverage_field(theta_mask, measure)
    u_r = _representative_mean(u, theta_mask, measure)
    v_r = _representative_mean(v, theta_mask, measure)
    u_star = _representative_eddy_component(u, theta_mask, measure)
    v_star = _representative_eddy_component(v, theta_mask, measure)
    omega_star = _representative_eddy_component(omega, theta_mask, measure)

    mean_u_shear = u_r / metric
    mean_v_shear = v_r / metric

    uv_flux = zonal_mean(weight * u_star * v_star)
    uomega_flux = zonal_mean(weight * u_star * omega_star)
    vv_flux = zonal_mean(weight * v_star * v_star)
    vomega_flux = zonal_mean(weight * v_star * omega_star)
    eddy_speed_sq = zonal_mean(weight * (u_star * u_star + v_star * v_star))

    u_block = _zonal_advective_operator(
        uv_flux,
        uomega_flux,
        mean_u_shear,
        constants=constants,
    )
    v_block = _zonal_advective_operator(
        vv_flux,
        vomega_flux,
        mean_v_shear,
        constants=constants,
    ) - (tanphi / constants.a) * eddy_speed_sq * mean_v_shear

    integrand = -metric * (u_block + v_block)
    result = _integrate_zonal_mass_aware(integrand, coverage, integrator, measure)
    result.name = "C_K1"
    return _annotate_quantity(result, units="W", base_quantity="power", measure=measure)


def conversion_zonal_ke_to_eddy_ke_part2(
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    geopotential: xr.DataArray | None = None,
    interface_geopotential: xr.DataArray | None = None,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    geopotential_mode: str = "strict",
    allow_geopotential_reconstruction: bool | None = None,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return ``C_K2`` in Watts.

    ``C_K2`` is evaluated with a cut-cell finite-volume discretization. The
    pressure-layer integral of ``Phi*`` is reconstructed to the effective
    above-ground bottom pressure and differentiated with the corresponding
    Leibniz lower-boundary correction. Pass ``geopotential`` or
    ``interface_geopotential`` from the GCM whenever available. Hydrostatic
    reconstruction from ``temperature``, ``pressure``, ``ps``, and ``phis`` is
    approximate and requires ``geopotential_mode='hydrostatic'``. Existing
    ``allow_geopotential_reconstruction=True`` calls are accepted for one
    transition window with a ``FutureWarning``.
    """

    geopotential_mode = resolve_deprecated_geopotential_mode(
        geopotential_mode,
        allow_geopotential_reconstruction,
        deprecated_name="allow_geopotential_reconstruction",
    )
    integrator = _require_integrator(integrator)
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    omega = normalize_field(omega, "omega")
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    ensure_matching_coordinates(u, [v, omega, theta_mask])
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_K2",
    )
    valid_mask = (theta_mask > 0.0).rename("theta_valid_mask")

    phi_star = None
    phi_top_star = None
    phi_bottom_star = None
    ck2_reconstruction = "linear_pressure_phi_star_to_layer_faces"
    ck2_geopotential_source = "level_center_geopotential"
    ck2_geopotential_reconstruction_approximate = False
    if interface_geopotential is not None:
        center_geopotential = None
        if geopotential is not None:
            center_geopotential = normalize_field(geopotential, "geopotential")
            ensure_matching_coordinates(u, [center_geopotential])
        (
            phi_top_star,
            phi_bottom_star,
            ck2_reconstruction,
        ) = _ck2_interface_geopotential_face_stars(
            interface_geopotential,
            theta_mask,
            measure,
            center_geopotential=center_geopotential,
            phis=phis,
        )
        ck2_geopotential_source = "interface_geopotential"
    else:
        phi = resolve_geopotential(
            geopotential=geopotential,
            temperature=temperature,
            pressure=pressure,
            ps=ps,
            phis=phis,
            theta_mask=theta_mask,
            valid_mask=valid_mask,
            geopotential_mode=geopotential_mode,
            constants=constants,
        )
        ck2_geopotential_source = _ck2_level_center_geopotential_source(
            phi,
        )
        ck2_geopotential_reconstruction_approximate = _ck2_geopotential_reconstruction_approximate(phi)
        ensure_matching_coordinates(u, [phi])
        phi = phi.where(valid_mask)
        phi_star = _weighted_representative_eddy_on_faces(phi, _weight_field(theta_mask, measure))

    (
        tendency_term,
        gradient_x_term,
        gradient_y_term,
        pressure_term,
    ) = _ck2_finite_volume_terms(
        phi_star,
        measure,
        integrator,
        constants=constants,
        phi_top_star=phi_top_star,
        phi_bottom_star=phi_bottom_star,
    )

    u_r = _representative_mean(u, theta_mask, measure)
    v_r = _representative_mean(v, theta_mask, measure)
    omega_r = _representative_mean(omega, theta_mask, measure)

    integrand = tendency_term + u_r * gradient_x_term + v_r * gradient_y_term + omega_r * pressure_term
    result = integrator.integrate_zonal(integrand)
    result.name = "C_K2"
    return _annotate_quantity(
        result,
        units="W",
        base_quantity="power",
        measure=measure,
        extra_attrs=_ck2_boundary_attrs(
            reconstruction=ck2_reconstruction,
            geopotential_source=ck2_geopotential_source,
            geopotential_mode=geopotential_mode,
            geopotential_reconstruction_allowed=geopotential_mode != "strict",
            geopotential_reconstruction_approximate=ck2_geopotential_reconstruction_approximate,
        ),
    )


def conversion_zonal_ke_to_eddy_ke(
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    theta_mask: xr.DataArray | None = None,
    integrator: MassIntegrator | None = None,
    *,
    theta: xr.DataArray | None = None,
    measure: TopographyAwareMeasure | None = None,
    geopotential: xr.DataArray | None = None,
    interface_geopotential: xr.DataArray | None = None,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    geopotential_mode: str = "strict",
    allow_geopotential_reconstruction: bool | None = None,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return the total ``C_K = C_K1 + C_K2`` in Watts.

    The total inherits the same finite-volume provenance as ``C_K2`` because
    the topographic contribution is computed with the cut-cell CK2 operator.
    Hydrostatic geopotential reconstruction is approximate and is disabled
    unless ``geopotential_mode='hydrostatic'``. Existing
    ``allow_geopotential_reconstruction=True`` calls are accepted for one
    transition window with a ``FutureWarning``.
    """

    geopotential_mode = resolve_deprecated_geopotential_mode(
        geopotential_mode,
        allow_geopotential_reconstruction,
        deprecated_name="allow_geopotential_reconstruction",
    )
    integrator = _require_integrator(integrator)
    theta_mask = resolve_deprecated_theta_mask(theta_mask, theta)
    measure = _resolved_measure(
        integrator,
        theta_mask=theta_mask,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        diagnostic_name="C_K",
    )
    ck1 = conversion_zonal_ke_to_eddy_ke_part1(
        u,
        v,
        omega,
        theta_mask,
        integrator,
        measure=measure,
        ps=ps,
        surface_pressure_policy=surface_pressure_policy,
        constants=constants,
    )
    ck2 = conversion_zonal_ke_to_eddy_ke_part2(
        u,
        v,
        omega,
        theta_mask,
        integrator,
        measure=measure,
        geopotential=geopotential,
        interface_geopotential=interface_geopotential,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
        surface_pressure_policy=surface_pressure_policy,
        geopotential_mode=geopotential_mode,
        constants=constants,
    )
    result = ck1 + ck2
    result.name = "C_K"
    ck2_attrs = {
        key: value
        for key, value in ck2.attrs.items()
        if key.startswith("ck2_")
    }
    return _annotate_quantity(
        result,
        units="W",
        base_quantity="power",
        measure=measure,
        extra_attrs=ck2_attrs,
    )


C_Z1 = conversion_zonal_ape_to_ke_part1
C_Z2 = conversion_zonal_ape_to_ke_part2
C_Z = conversion_zonal_ape_to_ke
C_A = conversion_zonal_ape_to_eddy_ape
C_E = conversion_eddy_ape_to_ke
C_K1 = conversion_zonal_ke_to_eddy_ke_part1
C_K2 = conversion_zonal_ke_to_eddy_ke_part2
C_K = conversion_zonal_ke_to_eddy_ke


__all__ = [
    "conversion_zonal_ape_to_ke_part1",
    "conversion_zonal_ape_to_ke_part2",
    "conversion_zonal_ape_to_ke",
    "conversion_eddy_ape_to_ke",
    "conversion_zonal_ape_to_eddy_ape",
    "conversion_zonal_ke_to_eddy_ke_part1",
    "conversion_zonal_ke_to_eddy_ke_part2",
    "conversion_zonal_ke_to_eddy_ke",
    "C_Z1",
    "C_Z2",
    "C_Z",
    "C_E",
    "C_A",
    "C_K1",
    "C_K2",
    "C_K",
]
