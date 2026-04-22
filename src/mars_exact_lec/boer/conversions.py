"""Boer exact conversion terms used by the Mars exact Lorenz cycle branch."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._validation import (
    ensure_matching_coordinates,
    ensure_matching_surface_coordinates,
    normalize_field,
    normalize_surface_field,
    normalize_zonal_field,
)
from ..common.geopotential import resolve_geopotential
from ..common.integrals import MassIntegrator
from ..common.time_derivatives import coordinate_derivative, time_derivative
from ..common.zonal_ops import representative_eddy, representative_zonal_mean, theta_coverage, zonal_mean
from ..constants_mars import MARS, MarsConstants


_METRIC_TOL = 1.0e-12


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
        edge_order = 2 if segment_values.size > 2 else 1
        result[indices] = np.gradient(segment_values, segment_coordinate, edge_order=edge_order)
    return result


def _longitude_gradient(
    field: xr.DataArray,
    *,
    constants: MarsConstants,
    valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
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


def _meridional_gradient(
    field: xr.DataArray,
    *,
    constants: MarsConstants,
    valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
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


def conversion_zonal_ape_to_ke_part1(
    omega: xr.DataArray,
    alpha: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return ``C_Z1 = - ∫_M [Theta] [omega]_R [alpha]_R dm`` in Watts."""

    omega = normalize_field(omega, "omega")
    alpha = normalize_field(alpha, "alpha")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(omega, [alpha, theta])

    coverage = theta_coverage(theta)
    omega_r = representative_zonal_mean(omega, theta)
    alpha_r = representative_zonal_mean(alpha, theta)
    integrand = -coverage * omega_r * alpha_r
    result = integrator.integrate_zonal(integrand)
    result.name = "C_Z1"
    result.attrs["units"] = "W"
    return result


def conversion_zonal_ape_to_ke_part2(
    ps: xr.DataArray,
    phis: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return ``C_Z2 = - ∫_S (dps/dt * Phi_s) dσ / g`` in Watts."""

    ps = normalize_surface_field(ps, "ps")
    phis = normalize_surface_field(phis, "phis")
    ensure_matching_surface_coordinates(ps, [phis])

    integrand = -time_derivative(ps) * phis
    result = integrator.integrate_surface(integrand)
    result.name = "C_Z2"
    result.attrs["units"] = "W"
    return result


def conversion_zonal_ape_to_ke(
    omega: xr.DataArray,
    alpha: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    *,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return the total ``C_Z = C_Z1 + C_Z2`` in Watts."""

    if ps is None or phis is None:
        raise ValueError("Total C_Z requires both 'ps' and 'phis'.")
    result = conversion_zonal_ape_to_ke_part1(omega, alpha, theta, integrator) + conversion_zonal_ape_to_ke_part2(
        ps,
        phis,
        integrator,
    )
    result.name = "C_Z"
    result.attrs["units"] = "W"
    return result


def conversion_zonal_ape_to_eddy_ape(
    temperature: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    n_z: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    *,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return ``C_A`` in Watts."""

    temperature = normalize_field(temperature, "temperature")
    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    omega = normalize_field(omega, "omega")
    theta = normalize_field(theta, "theta")
    n_z = _coerce_to_zonal(n_z, "n_z")
    ensure_matching_coordinates(temperature, [u, v, omega, theta])
    _ensure_matching_zonal_coordinates(temperature, n_z, "n_z")

    temperature_star = representative_eddy(temperature, theta)
    v_star = representative_eddy(v, theta)
    omega_star = representative_eddy(omega, theta)

    meridional_flux = zonal_mean(theta * temperature_star * v_star)
    vertical_flux = zonal_mean(theta * temperature_star * omega_star)

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
    result = integrator.integrate_zonal(integrand)
    result.name = "C_A"
    result.attrs["units"] = "W"
    return result


def conversion_eddy_ape_to_ke(
    omega: xr.DataArray,
    alpha: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return ``C_E = - ∫_M [Theta omega* alpha*] dm`` in Watts."""

    omega = normalize_field(omega, "omega")
    alpha = normalize_field(alpha, "alpha")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(omega, [alpha, theta])

    omega_star = representative_eddy(omega, theta)
    alpha_star = representative_eddy(alpha, theta)
    integrand = -zonal_mean(theta * omega_star * alpha_star)
    result = integrator.integrate_zonal(integrand)
    result.name = "C_E"
    result.attrs["units"] = "W"
    return result


def conversion_zonal_ke_to_eddy_ke_part1(
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    *,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return ``C_K1`` in Watts."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    omega = normalize_field(omega, "omega")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, omega, theta])

    _, tanphi, metric = _metric_factors(u.coords["latitude"], constants=constants)

    u_r = representative_zonal_mean(u, theta)
    v_r = representative_zonal_mean(v, theta)
    u_star = representative_eddy(u, theta)
    v_star = representative_eddy(v, theta)
    omega_star = representative_eddy(omega, theta)

    mean_u_shear = u_r / metric
    mean_v_shear = v_r / metric

    uv_flux = zonal_mean(theta * u_star * v_star)
    uomega_flux = zonal_mean(theta * u_star * omega_star)
    vv_flux = zonal_mean(theta * v_star * v_star)
    vomega_flux = zonal_mean(theta * v_star * omega_star)
    eddy_speed_sq = zonal_mean(theta * (u_star * u_star + v_star * v_star))

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
    result = integrator.integrate_zonal(integrand)
    result.name = "C_K1"
    result.attrs["units"] = "W"
    return result


def conversion_zonal_ke_to_eddy_ke_part2(
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    *,
    geopotential: xr.DataArray | None = None,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return ``C_K2`` in Watts."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    omega = normalize_field(omega, "omega")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, omega, theta])
    valid_mask = (theta > 0.0).rename("theta_valid_mask")

    phi = resolve_geopotential(
        geopotential=geopotential,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
        theta=theta,
        valid_mask=valid_mask,
        constants=constants,
    )
    ensure_matching_coordinates(u, [phi])
    phi = phi.where(valid_mask)

    # Boer (1989) Eq. (5') is written in terms of [Theta ∂Phi*/∂x], so the
    # derivatives must be taken only on the above-ground domain.
    phi_star = representative_eddy(phi, theta)
    dphi_dt_star = time_derivative(phi_star, valid_mask=valid_mask)
    dphi_dp_star = _pressure_derivative(phi_star, valid_mask=valid_mask)
    dphi_dx_star = _longitude_gradient(phi_star, constants=constants, valid_mask=valid_mask)
    dphi_dy_star = _meridional_gradient(phi_star, constants=constants, valid_mask=valid_mask)

    u_r = representative_zonal_mean(u, theta)
    v_r = representative_zonal_mean(v, theta)
    omega_r = representative_zonal_mean(omega, theta)

    tendency_term = zonal_mean(theta * dphi_dt_star)
    gradient_x_term = zonal_mean(theta * dphi_dx_star)
    gradient_y_term = zonal_mean(theta * dphi_dy_star)
    pressure_term = zonal_mean(theta * dphi_dp_star)

    integrand = tendency_term + u_r * gradient_x_term + v_r * gradient_y_term + omega_r * pressure_term
    result = integrator.integrate_zonal(integrand)
    result.name = "C_K2"
    result.attrs["units"] = "W"
    return result


def conversion_zonal_ke_to_eddy_ke(
    u: xr.DataArray,
    v: xr.DataArray,
    omega: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
    *,
    geopotential: xr.DataArray | None = None,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    ps: xr.DataArray | None = None,
    phis: xr.DataArray | None = None,
    constants: MarsConstants = MARS,
) -> xr.DataArray:
    """Return the total ``C_K = C_K1 + C_K2`` in Watts."""

    result = conversion_zonal_ke_to_eddy_ke_part1(
        u,
        v,
        omega,
        theta,
        integrator,
        constants=constants,
    ) + conversion_zonal_ke_to_eddy_ke_part2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=geopotential,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
        constants=constants,
    )
    result.name = "C_K"
    result.attrs["units"] = "W"
    return result


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
