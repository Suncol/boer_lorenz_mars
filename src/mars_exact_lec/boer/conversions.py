"""Boer exact conversion terms used by the Mars exact Lorenz cycle branch."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._validation import ensure_matching_coordinates, normalize_field, normalize_zonal_field
from ..constants_mars import MARS, MarsConstants
from ..common.integrals import MassIntegrator
from ..common.zonal_ops import representative_eddy, representative_zonal_mean, theta_coverage, zonal_mean


_METRIC_TOL = 1.0e-12


def _coerce_to_zonal(field: xr.DataArray, name: str) -> xr.DataArray:
    """Accept either a zonal field or a longitude-constant full field."""

    try:
        return normalize_zonal_field(field, name)
    except ValueError:
        return zonal_mean(normalize_field(field, name))


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


def _pressure_derivative(field: xr.DataArray) -> xr.DataArray:
    """Return ``∂field/∂p`` using the canonical pressure coordinate."""

    edge_order = 2 if field.sizes["level"] > 2 else 1
    return field.differentiate("level", edge_order=edge_order)


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
        raise ValueError("C_K1 requires latitude points away from exact poles because a*cos(phi) appears in the denominator.")

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
    """Return ``C_A`` in Watts.

    The implementation follows the normalized eq. (8):

    ``C_A = - ∫ cp (theta/T) ([Theta T* V*]·∇ + [Theta T* omega*] ∂_p) ((T/theta) N_Z) dm``

    where lower-case ``theta`` is potential temperature and upper-case ``Theta``
    is the terrain mask carried by the ``theta`` argument in this codebase.
    Since ``theta/T = (p00/p)^kappa`` and ``T/theta = (p/p00)^kappa`` in
    pressure coordinates, both thermodynamic factors are evaluated from the
    pressure-level coordinate, avoiding an unnecessary division by temperature.
    """

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
    """Return ``C_K1`` in Watts.

    This follows the normalized eq. (5), interpreted as the sum of two
    mean-shear contributions, not their product. Because the shears
    ``[u]_R / (a cos(phi))`` and ``[v]_R / (a cos(phi))`` are zonal by
    construction, the longitudinal-gradient contribution vanishes and only the
    meridional and pressure derivatives are retained.
    """

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


C_A = conversion_zonal_ape_to_eddy_ape
C_K1 = conversion_zonal_ke_to_eddy_ke_part1


__all__ = [
    "conversion_zonal_ape_to_ke_part1",
    "conversion_eddy_ape_to_ke",
    "conversion_zonal_ape_to_eddy_ape",
    "conversion_zonal_ke_to_eddy_ke_part1",
    "C_A",
    "C_K1",
]
