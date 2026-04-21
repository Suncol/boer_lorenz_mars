"""Phase-1 APE-to-KE conversion terms that do not depend on a reference state."""

from __future__ import annotations

import xarray as xr

from .._validation import ensure_matching_coordinates, normalize_field
from ..common.integrals import MassIntegrator
from ..common.zonal_ops import representative_eddy, representative_zonal_mean, theta_coverage, zonal_mean


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


__all__ = ["conversion_zonal_ape_to_ke_part1", "conversion_eddy_ape_to_ke"]
