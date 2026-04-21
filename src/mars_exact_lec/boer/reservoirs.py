"""Phase-1 kinetic-energy reservoirs for the exact Boer decomposition."""

from __future__ import annotations

import xarray as xr

from .._validation import ensure_matching_coordinates, normalize_field
from ..common.integrals import MassIntegrator
from ..common.zonal_ops import representative_eddy, representative_zonal_mean, theta_coverage, zonal_mean


def total_horizontal_ke(
    u: xr.DataArray,
    v: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return ``∫_M 0.5 Theta (u² + v²) dm`` in Joules."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, theta])

    integrand = 0.5 * theta * (u**2 + v**2)
    result = integrator.integrate_full(integrand)
    result.name = "total_horizontal_ke"
    result.attrs["units"] = "J"
    return result


def kinetic_energy_zonal(
    u: xr.DataArray,
    v: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return the zonal kinetic-energy reservoir ``K_Z`` in Joules."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, theta])

    coverage = theta_coverage(theta)
    u_r = representative_zonal_mean(u, theta)
    v_r = representative_zonal_mean(v, theta)
    integrand = 0.5 * coverage * (u_r**2 + v_r**2)
    result = integrator.integrate_zonal(integrand)
    result.name = "K_Z"
    result.attrs["units"] = "J"
    return result


def kinetic_energy_eddy(
    u: xr.DataArray,
    v: xr.DataArray,
    theta: xr.DataArray,
    integrator: MassIntegrator,
) -> xr.DataArray:
    """Return the eddy kinetic-energy reservoir ``K_E`` in Joules."""

    u = normalize_field(u, "u")
    v = normalize_field(v, "v")
    theta = normalize_field(theta, "theta")
    ensure_matching_coordinates(u, [v, theta])

    u_star = representative_eddy(u, theta)
    v_star = representative_eddy(v, theta)
    integrand = 0.5 * zonal_mean(theta * (u_star**2 + v_star**2))
    result = integrator.integrate_zonal(integrand)
    result.name = "K_E"
    result.attrs["units"] = "J"
    return result


__all__ = ["total_horizontal_ke", "kinetic_energy_zonal", "kinetic_energy_eddy"]
