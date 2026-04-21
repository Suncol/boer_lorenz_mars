"""Mass-conserving phase-2 reference-state solver for Mars exact APE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .._validation import (
    ensure_matching_coordinates,
    normalize_field,
    normalize_zonal_field,
    require_dataarray,
)
from ..common.integrals import build_mass_integrator
from ..constants_mars import MARS, MarsConstants
from ..io.mask_below_ground import make_theta
from .interpolate_isentropes import pressure_level_edges


REFERENCE_SAMPLE_DIM = "reference_sample"


def _normalize_reference_target(field: xr.DataArray, name: str) -> xr.DataArray:
    field = require_dataarray(field, name)
    if "longitude" in field.dims:
        return normalize_field(field, name)
    return normalize_zonal_field(field, name)


def _interp_reference_curve(
    theta_target: np.ndarray,
    theta_reference: np.ndarray,
    pi_reference: np.ndarray,
    lower_fill: float,
    upper_fill: float,
) -> np.ndarray:
    valid = np.isfinite(theta_reference) & np.isfinite(pi_reference)
    count = np.count_nonzero(valid)
    if count == 0:
        return np.full(theta_target.shape, np.nan, dtype=float)
    if count == 1:
        return np.full(theta_target.shape, float(pi_reference[valid][0]), dtype=float)

    theta_curve = theta_reference[valid]
    pi_curve = pi_reference[valid]
    order = np.argsort(theta_curve, kind="mergesort")
    return np.interp(
        theta_target,
        theta_curve[order],
        pi_curve[order],
        left=lower_fill,
        right=upper_fill,
    )


@dataclass(frozen=True)
class ReferenceStateSolution:
    """Reference-state diagnostics returned by :class:`KoehlerReferenceState`.

    The phase-2 implementation stores the globally mass-conserving monotone
    ``pi(theta, t)`` curve and exposes helpers to evaluate it on full fields or
    on representative zonal-mean fields. Explicit terrain-dependent surface
    diagnostics are deferred to phase 3.
    """

    theta_reference: xr.DataArray
    pi_reference: xr.DataArray
    mass_reference: xr.DataArray
    total_mass: xr.DataArray
    reference_surface_pressure: xr.DataArray
    reference_top_pressure: xr.DataArray
    constants: MarsConstants = MARS

    def reference_pressure(
        self,
        potential_temperature: xr.DataArray,
        *,
        name: str = "pi",
    ) -> xr.DataArray:
        """Evaluate ``pi(theta, t)`` on a full or zonal field."""

        potential_temperature = _normalize_reference_target(
            potential_temperature,
            "potential_temperature",
        )

        values = np.empty(potential_temperature.shape, dtype=float)
        for time_index in range(potential_temperature.sizes["time"]):
            theta_target = np.asarray(
                potential_temperature.isel(time=time_index).values,
                dtype=float,
            )
            values[time_index] = _interp_reference_curve(
                theta_target=theta_target.reshape(-1),
                theta_reference=np.asarray(
                    self.theta_reference.isel(time=time_index).values,
                    dtype=float,
                ),
                pi_reference=np.asarray(
                    self.pi_reference.isel(time=time_index).values,
                    dtype=float,
                ),
                lower_fill=float(self.reference_surface_pressure.isel(time=time_index)),
                upper_fill=float(self.reference_top_pressure.isel(time=time_index)),
            ).reshape(theta_target.shape)

        result = xr.DataArray(
            values,
            dims=potential_temperature.dims,
            coords=potential_temperature.coords,
            name=name,
            attrs={"units": "Pa"},
        )
        return result

    def zonal_reference_pressure(self, representative_theta: xr.DataArray) -> xr.DataArray:
        """Evaluate ``pi([theta]_R, t)`` on a zonal field."""

        return self.reference_pressure(representative_theta, name="pi_Z")

    def efficiency(
        self,
        potential_temperature: xr.DataArray,
        pressure: xr.DataArray,
        *,
        name: str = "N",
    ) -> xr.DataArray:
        """Return the efficiency factor ``N = 1 - (pi / p)^kappa``."""

        pressure = _normalize_reference_target(pressure, "pressure")
        pi = self.reference_pressure(potential_temperature)
        efficiency = 1.0 - (pi / pressure) ** self.constants.kappa
        efficiency.name = name
        efficiency.attrs["units"] = "1"
        return efficiency

    def zonal_efficiency(
        self,
        representative_theta: xr.DataArray,
        pressure: xr.DataArray,
    ) -> xr.DataArray:
        """Return ``N_Z = 1 - (pi_Z / p)^kappa`` on a zonal field."""

        pressure = _normalize_reference_target(pressure, "pressure")
        pi_z = self.zonal_reference_pressure(representative_theta)
        n_z = 1.0 - (pi_z / pressure) ** self.constants.kappa
        n_z.name = "N_Z"
        n_z.attrs["units"] = "1"
        return n_z


class KoehlerReferenceState:
    """Solve the phase-2 mass-conserving ``pi(theta, t)`` reference state.

    This implementation targets stage 2 of the project plan:

    - it preserves isentropic-layer mass exactly within the repository's
      discrete pressure-level / sharp-mask convention;
    - it yields a monotone reference curve ``pi(theta, t)`` suitable for
      ``A_Z1``, ``A_E1``, ``C_A``, and future extensions;
    - it does not yet resolve explicit terrain-dependent surface terms
      ``A_2 / C_2`` or the full Koehler lower-boundary iteration.
    """

    def __init__(self, constants: MarsConstants = MARS) -> None:
        self.constants = constants

    def solve(
        self,
        potential_temperature: xr.DataArray,
        pressure: xr.DataArray,
        ps: xr.DataArray,
        phis: xr.DataArray | None = None,
    ) -> ReferenceStateSolution:
        """Return a mass-conserving phase-2 reference-state solution."""

        del phis  # Phase 2 keeps the interface but does not use explicit surface terms.

        potential_temperature = normalize_field(
            potential_temperature,
            "potential_temperature",
        )
        pressure = normalize_field(pressure, "pressure")
        ensure_matching_coordinates(potential_temperature, [pressure])

        theta_mask = make_theta(pressure, ps)
        integrator = build_mass_integrator(
            potential_temperature.coords["level"],
            potential_temperature.coords["latitude"],
            potential_temperature.coords["longitude"],
            constants=self.constants,
        )
        parcel_mass = theta_mask * integrator.full_mass_weights

        level_edges = pressure_level_edges(potential_temperature.coords["level"])
        reference_surface_pressure = xr.full_like(
            potential_temperature.coords["time"],
            float(level_edges.isel(level_edge=0)),
            dtype=float,
        )
        reference_top_pressure = xr.full_like(
            potential_temperature.coords["time"],
            float(level_edges.isel(level_edge=-1)),
            dtype=float,
        )
        planetary_area = float(integrator.cell_area.sum())

        ntime = potential_temperature.sizes["time"]
        max_groups = potential_temperature.sizes["level"] * potential_temperature.sizes["latitude"] * potential_temperature.sizes["longitude"]
        theta_reference = np.full((ntime, max_groups), np.nan, dtype=float)
        pi_reference = np.full((ntime, max_groups), np.nan, dtype=float)
        mass_reference = np.full((ntime, max_groups), np.nan, dtype=float)
        total_mass = np.zeros(ntime, dtype=float)

        theta_values = np.asarray(potential_temperature.values, dtype=float)
        mass_values = np.asarray(parcel_mass.values, dtype=float)
        top_pressure_value = float(level_edges.isel(level_edge=-1))

        for time_index in range(ntime):
            theta_flat = theta_values[time_index].reshape(-1)
            mass_flat = mass_values[time_index].reshape(-1)
            valid = np.isfinite(theta_flat) & np.isfinite(mass_flat) & (mass_flat > 0.0)
            if not np.any(valid):
                raise ValueError("Reference-state solve requires at least one above-ground parcel.")

            theta_valid = theta_flat[valid]
            mass_valid = mass_flat[valid]
            order = np.argsort(theta_valid, kind="mergesort")
            theta_sorted = theta_valid[order]
            mass_sorted = mass_valid[order]

            theta_groups, group_start, _ = np.unique(
                theta_sorted,
                return_index=True,
                return_counts=True,
            )
            group_mass = np.add.reduceat(mass_sorted, group_start)
            cumulative_mass = np.cumsum(group_mass)
            total_mass[time_index] = float(group_mass.sum())
            mass_hotter = total_mass[time_index] - cumulative_mass
            group_pi = top_pressure_value + (
                mass_hotter + 0.5 * group_mass
            ) * self.constants.g / planetary_area

            ngroups = theta_groups.size
            theta_reference[time_index, :ngroups] = theta_groups
            pi_reference[time_index, :ngroups] = group_pi
            mass_reference[time_index, :ngroups] = group_mass
            reference_surface_pressure[time_index] = top_pressure_value + total_mass[time_index] * self.constants.g / planetary_area

        coords = {
            "time": potential_temperature.coords["time"].values,
            REFERENCE_SAMPLE_DIM: np.arange(max_groups),
        }
        return ReferenceStateSolution(
            theta_reference=xr.DataArray(
                theta_reference,
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=coords,
                name="theta_reference",
                attrs={"units": "K"},
            ),
            pi_reference=xr.DataArray(
                pi_reference,
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=coords,
                name="pi_reference",
                attrs={"units": "Pa"},
            ),
            mass_reference=xr.DataArray(
                mass_reference,
                dims=("time", REFERENCE_SAMPLE_DIM),
                coords=coords,
                name="isentropic_mass",
                attrs={"units": "kg"},
            ),
            total_mass=xr.DataArray(
                total_mass,
                dims=("time",),
                coords={"time": potential_temperature.coords["time"].values},
                name="total_mass",
                attrs={"units": "kg"},
            ),
            reference_surface_pressure=reference_surface_pressure.rename("pi_surface_reference"),
            reference_top_pressure=reference_top_pressure.rename("pi_top_reference"),
            constants=self.constants,
        )


__all__ = [
    "REFERENCE_SAMPLE_DIM",
    "ReferenceStateSolution",
    "KoehlerReferenceState",
]
