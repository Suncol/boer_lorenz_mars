"""Shared reference-state solution object and curve-evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .._validation import normalize_field, normalize_zonal_field, require_dataarray
from ..constants_mars import MARS, MarsConstants


REFERENCE_SAMPLE_DIM = "reference_sample"
REFERENCE_INTERFACE_DIM = "reference_interface"


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
    *,
    interpolation_space: str = "pressure",
    constants: MarsConstants = MARS,
) -> np.ndarray:
    valid = np.isfinite(theta_reference) & np.isfinite(pi_reference)
    count = np.count_nonzero(valid)
    if count == 0:
        return np.full(theta_target.shape, np.nan, dtype=float)
    if count == 1:
        raise ValueError("Single-sample reference slices require an explicit pressure field.")

    theta_curve = theta_reference[valid]
    pi_curve = pi_reference[valid]
    order = np.argsort(theta_curve, kind="mergesort")
    theta_curve = theta_curve[order]
    pi_curve = pi_curve[order]

    interpolation_space = str(interpolation_space).strip().lower()
    if interpolation_space == "pressure":
        return np.interp(
            theta_target,
            theta_curve,
            pi_curve,
            left=lower_fill,
            right=upper_fill,
        )
    if interpolation_space != "exner":
        raise ValueError("'interpolation_space' must be either 'pressure' or 'exner'.")

    exner_curve = np.power(pi_curve / constants.p00, constants.kappa)
    lower_fill_exner = float(lower_fill / constants.p00) ** constants.kappa
    upper_fill_exner = float(upper_fill / constants.p00) ** constants.kappa
    exner_target = np.interp(
        theta_target,
        theta_curve,
        exner_curve,
        left=lower_fill_exner,
        right=upper_fill_exner,
    )
    return constants.p00 * np.power(np.maximum(exner_target, 0.0), 1.0 / constants.kappa)


@dataclass(frozen=True)
class ReferenceStateSolution:
    """Reference-state diagnostics returned by reference-state solvers."""

    theta_reference: xr.DataArray
    pi_reference: xr.DataArray
    mass_reference: xr.DataArray
    reference_interface_pressure: xr.DataArray
    reference_interface_geopotential: xr.DataArray
    total_mass: xr.DataArray
    reference_surface_pressure: xr.DataArray
    reference_bottom_pressure: xr.DataArray
    reference_top_pressure: xr.DataArray
    ps_effective: xr.DataArray | None = None
    pi_s: xr.DataArray | None = None
    pi_sZ: xr.DataArray | None = None
    iterations: xr.DataArray | None = None
    converged: xr.DataArray | None = None
    iterations_zonal: xr.DataArray | None = None
    converged_zonal: xr.DataArray | None = None
    monotonic_violations: xr.DataArray | None = None
    monotonic_repairs: xr.DataArray | None = None
    monotonic_violations_zonal: xr.DataArray | None = None
    monotonic_repairs_zonal: xr.DataArray | None = None
    method: str | None = None
    constants: MarsConstants = MARS
    _theta_reference_zonal: xr.DataArray | None = field(default=None, repr=False)
    _pi_reference_zonal: xr.DataArray | None = field(default=None, repr=False)
    _reference_bottom_pressure_zonal: xr.DataArray | None = field(default=None, repr=False)

    @staticmethod
    def _convergence_truth_values(status: xr.DataArray) -> np.ndarray:
        values = np.asarray(status.values)
        with np.errstate(invalid="ignore"):
            truth = np.equal(values, True)
        return np.asarray(truth, dtype=bool)

    def _ensure_converged(self, *, zonal: bool, api_name: str) -> None:
        field_name = "converged_zonal" if zonal else "converged"
        status = getattr(self, field_name)
        if status is None:
            raise ValueError(
                f"ReferenceStateSolution.{api_name}() cannot verify reference-state convergence "
                f"because {field_name!r} is None."
            )

        truth = self._convergence_truth_values(status)
        if truth.size == 0:
            raise ValueError(
                f"ReferenceStateSolution.{api_name}() cannot verify reference-state convergence "
                f"because {field_name!r} is empty."
            )
        if not bool(np.all(truth)):
            failed = int(truth.size - np.count_nonzero(truth))
            raise ValueError(
                f"ReferenceStateSolution.{api_name}() cannot evaluate a non-converged reference state; "
                f"{field_name!r} contains {failed} false or missing value(s)."
            )

    def _evaluate_curve(
        self,
        theta_target: xr.DataArray,
        theta_reference: xr.DataArray,
        pi_reference: xr.DataArray,
        lower_fill: xr.DataArray,
        *,
        pressure: xr.DataArray | None = None,
        name: str,
    ) -> xr.DataArray:
        if pressure is not None:
            pressure = _normalize_reference_target(pressure, "pressure")
            if pressure.dims != theta_target.dims:
                raise ValueError("'pressure' must share the same dims as the reference-pressure target field.")
            for coord_name in theta_target.dims:
                reference = theta_target.coords[coord_name].values
                current = pressure.coords[coord_name].values
                equal = np.array_equal(reference, current) if coord_name == "time" else np.allclose(reference, current)
                if not equal:
                    raise ValueError(
                        f"Coordinate {coord_name!r} of 'pressure' does not match the reference-pressure target field."
                    )

        interpolation_space = str(
            pi_reference.attrs.get(
                "reference_curve_interpolation_space",
                theta_reference.attrs.get("reference_curve_interpolation_space", "pressure"),
            )
        ).strip().lower()

        values = np.empty(theta_target.shape, dtype=float)
        for time_index in range(theta_target.sizes["time"]):
            theta_values = np.asarray(theta_target.isel(time=time_index).values, dtype=float)
            theta_reference_values = np.asarray(theta_reference.isel(time=time_index).values, dtype=float)
            pi_reference_values = np.asarray(pi_reference.isel(time=time_index).values, dtype=float)
            valid = np.isfinite(theta_reference_values) & np.isfinite(pi_reference_values)
            count = np.count_nonzero(valid)
            if count == 1:
                if pressure is None:
                    raise ValueError(
                        "Single-sample reference slices require an explicit pressure field when evaluating "
                        "reference_pressure() or zonal_reference_pressure()."
                    )
                values[time_index] = np.asarray(pressure.isel(time=time_index).values, dtype=float)
                continue
            values[time_index] = _interp_reference_curve(
                theta_target=theta_values.reshape(-1),
                theta_reference=theta_reference_values,
                pi_reference=pi_reference_values,
                lower_fill=float(lower_fill.isel(time=time_index)),
                upper_fill=float(self.reference_top_pressure.isel(time=time_index)),
                interpolation_space=interpolation_space,
                constants=self.constants,
            ).reshape(theta_values.shape)
        return xr.DataArray(
            values,
            dims=theta_target.dims,
            coords=theta_target.coords,
            name=name,
            attrs={"units": "Pa"},
        )

    def reference_pressure(
        self,
        potential_temperature: xr.DataArray,
        *,
        pressure: xr.DataArray | None = None,
        name: str = "pi",
    ) -> xr.DataArray:
        """Evaluate the full reference pressure curve."""

        self._ensure_converged(zonal=False, api_name="reference_pressure")
        potential_temperature = _normalize_reference_target(
            potential_temperature,
            "potential_temperature",
        )
        return self._evaluate_curve(
            potential_temperature,
            self.theta_reference,
            self.pi_reference,
            self.reference_bottom_pressure,
            pressure=pressure,
            name=name,
        )

    def zonal_reference_pressure(
        self,
        representative_theta: xr.DataArray,
        *,
        pressure: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Evaluate the zonal reference pressure curve on a zonal field."""

        self._ensure_converged(zonal=True, api_name="zonal_reference_pressure")
        representative_theta = normalize_zonal_field(
            representative_theta,
            "representative_theta",
        )
        theta_reference = self._theta_reference_zonal if self._theta_reference_zonal is not None else self.theta_reference
        pi_reference = self._pi_reference_zonal if self._pi_reference_zonal is not None else self.pi_reference
        lower_fill = (
            self._reference_bottom_pressure_zonal
            if self._reference_bottom_pressure_zonal is not None
            else self.reference_bottom_pressure
        )
        return self._evaluate_curve(
            representative_theta,
            theta_reference,
            pi_reference,
            lower_fill,
            pressure=pressure,
            name="pi_Z",
        )

    def efficiency(
        self,
        potential_temperature: xr.DataArray,
        pressure: xr.DataArray,
        *,
        name: str = "N",
    ) -> xr.DataArray:
        """Return the full efficiency factor ``N = 1 - (pi / p)^kappa``."""

        pressure = _normalize_reference_target(pressure, "pressure")
        pi = self.reference_pressure(potential_temperature, pressure=pressure)
        efficiency = 1.0 - (pi / pressure) ** self.constants.kappa
        efficiency.name = name
        efficiency.attrs["units"] = "1"
        return efficiency

    def zonal_efficiency(
        self,
        representative_theta: xr.DataArray,
        pressure: xr.DataArray,
    ) -> xr.DataArray:
        """Return the zonal efficiency factor ``N_Z = 1 - (pi_Z / p)^kappa``."""

        pressure = normalize_zonal_field(pressure, "pressure")
        pi_z = self.zonal_reference_pressure(representative_theta, pressure=pressure)
        n_z = 1.0 - (pi_z / pressure) ** self.constants.kappa
        n_z.name = "N_Z"
        n_z.attrs["units"] = "1"
        return n_z


__all__ = [
    "REFERENCE_SAMPLE_DIM",
    "REFERENCE_INTERFACE_DIM",
    "ReferenceStateSolution",
]
