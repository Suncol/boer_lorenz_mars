"""Topography-aware finite-volume mass measures for exact Mars diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import xarray as xr

from .._validation import normalize_field, normalize_surface_field, normalize_zonal_field, require_dataarray
from ..constants_mars import MARS, MarsConstants
from .integrals import MassIntegrator, pressure_level_edges


def _coordinate_matches(reference: xr.DataArray, current: xr.DataArray, coord_name: str) -> bool:
    reference_values = reference.coords[coord_name].values
    current_values = current.coords[coord_name].values
    if coord_name == "time":
        return np.array_equal(reference_values, current_values)
    return np.allclose(reference_values, current_values)


def _coerce_surface_pressure(surface_pressure: xr.DataArray, integrator: MassIntegrator) -> xr.DataArray:
    surface_pressure = require_dataarray(surface_pressure, "surface_pressure")
    if set(surface_pressure.dims) != {"time", "latitude", "longitude"}:
        raise ValueError(
            "'surface_pressure' must contain exactly the dims ('time', 'latitude', 'longitude')."
        )

    surface_pressure = normalize_surface_field(surface_pressure, "surface_pressure")
    for coord_name in ("latitude", "longitude"):
        if not _coordinate_matches(integrator.cell_area, surface_pressure, coord_name):
            raise ValueError(
                f"Coordinate {coord_name!r} of 'surface_pressure' does not match the integrator grid."
            )
    return surface_pressure


def _validate_surface_pressure_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in {"raise", "clip"}:
        raise ValueError("'surface_pressure_policy' must be either 'raise' or 'clip'.")
    return normalized


def _domain_metadata_from_policy(surface_pressure_policy: str) -> dict[str, str | bool]:
    normalized_policy = _validate_surface_pressure_policy(surface_pressure_policy)
    if normalized_policy == "clip":
        return {
            "surface_pressure_policy": "clip",
            "domain": "truncated_to_model_pressure_domain",
            "not_exact_full_atmosphere": True,
        }
    return {
        "surface_pressure_policy": "raise",
        "domain": "full_model_pressure_domain",
        "not_exact_full_atmosphere": False,
    }


def _annotate_with_domain_metadata(
    field: xr.DataArray,
    *,
    surface_pressure_policy: str,
) -> xr.DataArray:
    annotated = field.copy(deep=False)
    annotated.attrs = dict(field.attrs)
    annotated.attrs.update(_domain_metadata_from_policy(surface_pressure_policy))
    return annotated


def _surface_pressure_comparison_atol(pressure_tolerance: float) -> float:
    return max(float(pressure_tolerance), 1.0e-12)


def _surface_pressure_max_abs_difference(reference: xr.DataArray, current: xr.DataArray) -> float:
    difference = np.abs(
        np.asarray(reference.values, dtype=float) - np.asarray(current.values, dtype=float)
    )
    return float(np.nanmax(difference)) if difference.size else 0.0


def _surface_pressure_values_match(
    reference: xr.DataArray,
    current: xr.DataArray,
    *,
    pressure_tolerance: float,
) -> tuple[bool, float]:
    max_abs_difference = _surface_pressure_max_abs_difference(reference, current)
    matches = np.allclose(
        np.asarray(reference.values, dtype=float),
        np.asarray(current.values, dtype=float),
        rtol=0.0,
        atol=_surface_pressure_comparison_atol(pressure_tolerance),
        equal_nan=True,
    )
    return bool(matches), max_abs_difference


def _expected_effective_surface_pressure(
    surface_pressure: xr.DataArray,
    integrator: MassIntegrator,
    *,
    level_bounds: xr.DataArray | None = None,
    pressure_tolerance: float = 1.0e-6,
    surface_pressure_policy: str = "raise",
) -> xr.DataArray:
    level_edges = pressure_level_edges(integrator.delta_p.coords["level"], bounds=level_bounds)
    deepest_level_edge = float(level_edges.isel(level_edge=0))
    normalized_policy = _validate_surface_pressure_policy(surface_pressure_policy)

    if normalized_policy == "raise":
        too_deep = np.asarray(surface_pressure.values, dtype=float) > deepest_level_edge + float(pressure_tolerance)
        if np.any(too_deep):
            raise ValueError(
                "Surface pressure extends below the deepest model pressure interface; "
                "use surface_pressure_policy='clip' to truncate to the model domain."
            )
        effective_surface_pressure = surface_pressure
    else:
        effective_surface_pressure = xr.apply_ufunc(
            np.minimum,
            surface_pressure,
            xr.full_like(surface_pressure, deepest_level_edge, dtype=float),
        )

    effective_surface_pressure = effective_surface_pressure.astype(float)
    effective_surface_pressure.name = "ps_effective"
    effective_surface_pressure.attrs.update(
        {
            "units": "Pa",
            "long_name": "effective surface pressure used by the exact finite-volume measure",
            "surface_pressure_policy": normalized_policy,
        }
    )
    return effective_surface_pressure


def _validate_measure_compatibility(
    measure: "TopographyAwareMeasure",
    integrator: MassIntegrator,
    *,
    theta: xr.DataArray | None = None,
    surface_pressure: xr.DataArray | None = None,
) -> None:
    if not _coordinate_matches(integrator.delta_p, measure.integrator.delta_p, "level"):
        raise ValueError("Explicit 'measure' does not match the integrator level coordinate.")
    for coord_name in ("latitude", "longitude"):
        if not _coordinate_matches(integrator.cell_area, measure.integrator.cell_area, coord_name):
            raise ValueError(f"Explicit 'measure' does not match the integrator {coord_name!r} coordinate.")

    if integrator.level_bounds is not None and measure.integrator.level_bounds is not None:
        if not np.allclose(
            np.asarray(integrator.level_bounds.values, dtype=float),
            np.asarray(measure.integrator.level_bounds.values, dtype=float),
        ):
            raise ValueError("Explicit 'measure' does not use the same level bounds as the integrator.")

    if theta is not None:
        theta = normalize_field(theta, "theta")
        for coord_name in ("time", "level", "latitude", "longitude"):
            if not _coordinate_matches(measure.cell_fraction, theta, coord_name):
                raise ValueError(f"Explicit 'measure' does not match the {coord_name!r} coordinate of 'theta'.")

    if surface_pressure is not None:
        surface_pressure = _coerce_surface_pressure(surface_pressure, integrator)
        for coord_name in ("time", "latitude", "longitude"):
            if not _coordinate_matches(measure.surface_pressure, surface_pressure, coord_name):
                raise ValueError(
                    f"Explicit 'measure' does not match the {coord_name!r} coordinate of 'surface_pressure'."
                )
        matches_surface_pressure, max_abs_difference = _surface_pressure_values_match(
            measure.surface_pressure,
            surface_pressure,
            pressure_tolerance=measure.pressure_tolerance,
        )
        if not matches_surface_pressure:
            raise ValueError(
                "Explicit 'measure' was built from a different surface pressure field than the supplied 'ps' "
                f"(max |Δps| = {max_abs_difference:.6g} Pa). Rebuild the measure from this 'ps' or omit 'measure'."
            )

        expected_effective_surface_pressure = _expected_effective_surface_pressure(
            surface_pressure,
            measure.integrator,
            level_bounds=measure.level_bounds,
            pressure_tolerance=measure.pressure_tolerance,
            surface_pressure_policy=measure.surface_pressure_policy,
        )
        matches_effective_surface_pressure, max_effective_difference = _surface_pressure_values_match(
            measure.effective_surface_pressure,
            expected_effective_surface_pressure,
            pressure_tolerance=measure.pressure_tolerance,
        )
        if not matches_effective_surface_pressure:
            raise ValueError(
                "Explicit 'measure' stores an effective surface pressure inconsistent with its raw "
                f"'surface_pressure' and policy (max |Δps_effective| = {max_effective_difference:.6g} Pa). "
                "Rebuild the measure from the supplied 'ps'."
            )


@dataclass(frozen=True)
class TopographyAwareMeasure:
    """Shared partial-cell finite-volume measure for Mars exact diagnostics."""

    integrator: MassIntegrator
    surface_pressure: xr.DataArray
    effective_surface_pressure: xr.DataArray
    level_bounds: xr.DataArray | None = None
    pressure_tolerance: float = 1.0e-6
    surface_pressure_policy: str = "raise"
    constants: MarsConstants = MARS

    def __post_init__(self) -> None:
        surface_pressure = _coerce_surface_pressure(self.surface_pressure, self.integrator)
        effective_surface_pressure = _coerce_surface_pressure(self.effective_surface_pressure, self.integrator)
        if surface_pressure.shape != effective_surface_pressure.shape:
            raise ValueError("'effective_surface_pressure' must share the same shape as 'surface_pressure'.")
        for coord_name in ("time", "latitude", "longitude"):
            if not _coordinate_matches(surface_pressure, effective_surface_pressure, coord_name):
                raise ValueError(
                    f"Coordinate {coord_name!r} of 'effective_surface_pressure' does not match 'surface_pressure'."
                )

        normalized_policy = _validate_surface_pressure_policy(self.surface_pressure_policy)

        if self.level_bounds is not None:
            level_bounds = xr.DataArray(self.level_bounds)
            if set(level_bounds.dims) != {"level", "bounds"}:
                raise ValueError("'level_bounds' must contain exactly the dims ('level', 'bounds').")
            level_bounds = level_bounds.transpose("level", "bounds")
            if not _coordinate_matches(self.integrator.delta_p, level_bounds, "level"):
                raise ValueError("Coordinate 'level' of 'level_bounds' does not match the integrator grid.")
        else:
            level_bounds = None

        if float(self.pressure_tolerance) < 0.0:
            raise ValueError("'pressure_tolerance' must be non-negative.")

        expected_effective_surface_pressure = _expected_effective_surface_pressure(
            surface_pressure,
            self.integrator,
            level_bounds=level_bounds,
            pressure_tolerance=float(self.pressure_tolerance),
            surface_pressure_policy=normalized_policy,
        )
        matches_effective_surface_pressure, max_abs_difference = _surface_pressure_values_match(
            effective_surface_pressure,
            expected_effective_surface_pressure,
            pressure_tolerance=float(self.pressure_tolerance),
        )
        if not matches_effective_surface_pressure:
            raise ValueError(
                "'effective_surface_pressure' is inconsistent with 'surface_pressure', "
                f"'surface_pressure_policy', and the integrator geometry (max |Δps_effective| = {max_abs_difference:.6g} Pa)."
            )

        object.__setattr__(
            self,
            "surface_pressure",
            _annotate_with_domain_metadata(surface_pressure, surface_pressure_policy=normalized_policy),
        )
        object.__setattr__(
            self,
            "effective_surface_pressure",
            _annotate_with_domain_metadata(effective_surface_pressure, surface_pressure_policy=normalized_policy),
        )
        object.__setattr__(self, "surface_pressure_policy", normalized_policy)
        if level_bounds is not None:
            object.__setattr__(self, "level_bounds", level_bounds)

    @property
    def domain_metadata(self) -> dict[str, str | bool]:
        return _domain_metadata_from_policy(self.surface_pressure_policy)

    def annotate_domain_metadata(self, field: xr.DataArray) -> xr.DataArray:
        return _annotate_with_domain_metadata(field, surface_pressure_policy=self.surface_pressure_policy)

    @classmethod
    def from_surface_pressure(
        cls,
        level: xr.DataArray,
        surface_pressure: xr.DataArray,
        integrator: MassIntegrator,
        *,
        level_bounds: xr.DataArray | None = None,
        pressure_tolerance: float = 1.0e-6,
        surface_pressure_policy: str = "raise",
    ) -> TopographyAwareMeasure:
        level = require_dataarray(level, "level")
        if not _coordinate_matches(integrator.delta_p, level, "level"):
            raise ValueError("Coordinate 'level' does not match the integrator grid.")

        resolved_level_bounds = integrator.level_bounds if level_bounds is None else xr.DataArray(level_bounds)
        if resolved_level_bounds is not None:
            if set(resolved_level_bounds.dims) != {"level", "bounds"}:
                raise ValueError("'level_bounds' must contain exactly the dims ('level', 'bounds').")
            resolved_level_bounds = resolved_level_bounds.transpose("level", "bounds")
            if not _coordinate_matches(integrator.delta_p, resolved_level_bounds, "level"):
                raise ValueError("Coordinate 'level' of 'level_bounds' does not match the integrator grid.")
            if integrator.level_bounds is not None and not np.allclose(
                np.asarray(integrator.level_bounds.values, dtype=float),
                np.asarray(resolved_level_bounds.values, dtype=float),
            ):
                raise ValueError("'level_bounds' must match the vertical geometry stored on the integrator.")

        surface_pressure = _coerce_surface_pressure(surface_pressure, integrator)
        normalized_policy = _validate_surface_pressure_policy(surface_pressure_policy)
        effective_surface_pressure = _expected_effective_surface_pressure(
            surface_pressure,
            integrator,
            level_bounds=resolved_level_bounds,
            pressure_tolerance=float(pressure_tolerance),
            surface_pressure_policy=normalized_policy,
        )

        return cls(
            integrator=integrator,
            surface_pressure=surface_pressure,
            effective_surface_pressure=effective_surface_pressure,
            level_bounds=resolved_level_bounds,
            pressure_tolerance=float(pressure_tolerance),
            surface_pressure_policy=normalized_policy,
            constants=integrator.constants,
        )

    @cached_property
    def level_edges(self) -> xr.DataArray:
        return pressure_level_edges(self.integrator.delta_p.coords["level"], bounds=self.level_bounds)

    @cached_property
    def lower_edge(self) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(self.level_edges.values[:-1], dtype=float),
            dims=("level",),
            coords={"level": self.integrator.delta_p.coords["level"].values},
            name="level_lower_edge",
            attrs={"units": "Pa"},
        )

    @cached_property
    def upper_edge(self) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(self.level_edges.values[1:], dtype=float),
            dims=("level",),
            coords={"level": self.integrator.delta_p.coords["level"].values},
            name="level_upper_edge",
            attrs={"units": "Pa"},
        )

    @cached_property
    def above_ground_dp(self) -> xr.DataArray:
        ps_4d = self.effective_surface_pressure.expand_dims(level=self.integrator.delta_p.coords["level"]).transpose(
            "time",
            "level",
            "latitude",
            "longitude",
        )
        lower_4d = self.lower_edge.broadcast_like(ps_4d)
        upper_4d = self.upper_edge.broadcast_like(ps_4d)
        clipped_bottom = xr.apply_ufunc(np.minimum, ps_4d, lower_4d)
        above_ground_dp = xr.where(ps_4d > upper_4d, clipped_bottom - upper_4d, 0.0)
        above_ground_dp = above_ground_dp.clip(min=0.0).astype(float)
        above_ground_dp.name = "above_ground_dp"
        above_ground_dp.attrs.update(
            {
                "units": "Pa",
                "long_name": "topography-aware above-ground pressure thickness",
            }
        )
        return self.annotate_domain_metadata(above_ground_dp)

    @cached_property
    def cell_fraction(self) -> xr.DataArray:
        delta_p = self.integrator.delta_p.broadcast_like(self.above_ground_dp)
        fraction = xr.where(delta_p > 0.0, self.above_ground_dp / delta_p, 0.0).clip(min=0.0, max=1.0)
        fraction.name = "cell_fraction"
        fraction.attrs.update(
            {
                "units": "1",
                "long_name": "fraction of each pressure layer that remains above ground",
            }
        )
        return self.annotate_domain_metadata(fraction)

    @cached_property
    def parcel_mass(self) -> xr.DataArray:
        weights = self.integrator.full_mass_weights.broadcast_like(self.cell_fraction)
        parcel_mass = self.cell_fraction * weights
        parcel_mass.name = "parcel_mass"
        parcel_mass.attrs.update(
            {
                "units": "kg",
                "long_name": "topography-aware parcel mass of each pressure cell",
            }
        )
        return self.annotate_domain_metadata(parcel_mass)

    @cached_property
    def zonal_mass(self) -> xr.DataArray:
        zonal_mass = self.parcel_mass.sum(dim="longitude")
        zonal_mass.name = "zonal_mass"
        zonal_mass.attrs.update(
            {
                "units": "kg",
                "long_name": "topography-aware zonal atmospheric mass on each level-latitude band",
            }
        )
        return self.annotate_domain_metadata(zonal_mass)

    @cached_property
    def zonal_fraction(self) -> xr.DataArray:
        zonal_weights = self.integrator.zonal_mass_weights.broadcast_like(self.zonal_mass)
        zonal_fraction = xr.where(zonal_weights > 0.0, self.zonal_mass / zonal_weights, 0.0).clip(min=0.0, max=1.0)
        zonal_fraction.name = "zonal_fraction"
        zonal_fraction.attrs.update(
            {
                "units": "1",
                "long_name": "zonal finite-volume coverage fraction of each level-latitude band",
            }
        )
        return self.annotate_domain_metadata(zonal_fraction)

    def integrate_full(self, field: xr.DataArray) -> xr.DataArray:
        field = normalize_field(field, "field")
        self.integrator._ensure_full_grid_matches(field)
        return (field * self.parcel_mass).sum(dim=("level", "latitude", "longitude"))

    def integrate_zonal(self, field: xr.DataArray) -> xr.DataArray:
        field = normalize_zonal_field(field, "field")
        self.integrator._ensure_zonal_grid_matches(field)
        return (field * self.zonal_mass).sum(dim=("level", "latitude"))

def resolve_exact_measure(
    integrator: MassIntegrator,
    *,
    measure: TopographyAwareMeasure | None = None,
    ps: xr.DataArray | None = None,
    theta: xr.DataArray | None = None,
    surface_pressure_policy: str = "raise",
    pressure_tolerance: float = 1.0e-6,
    diagnostic_name: str = "Exact Boer diagnostics",
) -> TopographyAwareMeasure:
    """Resolve the finite-volume measure for an exact diagnostic.

    When an explicit ``measure`` is supplied it remains authoritative; the optional
    ``ps`` argument is used only to validate that the measure was built from the same
    raw surface-pressure field.
    """

    if measure is not None:
        _validate_measure_compatibility(
            measure,
            integrator,
            theta=theta,
            surface_pressure=ps,
        )
        return measure

    if ps is None:
        raise ValueError(
            f"{diagnostic_name} default to measure-aware finite-volume weights; "
            "provide 'ps' or an explicit 'measure'."
        )

    level = theta.coords["level"] if theta is not None else integrator.delta_p.coords["level"]
    return TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=integrator.level_bounds,
        pressure_tolerance=pressure_tolerance,
        surface_pressure_policy=surface_pressure_policy,
    )


__all__ = ["TopographyAwareMeasure", "resolve_exact_measure"]
