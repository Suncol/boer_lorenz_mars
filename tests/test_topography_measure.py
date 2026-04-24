from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure

from .helpers import make_coords, pressure_field, surface_pressure


def _build_partial_measure_case():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [700.0, 500.0, 650.0, 250.0],
                [700.0, 500.0, 650.0, 250.0],
            ]
        ),
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    return pressure, ps, integrator, measure


def test_topography_measure_above_ground_dp_cell_fraction_and_parcel_mass_match_partial_cell_geometry():
    pressure, _, integrator, measure = _build_partial_measure_case()

    expected_dp = np.asarray(
        [
            [
                [[100.0, 0.0, 50.0, 0.0], [100.0, 0.0, 50.0, 0.0]],
                [[400.0, 300.0, 400.0, 50.0], [400.0, 300.0, 400.0, 50.0]],
            ]
        ]
    )
    expected_fraction = expected_dp / np.asarray([[[[400.0]], [[400.0]]]])

    np.testing.assert_allclose(measure.above_ground_dp.values, expected_dp)
    np.testing.assert_allclose(
        measure.effective_bottom_pressure.values,
        measure.upper_edge.broadcast_like(measure.above_ground_dp).values + expected_dp,
    )
    assert measure.effective_bottom_pressure.dims == ("time", "level", "latitude", "longitude")
    assert measure.effective_bottom_pressure.attrs["units"] == "Pa"
    assert measure.effective_bottom_pressure.attrs["domain"] == "full_model_pressure_domain"
    np.testing.assert_allclose(measure.cell_fraction.values, expected_fraction)
    np.testing.assert_allclose(
        measure.parcel_mass.values,
        expected_fraction * integrator.full_mass_weights.broadcast_like(pressure).values,
    )
    np.testing.assert_allclose(
        measure.zonal_mass.values,
        measure.parcel_mass.sum(dim="longitude").values,
    )
    np.testing.assert_allclose(
        measure.zonal_fraction.values,
        (
            measure.zonal_mass / integrator.zonal_mass_weights.broadcast_like(measure.zonal_mass)
        ).fillna(0.0).values,
    )


def test_topography_measure_integrators_reduce_to_parcel_mass_and_zonal_mass_sums_for_unit_fields():
    pressure, _, _, measure = _build_partial_measure_case()

    full_ones = pressure * 0.0 + 1.0
    zonal_ones = measure.zonal_mass * 0.0 + 1.0

    np.testing.assert_allclose(
        measure.integrate_full(full_ones).values,
        measure.parcel_mass.sum(dim=("level", "latitude", "longitude")).values,
    )
    np.testing.assert_allclose(
        measure.integrate_zonal(zonal_ones).values,
        measure.zonal_mass.sum(dim=("level", "latitude")).values,
    )


def test_topography_measure_surface_pressure_policy_raise_and_clip():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 1100.0)

    with pytest.raises(ValueError, match="Surface pressure extends below the deepest model pressure interface"):
        TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)

    with pytest.warns(UserWarning, match="truncated pressure domain"):
        clipped = TopographyAwareMeasure.from_surface_pressure(
            level,
            ps,
            integrator,
            surface_pressure_policy="clip",
        )

    np.testing.assert_allclose(clipped.effective_surface_pressure.values, 1000.0)
    assert clipped.surface_pressure_policy == "clip"
    assert clipped.surface_pressure.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert clipped.effective_surface_pressure.attrs["not_exact_full_atmosphere"] is True
    assert clipped.above_ground_dp.attrs["surface_pressure_policy"] == "clip"
    assert clipped.effective_bottom_pressure.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert clipped.cell_fraction.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert clipped.parcel_mass.attrs["not_exact_full_atmosphere"] is True
    assert clipped.zonal_mass.attrs["surface_pressure_policy"] == "clip"

    raised = TopographyAwareMeasure.from_surface_pressure(
        level,
        surface_pressure(time, latitude, longitude, 900.0),
        integrator,
        surface_pressure_policy="raise",
    )
    assert raised.effective_surface_pressure.attrs["domain"] == "full_model_pressure_domain"
    assert raised.effective_surface_pressure.attrs["not_exact_full_atmosphere"] is False


def test_clip_policy_does_not_warn_when_surface_pressure_is_inside_model_domain():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measure = TopographyAwareMeasure.from_surface_pressure(
            level,
            ps,
            integrator,
            surface_pressure_policy="clip",
        )

    assert not [warning for warning in caught if warning.category is UserWarning]
    assert measure.effective_surface_pressure.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert measure.effective_surface_pressure.attrs["not_exact_full_atmosphere"] is True


def test_topography_measure_rejects_effective_surface_pressure_inconsistent_with_policy():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 1100.0)
    inconsistent_effective = surface_pressure(time, latitude, longitude, 950.0)

    with pytest.raises(ValueError, match="max \\|Δps_effective\\|"):
        TopographyAwareMeasure(
            integrator=integrator,
            surface_pressure=ps,
            effective_surface_pressure=inconsistent_effective,
            surface_pressure_policy="clip",
        )
