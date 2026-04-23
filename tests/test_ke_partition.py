from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.reservoirs import kinetic_energy_eddy, kinetic_energy_zonal, total_horizontal_ke
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.io.mask_below_ground import make_theta

from .helpers import (
    full_field,
    make_coords,
    pressure_field,
    reference_case_level_bounds,
    surface_pressure,
    surface_pressure_policy_for_case,
)


@pytest.mark.parametrize("ps_value", [900.0, None])
def test_ke_partition_closes_with_and_without_topographic_truncation(ps_value):
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    if ps_value is None:
        ps_values = np.asarray(
            [
                [900.0, 650.0, 450.0, 250.0],
                [900.0, 650.0, 450.0, 250.0],
                [900.0, 650.0, 450.0, 250.0],
                [900.0, 650.0, 450.0, 250.0],
            ]
        )
    else:
        ps_values = ps_value

    ps = surface_pressure(time, latitude, longitude, ps_values)
    policy = surface_pressure_policy_for_case(ps, level)
    theta = make_theta(pressure, ps)
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(1.0, 4.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.5, 2.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="v",
        units="m s-1",
    )
    integrator = build_mass_integrator(level, latitude, longitude)

    total = total_horizontal_ke(u, v, theta, integrator, ps=ps, surface_pressure_policy=policy)
    kz = kinetic_energy_zonal(u, v, theta, integrator, ps=ps, surface_pressure_policy=policy)
    ke = kinetic_energy_eddy(u, v, theta, integrator, ps=ps, surface_pressure_policy=policy)

    np.testing.assert_allclose((kz + ke).values, total.values)


def test_ke_partition_closes_with_partial_cell_measure():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [700.0, 650.0, 500.0, 250.0],
                [700.0, 650.0, 500.0, 250.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(1.0, 4.0, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.5, 2.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="v",
        units="m s-1",
    )

    total = total_horizontal_ke(u, v, theta, integrator, measure=measure)
    kz = kinetic_energy_zonal(u, v, theta, integrator, measure=measure)
    ke = kinetic_energy_eddy(u, v, theta, integrator, measure=measure)
    manual_total = measure.integrate_full(0.5 * (u**2 + v**2))

    np.testing.assert_allclose((kz + ke).values, total.values)
    np.testing.assert_allclose(total.values, manual_total.values)
    assert total.attrs["normalization"] == "global_integral"
    assert total.attrs["base_quantity"] == "energy"


def test_ke_auto_measure_matches_explicit_measure():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [700.0, 650.0, 500.0, 250.0],
                [700.0, 650.0, 500.0, 250.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    u = full_field(time, level, latitude, longitude, 2.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    auto = total_horizontal_ke(u, v, theta, integrator, ps=ps)
    explicit = total_horizontal_ke(u, v, theta, integrator, measure=measure)

    np.testing.assert_allclose(auto.values, explicit.values)


def test_explicit_measure_with_matching_ps_matches_measure_only_path():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [700.0, 650.0, 500.0, 250.0],
                [700.0, 650.0, 500.0, 250.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    u = full_field(time, level, latitude, longitude, 2.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    with_ps = total_horizontal_ke(u, v, theta, integrator, measure=measure, ps=ps)
    measure_only = total_horizontal_ke(u, v, theta, integrator, measure=measure)

    np.testing.assert_allclose(with_ps.values, measure_only.values)


def test_explicit_measure_rejects_different_raw_surface_pressure_field():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_measure = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [700.0, 650.0, 500.0, 250.0],
                [700.0, 650.0, 500.0, 250.0],
            ]
        ),
    )
    ps_runtime = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [700.0, 640.0, 500.0, 250.0],
                [700.0, 640.0, 500.0, 250.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps_runtime)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps_measure, integrator)
    u = full_field(time, level, latitude, longitude, 2.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    with pytest.raises(ValueError, match="different surface pressure field.*max \\|Δps\\|"):
        total_horizontal_ke(u, v, theta, integrator, measure=measure, ps=ps_runtime)


def test_clip_policy_rejects_raw_ps_mismatch_even_when_effective_surface_pressure_matches():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_measure = surface_pressure(time, latitude, longitude, 1100.0)
    ps_runtime = surface_pressure(time, latitude, longitude, 1050.0)
    theta = make_theta(pressure, ps_runtime)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps_measure,
        integrator,
        surface_pressure_policy="clip",
    )
    u = full_field(time, level, latitude, longitude, 2.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    np.testing.assert_allclose(measure.effective_surface_pressure.values, 1000.0)
    with pytest.raises(ValueError, match="different surface pressure field.*max \\|Δps\\|"):
        total_horizontal_ke(
            u,
            v,
            theta,
            integrator,
            measure=measure,
            ps=ps_runtime,
            surface_pressure_policy="clip",
        )


def test_ke_outputs_expose_pressure_domain_metadata_for_raise_and_clip():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    integrator = build_mass_integrator(level, latitude, longitude)
    u = full_field(time, level, latitude, longitude, 2.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    ps_raise = surface_pressure(time, latitude, longitude, 900.0)
    theta_raise = make_theta(pressure, ps_raise)
    kz_raise = kinetic_energy_zonal(u, v, theta_raise, integrator, ps=ps_raise, surface_pressure_policy="raise")
    assert kz_raise.attrs["surface_pressure_policy"] == "raise"
    assert kz_raise.attrs["domain"] == "full_model_pressure_domain"
    assert kz_raise.attrs["not_exact_full_atmosphere"] is False

    ps_clip = surface_pressure(time, latitude, longitude, 1100.0)
    theta_clip = make_theta(pressure, ps_clip)
    kz_clip = kinetic_energy_zonal(u, v, theta_clip, integrator, ps=ps_clip, surface_pressure_policy="clip")
    assert kz_clip.attrs["surface_pressure_policy"] == "clip"
    assert kz_clip.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert kz_clip.attrs["not_exact_full_atmosphere"] is True


def test_ke_partition_requires_ps_or_explicit_measure():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    u = full_field(time, level, latitude, longitude, 1.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        total_horizontal_ke(u, v, theta, integrator)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        kinetic_energy_zonal(u, v, theta, integrator)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        kinetic_energy_eddy(u, v, theta, integrator)


def test_ke_auto_measure_reuses_integrator_level_bounds():
    time, level, latitude, longitude = make_coords(
        ntime=1,
        level_values=[820.0, 560.0, 240.0],
        nlat=2,
        nlon=4,
    )
    level_bounds = reference_case_level_bounds(level)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [650.0, 540.0, 380.0, 220.0],
                [650.0, 540.0, 380.0, 220.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude, level_bounds=level_bounds)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        level_bounds=level_bounds,
    )
    u = full_field(time, level, latitude, longitude, 2.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 1.0, name="v", units="m s-1")

    auto_total = total_horizontal_ke(u, v, theta, integrator, ps=ps)
    explicit_total = total_horizontal_ke(u, v, theta, integrator, measure=measure)
    auto_kz = kinetic_energy_zonal(u, v, theta, integrator, ps=ps)
    explicit_kz = kinetic_energy_zonal(u, v, theta, integrator, measure=measure)
    auto_ke = kinetic_energy_eddy(u, v, theta, integrator, ps=ps)
    explicit_ke = kinetic_energy_eddy(u, v, theta, integrator, measure=measure)

    np.testing.assert_allclose(auto_total.values, explicit_total.values)
    np.testing.assert_allclose(auto_kz.values, explicit_kz.values)
    np.testing.assert_allclose(auto_ke.values, explicit_ke.values)
