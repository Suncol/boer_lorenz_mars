from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.conversions import (
    conversion_eddy_ape_to_ke,
    conversion_zonal_ape_to_eddy_ape,
    conversion_zonal_ape_to_ke_part1,
    conversion_zonal_ke_to_eddy_ke_part1,
)
from mars_exact_lec.boer.reservoirs import kinetic_energy_eddy
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
    temperature_from_theta_values,
    zonal_field,
)


def _conversion_theta_mask_api_case():
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
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    values_shape = (time.size, level.size, latitude.size, longitude.size)
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.5, 0.5, np.prod(values_shape)).reshape(values_shape),
        name="omega",
        units="Pa s-1",
    )
    alpha = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.5, 1.5, np.prod(values_shape)).reshape(values_shape),
        name="alpha",
        units="m3 kg-1",
    )
    return omega, alpha, theta_mask, integrator, ps, measure


def _mask_filled_like(theta_mask, value):
    return theta_mask.copy(data=np.full(theta_mask.shape, value, dtype=float))


def test_eddy_free_fields_give_zero_ke_and_ce():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size).reshape(time.size, level.size, latitude.size)[..., None],
        name="u",
    )
    v = full_field(time, level, latitude, longitude, 2.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha")

    np.testing.assert_allclose(kinetic_energy_eddy(u, v, theta, integrator, ps=ps).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, ps=ps).values,
        0.0,
        atol=1e-12,
    )


def test_conversion_partition_closes_total_theta_omega_alpha_term():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [900.0, 650.0, 450.0, 250.0],
            [900.0, 650.0, 450.0, 250.0],
            [900.0, 650.0, 450.0, 250.0],
            [900.0, 650.0, 450.0, 250.0],
        ]
    )
    ps = surface_pressure(time, latitude, longitude, ps_values)
    policy = surface_pressure_policy_for_case(ps, level)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.5, 0.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="omega",
        units="Pa s-1",
    )
    alpha = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.5, 1.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="alpha",
        units="m3 kg-1",
    )

    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator, surface_pressure_policy=policy)
    ce = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, ps=ps, surface_pressure_policy=policy)
    cz1 = conversion_zonal_ape_to_ke_part1(omega, alpha, theta, integrator, ps=ps, surface_pressure_policy=policy)
    total = -measure.integrate_full(omega * alpha)

    np.testing.assert_allclose((ce + cz1).values, total.values)


def test_conversion_partition_closes_with_partial_cell_measure():
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

    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.5, 0.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="omega",
        units="Pa s-1",
    )
    alpha = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(0.5, 1.5, time.size * level.size * latitude.size * longitude.size).reshape(
            time.size, level.size, latitude.size, longitude.size
        ),
        name="alpha",
        units="m3 kg-1",
    )

    ce = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, measure=measure)
    cz1 = conversion_zonal_ape_to_ke_part1(omega, alpha, theta, integrator, measure=measure)
    total = -measure.integrate_full(omega * alpha)

    np.testing.assert_allclose((ce + cz1).values, total.values)
    assert ce.attrs["normalization"] == "global_integral"
    assert ce.attrs["base_quantity"] == "power"


def test_conversion_auto_measure_matches_explicit_measure():
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
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega", units="Pa s-1")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha", units="m3 kg-1")

    auto = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, ps=ps)
    explicit = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, measure=measure)

    np.testing.assert_allclose(auto.values, explicit.values)


def test_conversion_theta_mask_api_accepts_positional_keyword_and_deprecated_theta():
    omega, alpha, theta_mask, integrator, ps, _ = _conversion_theta_mask_api_case()

    positional = conversion_eddy_ape_to_ke(omega, alpha, theta_mask, integrator, ps=ps)
    keyword = conversion_eddy_ape_to_ke(
        omega,
        alpha,
        theta_mask=theta_mask,
        integrator=integrator,
        ps=ps,
    )
    with pytest.warns(FutureWarning, match="'theta' is deprecated"):
        deprecated = conversion_eddy_ape_to_ke(omega, alpha, theta=theta_mask, integrator=integrator, ps=ps)

    np.testing.assert_allclose(keyword.values, positional.values)
    np.testing.assert_allclose(deprecated.values, positional.values)
    with pytest.warns(FutureWarning, match="'theta' is deprecated"):
        with pytest.raises(ValueError, match="only one of 'theta_mask' or deprecated 'theta'"):
            conversion_eddy_ape_to_ke(
                omega,
                alpha,
                theta_mask=theta_mask,
                theta=theta_mask,
                integrator=integrator,
                ps=ps,
            )


@pytest.mark.parametrize(
    ("mask_value", "message"),
    [
        (180.0, r"\[0, 1\]"),
        (np.nan, "finite mask values"),
        (-0.1, r"\[0, 1\]"),
        (1.1, r"\[0, 1\]"),
    ],
)
def test_conversion_theta_mask_validation_rejects_invalid_values_even_with_measure(mask_value, message):
    omega, alpha, theta_mask, integrator, ps, measure = _conversion_theta_mask_api_case()
    bad_mask = _mask_filled_like(theta_mask, mask_value)

    with pytest.raises(ValueError, match=message):
        conversion_eddy_ape_to_ke(omega, alpha, theta_mask=bad_mask, integrator=integrator, ps=ps)
    with pytest.raises(ValueError, match=message):
        conversion_eddy_ape_to_ke(
            omega,
            alpha,
            theta_mask=bad_mask,
            integrator=integrator,
            measure=measure,
        )


def test_conversion_body_terms_require_ps_or_explicit_measure():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega", units="Pa s-1")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha", units="m3 kg-1")
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    n_z = zonal_field(time, level, latitude, 1.0, name="N_Z")

    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        conversion_zonal_ape_to_ke_part1(omega, alpha, theta, integrator)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        conversion_eddy_ape_to_ke(omega, alpha, theta, integrator)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z, theta, integrator)
    with pytest.raises(ValueError, match="provide 'ps' or an explicit 'measure'"):
        conversion_zonal_ke_to_eddy_ke_part1(u, v, omega, theta, integrator)


def test_conversion_auto_measure_reuses_integrator_level_bounds():
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
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega", units="Pa s-1")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha", units="m3 kg-1")

    auto = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, ps=ps)
    explicit = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, measure=measure)

    np.testing.assert_allclose(auto.values, explicit.values)


def test_conversion_outputs_expose_clip_domain_metadata():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[800.0, 400.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 1100.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega", units="Pa s-1")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha", units="m3 kg-1")

    ce = conversion_eddy_ape_to_ke(omega, alpha, theta, integrator, ps=ps, surface_pressure_policy="clip")

    assert ce.attrs["surface_pressure_policy"] == "clip"
    assert ce.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert ce.attrs["not_exact_full_atmosphere"] is True


def test_flat_surface_theta_and_zonal_mean_reduce_cleanly():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    theta = make_theta(pressure, surface_pressure(time, latitude, longitude, 800.0))
    np.testing.assert_allclose(theta.values, 1.0)


def test_conversion_zonal_ape_to_eddy_ape_rejects_longitude_varying_full_nz():
    time, level, latitude, longitude = make_coords()
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    n_z = zonal_field(time, level, latitude, 1.0, name="N_Z")
    n_z_broadcast = n_z.expand_dims(longitude=longitude).transpose("time", "level", "latitude", "longitude")
    n_z_bad = full_field(
        time,
        level,
        latitude,
        longitude,
        np.arange(time.size * level.size * latitude.size * longitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
            longitude.size,
        ),
        name="N_Z_bad",
    )

    zonal_result = conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z, theta, integrator, ps=ps)
    broadcast_result = conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z_broadcast, theta, integrator, ps=ps)

    np.testing.assert_allclose(zonal_result.values, broadcast_result.values)
    with pytest.raises(ValueError, match="longitude-varying"):
        conversion_zonal_ape_to_eddy_ape(temperature, u, v, omega, n_z_bad, theta, integrator, ps=ps)
