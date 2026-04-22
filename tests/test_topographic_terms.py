from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.conversions import C_K, C_K1, C_K2, C_Z, C_Z1, C_Z2
from mars_exact_lec.boer.reservoirs import A, A1, A2, A_E, A_E1, A_E2, A_Z, A_Z1, A_Z2
from mars_exact_lec.common.geopotential import reconstruct_hydrostatic_geopotential
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.time_derivatives import time_derivative
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import KoehlerReferenceState, potential_temperature

from .helpers import (
    full_field,
    make_coords,
    pressure_field,
    reference_case_coords,
    reference_case_surface_geopotential_values,
    reference_case_surface_pressure_values,
    reference_case_theta_profile,
    surface_geopotential,
    surface_pressure,
    surface_zonal_field,
    temperature_from_theta_values,
)


def test_time_derivative_supports_datetime_coordinates():
    time, _, latitude, longitude = make_coords(ntime=3, time_dtype="datetime")
    field = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray([0.0, 3600.0, 7200.0])[:, None, None],
        name="linear_field",
    )

    derivative = time_derivative(field)

    np.testing.assert_allclose(derivative.values, 1.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("hours", 1.0 / 3600.0),
        ("sols", 1.0 / 88_775.244),
    ],
)
def test_time_derivative_scales_numeric_time_units_to_per_second(units, expected):
    time, _, latitude, longitude = make_coords(ntime=3)
    time = xr.DataArray(time.values, dims=("time",), coords={"time": time.values}, attrs={"units": units})
    field = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray([0.0, 1.0, 2.0])[:, None, None],
        name="linear_field",
    )

    derivative = time_derivative(field)

    np.testing.assert_allclose(derivative.values, expected, atol=1.0e-12)


def test_time_derivative_rejects_unitless_numeric_time():
    time, _, latitude, longitude = make_coords(ntime=3)
    time = xr.DataArray(time.values, dims=("time",), coords={"time": time.values})
    field = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray([0.0, 1.0, 2.0])[:, None, None],
        name="linear_field",
    )

    with pytest.raises(ValueError, match="must declare units"):
        time_derivative(field)


def test_cz2_vanishes_when_surface_pressure_is_time_invariant():
    time, level, latitude, longitude = make_coords(ntime=3)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [0.0, 100.0, 200.0, 300.0],
                [50.0, 150.0, 250.0, 350.0],
                [100.0, 200.0, 300.0, 400.0],
                [150.0, 250.0, 350.0, 450.0],
            ]
        ),
    )

    np.testing.assert_allclose(C_Z2(ps, phis, integrator).values, 0.0, atol=1.0e-12)


def test_ck2_vanishes_for_zonally_symmetric_geopotential():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(100.0, 300.0, time.size * level.size * latitude.size).reshape(
            time.size, level.size, latitude.size
        )[..., None],
        name="geopotential",
        units="m2 s-2",
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    np.testing.assert_allclose(
        C_K2(u, v, omega, theta, integrator, geopotential=geopotential).values,
        0.0,
        atol=1.0e-12,
    )


def test_geopotential_reconstruction_matches_explicit_field_in_ck2():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [0.0, 80.0, 160.0, 240.0],
                [40.0, 120.0, 200.0, 280.0],
                [80.0, 160.0, 240.0, 320.0],
                [120.0, 200.0, 280.0, 360.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature_values = np.linspace(
        180.0,
        240.0,
        time.size * level.size * latitude.size * longitude.size,
    ).reshape(time.size, level.size, latitude.size, longitude.size)
    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        temperature_values,
        name="temperature",
        units="K",
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    geopotential = reconstruct_hydrostatic_geopotential(
        temperature,
        pressure,
        phis,
        ps=ps,
        theta=theta,
    )

    explicit = C_K2(u, v, omega, theta, integrator, geopotential=geopotential)
    reconstructed = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
    )

    np.testing.assert_allclose(reconstructed.values, explicit.values)


def test_ck2_remains_finite_when_surface_crosses_a_level_in_time():
    time, level, latitude, longitude = make_coords(ntime=3, time_dtype="datetime")
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.full((time.size, latitude.size, longitude.size), 900.0)
    ps_values[:, :, 0] = np.asarray([450.0, 650.0, 450.0])[:, None]
    ps = surface_pressure(time, latitude, longitude, ps_values)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [0.0, 150.0, 300.0, 450.0],
                [50.0, 200.0, 350.0, 500.0],
                [100.0, 250.0, 400.0, 550.0],
                [150.0, 300.0, 450.0, 600.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            180.0,
            235.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="temperature",
        units="K",
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    reconstructed_phi = reconstruct_hydrostatic_geopotential(
        temperature,
        pressure,
        phis,
        ps=ps,
        theta=theta,
    )
    masked_explicit_phi = reconstructed_phi.where(theta > 0.0)

    reconstructed = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
    )
    explicit = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=masked_explicit_phi,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
    )

    assert np.isfinite(reconstructed.values).all()
    assert np.isfinite(explicit.values).all()
    np.testing.assert_allclose(explicit.values, reconstructed.values)


def test_ck2_ignores_below_ground_explicit_geopotential_fill():
    time, level, latitude, longitude = make_coords(ntime=3, time_dtype="datetime")
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.full((time.size, latitude.size, longitude.size), 900.0)
    ps_values[:, :, 0] = np.asarray([450.0, 650.0, 450.0])[:, None]
    ps_values[:, :, 1] = np.asarray([500.0, 500.0, 700.0])[:, None]
    ps = surface_pressure(time, latitude, longitude, ps_values)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [0.0, 150.0, 300.0, 450.0],
                [50.0, 200.0, 350.0, 500.0],
                [100.0, 250.0, 400.0, 550.0],
                [150.0, 300.0, 450.0, 600.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            180.0,
            235.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="temperature",
        units="K",
    )
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            5.0,
            20.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            -4.0,
            3.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            -0.2,
            0.3,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="omega",
        units="Pa s-1",
    )

    reconstructed_phi = reconstruct_hydrostatic_geopotential(
        temperature,
        pressure,
        phis,
        ps=ps,
        theta=theta,
    )
    explicit_low = reconstructed_phi.where(theta > 0.0, -1.0e3)
    explicit_high = reconstructed_phi.where(theta > 0.0, 1.0e6)

    low_fill = C_K2(u, v, omega, theta, integrator, geopotential=explicit_low)
    high_fill = C_K2(u, v, omega, theta, integrator, geopotential=explicit_high)

    assert np.isfinite(low_fill.values).all()
    assert np.isfinite(high_fill.values).all()
    np.testing.assert_allclose(low_fill.values, high_fill.values)


def test_ck2_hydrostatic_reconstruction_ignores_below_ground_temperature_fill():
    time, level, latitude, longitude = make_coords(ntime=3, time_dtype="datetime")
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.full((time.size, latitude.size, longitude.size), 900.0)
    ps_values[:, :, 0] = np.asarray([450.0, 650.0, 450.0])[:, None]
    ps_values[:, :, 1] = np.asarray([500.0, 500.0, 700.0])[:, None]
    ps = surface_pressure(time, latitude, longitude, ps_values)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.asarray(
            [
                [0.0, 150.0, 300.0, 450.0],
                [50.0, 200.0, 350.0, 500.0],
                [100.0, 250.0, 400.0, 550.0],
                [150.0, 300.0, 450.0, 600.0],
            ]
        ),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            180.0,
            235.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="temperature",
        units="K",
    )
    temperature_perturbed = temperature.where(theta > 0.0, temperature + 1000.0)
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            5.0,
            20.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            -4.0,
            3.0,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(
            -0.2,
            0.3,
            time.size * level.size * latitude.size * longitude.size,
        ).reshape(time.size, level.size, latitude.size, longitude.size),
        name="omega",
        units="Pa s-1",
    )

    base = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
    )
    perturbed = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        temperature=temperature_perturbed,
        pressure=pressure,
        ps=ps,
        phis=phis,
    )

    assert np.isfinite(base.values).all()
    assert np.isfinite(perturbed.values).all()
    np.testing.assert_allclose(base.values, perturbed.values)


def test_flat_surface_topographic_terms_and_totals_reduce_to_phase2():
    time, level, latitude, longitude = reference_case_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 950.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        reference_case_theta_profile(level, base=190.0, step=4.0)[None, :, None, None],
        name="temperature",
    )
    omega = full_field(time, level, latitude, longitude, np.linspace(-0.3, 0.3, time.size)[:, None, None, None], name="omega")
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha")
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")

    pt = potential_temperature(temperature, pressure)
    solution = KoehlerReferenceState().solve(pt, pressure, ps, phis=phis)

    np.testing.assert_allclose(
        solution.pi_s.values,
        np.broadcast_to(
            solution.reference_surface_pressure.values[:, None, None],
            solution.pi_s.shape,
        ),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(solution.pi_sZ.values, solution.pi_s.values, atol=1.0e-12)
    np.testing.assert_allclose(A_Z2(ps, phis, integrator, reference_state=solution).values, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(A_E2(ps, phis, integrator, reference_state=solution).values, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(A2(ps, phis, integrator, reference_state=solution).values, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        (
            A_Z(
                temperature,
                pressure,
                theta,
                integrator,
                reference_state=solution,
                ps=ps,
                phis=phis,
            )
            - A_Z1(temperature, pressure, theta, integrator, reference_state=solution)
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        (
            A_E(
                temperature,
                pressure,
                theta,
                integrator,
                reference_state=solution,
                ps=ps,
                phis=phis,
            )
            - A_E1(temperature, pressure, theta, integrator, reference_state=solution)
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        (
            A(
                temperature,
                pressure,
                theta,
                integrator,
                reference_state=solution,
                ps=ps,
                phis=phis,
            )
            - A1(temperature, pressure, theta, integrator, reference_state=solution)
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(C_Z2(ps, phis, integrator).values, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        (C_Z(omega, alpha, theta, integrator, ps=ps, phis=phis) - C_Z1(omega, alpha, theta, integrator)).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        C_K2(
            u,
            v,
            omega,
            theta,
            integrator,
            temperature=temperature,
            pressure=pressure,
            ps=ps,
            phis=phis,
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        (
            C_K(
                u,
                v,
                omega,
                theta,
                integrator,
                temperature=temperature,
                pressure=pressure,
                ps=ps,
                phis=phis,
            )
            - C_K1(u, v, omega, theta, integrator)
        ).values,
        0.0,
        atol=1.0e-12,
    )


def test_total_exact_az_budget_has_smaller_residual_than_body_only_budget():
    time, level, latitude, longitude = reference_case_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray([880.0, 900.0, 920.0])[:, None, None],
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        0.1 * reference_case_surface_geopotential_values(latitude, longitude),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)

    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        (
            reference_case_theta_profile(level)[None, :, None, None]
            + np.asarray([0.0, 2.0, 4.0])[:, None, None, None]
        ),
    )
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")
    alpha = full_field(time, level, latitude, longitude, 0.0, name="alpha")

    pt = potential_temperature(temperature, pressure)
    solution = KoehlerReferenceState().solve(pt, pressure, ps, phis=phis)

    az_body = A_Z1(
        temperature,
        pressure,
        theta,
        integrator,
        reference_state=solution,
    )
    az_total = A_Z(
        temperature,
        pressure,
        theta,
        integrator,
        reference_state=solution,
        ps=ps,
        phis=phis,
    )
    residual_body = np.abs(
        time_derivative(az_body.assign_coords(time=time)) + C_Z1(omega, alpha, theta, integrator)
    )
    residual_total = np.abs(
        time_derivative(az_total.assign_coords(time=time)) + C_Z(omega, alpha, theta, integrator, ps=ps, phis=phis)
    )

    assert np.isfinite(residual_body.values).all()
    assert np.isfinite(residual_total.values).all()
    residual_body_norm = float(np.linalg.norm(np.asarray(residual_body.values, dtype=float).ravel()))
    residual_total_norm = float(np.linalg.norm(np.asarray(residual_total.values, dtype=float).ravel()))

    assert residual_total_norm <= residual_body_norm + 1.0e-12
    assert np.any(np.asarray(residual_total.values, dtype=float) < np.asarray(residual_body.values, dtype=float) - 1.0e-12)


def test_topographic_ape_surface_terms_match_explicit_solution_overrides():
    time, level, latitude, longitude = reference_case_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, reference_case_surface_pressure_values(latitude, longitude))
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_geopotential_values(latitude, longitude),
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        reference_case_theta_profile(level)[None, :, None, None],
    )
    pt = potential_temperature(temperature, pressure)
    solution = KoehlerReferenceState().solve(pt, pressure, ps, phis=phis)

    direct_az2 = A_Z2(ps, phis, integrator, reference_state=solution)
    explicit_az2 = A_Z2(ps, phis, integrator, pi_sZ=solution.pi_sZ)
    direct_ae2 = A_E2(ps, phis, integrator, reference_state=solution)
    explicit_ae2 = A_E2(ps, phis, integrator, pi_s=solution.pi_s, pi_sZ=solution.pi_sZ)
    direct_a2 = A2(ps, phis, integrator, reference_state=solution)
    explicit_a2 = A2(ps, phis, integrator, pi_s=solution.pi_s)

    np.testing.assert_allclose(direct_az2.values, explicit_az2.values)
    np.testing.assert_allclose(direct_ae2.values, explicit_ae2.values)
    np.testing.assert_allclose(direct_a2.values, explicit_a2.values)


def test_topographic_ape_surface_terms_reject_zonal_pi_sz_override():
    time, level, latitude, longitude = make_coords(ntime=1)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(time, latitude, longitude, 200.0)
    pi_sz_zonal = surface_zonal_field(time, latitude, 800.0, name="pi_sZ_zonal")

    with pytest.raises(ValueError):
        A_Z2(ps, phis, integrator, pi_sZ=pi_sz_zonal)
    with pytest.raises(ValueError):
        A_E2(ps, phis, integrator, pi_s=ps, pi_sZ=pi_sz_zonal)


def test_total_exact_az_and_ae_reject_zonal_pi_sz_override():
    time, level, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(time, latitude, longitude, 200.0)
    theta = make_theta(pressure, ps)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0])[None, :, None, None],
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    pi_sz_zonal = surface_zonal_field(time, latitude, 800.0, name="pi_sZ_zonal")
    pt = potential_temperature(temperature, pressure)
    solution = KoehlerReferenceState().solve(pt, pressure, ps, phis=phis)

    with pytest.raises(ValueError):
        A_Z(
            temperature,
            pressure,
            theta,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
            pi_sZ=pi_sz_zonal,
        )
    with pytest.raises(ValueError):
        A_E(
            temperature,
            pressure,
            theta,
            integrator,
            reference_state=solution,
            ps=ps,
            phis=phis,
            pi_sZ=pi_sz_zonal,
        )


def test_cz2_rejects_same_shape_surface_coordinate_mismatch():
    time, level, latitude, longitude = make_coords(ntime=3)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0).assign_coords(
        longitude=longitude.values + 0.5
    )

    with pytest.raises(ValueError, match="Coordinate 'longitude'"):
        C_Z2(ps, phis, integrator)


def test_ck2_rejects_same_shape_full_field_coordinate_mismatch():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    geopotential = full_field(time, level, latitude, longitude, 0.0, name="geopotential").assign_coords(
        longitude=longitude.values + 0.5
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    with pytest.raises(ValueError, match="Coordinate 'longitude'"):
        C_K2(u, v, omega, theta, integrator, geopotential=geopotential)
