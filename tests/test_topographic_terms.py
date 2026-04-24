from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.boer.conversions import (
    C_K,
    C_K1,
    C_K2,
    C_Z,
    C_Z1,
    C_Z2,
    _area_weighted_zonal_mean,
    _ck2_interface_geopotential_face_stars,
    _ck2_finite_volume_terms,
    _longitude_gradient,
)
from mars_exact_lec.boer.reservoirs import A, A1, A2, A_E, A_E1, A_E2, A_Z, A_Z1, A_Z2
from mars_exact_lec.common.geopotential import reconstruct_hydrostatic_geopotential
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.normalization import to_per_area
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.common.time_derivatives import time_derivative
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import FiniteVolumeReferenceState, potential_temperature

from .helpers import (
    full_field,
    make_coords,
    pressure_field,
    reference_case_coords,
    reference_case_surface_geopotential_values,
    reference_case_surface_pressure_values,
    reference_case_theta_profile,
    surface_pressure_policy_for_case,
    surface_geopotential,
    surface_pressure,
    surface_zonal_field,
    temperature_from_theta_values,
)
from .helpers_exact import build_flat_reference_case as build_shared_flat_reference_case


_CK2_BOUNDARY_ATTRS = {
    "ck2_discretization": "cut_cell_finite_volume_leibniz_corrected",
    "ck2_geopotential_source": "level_center_geopotential",
    "ck2_geopotential_mode": "strict",
    "ck2_geopotential_reconstruction_allowed": False,
    "ck2_geopotential_reconstruction_approximate": False,
    "ck2_vertical_integral": "trapezoidal_phi_star_dp",
    "ck2_reconstruction": "linear_pressure_phi_star_to_layer_faces",
    "ck2_bottom_pressure": "min(layer_lower_edge,effective_surface_pressure)_clamped_to_layer",
    "ck2_horizontal_boundary_correction": "subtract_phi_bottom_grad_p_bottom",
    "ck2_pressure_term": "phi_bottom_minus_phi_top_over_full_delta_p",
    "ck2_zonal_mean": "cell_area_weighted_full_longitude_ring",
    "ck2_derivative_mask": "finite_volume_above_ground_pressure_thickness_positive",
}


def _solver_for_case(
    ps,
    level,
    *,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
) -> FiniteVolumeReferenceState:
    return FiniteVolumeReferenceState(
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
        surface_pressure_policy=surface_pressure_policy_for_case(
            ps,
            level,
            pressure_tolerance=pressure_tolerance,
        ),
    )


def _assert_ck2_boundary_attrs(term):
    for key, value in _CK2_BOUNDARY_ATTRS.items():
        assert term.attrs[key] == value


def _interface_geopotential_field(time, level_edges, latitude, longitude, values):
    values = np.asarray(values, dtype=float)
    shape = (time.size, level_edges.size, latitude.size, longitude.size)
    if values.shape != shape:
        values = np.broadcast_to(values, shape)
    return xr.DataArray(
        values,
        dims=("time", "level_edge", "latitude", "longitude"),
        coords={
            "time": time,
            "level_edge": level_edges.values,
            "latitude": latitude,
            "longitude": longitude,
        },
        name="interface_geopotential",
        attrs={"units": "m2 s-2"},
    )


def _reference_convergence_guard_case():
    time, level, latitude, longitude = make_coords(ntime=2)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    phis = surface_geopotential(time, latitude, longitude, 10.0)
    theta = make_theta(pressure, ps)
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray([180.0, 200.0, 220.0], dtype=float)[None, :, None, None],
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    return {
        "temperature": temperature,
        "pressure": pressure,
        "theta": theta,
        "integrator": integrator,
        "measure": measure,
        "ps": ps,
        "phis": phis,
    }


def _fake_reference_state_for_convergence_guard(case, *, full=True, zonal=True):
    time = case["ps"].coords["time"]

    def flag(name, value):
        values = np.asarray(value)
        if values.shape == ():
            values = np.full(time.size, values.item())
        else:
            values = np.broadcast_to(values, (time.size,))
        return xr.DataArray(
            values,
            dims=("time",),
            coords={"time": time.values},
            name=name,
        )

    pressure = case["pressure"]
    ps = case["ps"]
    return {
        "n": xr.full_like(case["temperature"], 0.1).rename("N"),
        "n_z": xr.full_like(pressure.isel(longitude=0, drop=True), 0.1).rename("N_Z"),
        "pi_s": xr.full_like(ps, 760.0).rename("pi_s"),
        "pi_sZ": xr.full_like(ps, 780.0).rename("pi_sZ"),
        "converged": flag("reference_state_converged", full),
        "converged_zonal": flag("reference_state_converged_zonal", zonal),
    }


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
    ps = surface_pressure(time, latitude, longitude, 800.0)
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

    np.testing.assert_allclose(C_Z2(ps, phis, integrator).values, 0.0, atol=1.0)


def test_ck2_vanishes_for_zonally_symmetric_geopotential():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    policy = surface_pressure_policy_for_case(ps, level)
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
        C_K2(u, v, omega, theta, integrator, geopotential=geopotential, ps=ps, surface_pressure_policy=policy).values,
        0.0,
        atol=1.0e-12,
    )


def test_ck2_cut_cell_constant_geopotential_null_mode_with_moving_topography():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_row = np.asarray([720.0, 740.0, 760.0, 780.0], dtype=float)
    ps_values = np.broadcast_to(ps_row[None, None, :], (time.size, latitude.size, longitude.size)).copy()
    ps_values += np.asarray([0.0, 10.0, 20.0], dtype=float)[:, None, None]
    ps = surface_pressure(time, latitude, longitude, ps_values)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    policy = surface_pressure_policy_for_case(ps, level)
    geopotential = full_field(time, level, latitude, longitude, 1234.0, name="geopotential", units="m2 s-2")
    u = full_field(time, level, latitude, longitude, 12.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, -3.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega", units="Pa s-1")

    ck2 = C_K2(u, v, omega, theta, integrator, geopotential=geopotential, ps=ps, surface_pressure_policy=policy)

    np.testing.assert_allclose(ck2.values, 0.0, rtol=0.0, atol=1.0e-9)


def test_ck2_cut_cell_flat_topography_periodic_eddy_gradient_cancels():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=8)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    lon_wave = np.sin(np.deg2rad(longitude.values))[None, None, None, :]
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        100.0 * np.broadcast_to(lon_wave, (time.size, level.size, latitude.size, longitude.size)),
        name="geopotential",
        units="m2 s-2",
    )
    u = full_field(time, level, latitude, longitude, 10.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega", units="Pa s-1")

    ck2 = C_K2(u, v, omega, theta, integrator, geopotential=geopotential, ps=ps)

    np.testing.assert_allclose(ck2.values, 0.0, rtol=0.0, atol=1.0e-8)


def test_ck2_cut_cell_linear_pressure_term_matches_public_oracle():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_row = np.asarray([720.0, 740.0, 760.0, 780.0], dtype=float)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_row[None, :], (latitude.size, longitude.size)),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    beta = 0.04
    q = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float)
    geopotential_values = beta * pressure.values * q[None, None, None, :]
    geopotential = full_field(time, level, latitude, longitude, geopotential_values, name="geopotential", units="m2 s-2")
    u = full_field(time, level, latitude, longitude, 0.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega0 = 0.3
    omega = full_field(time, level, latitude, longitude, omega0, name="omega", units="Pa s-1")

    ck2 = C_K2(u, v, omega, theta, integrator, geopotential=geopotential, ps=ps)

    f_bottom = (ps_row - 600.0) / 200.0
    qbar_bottom = np.average(q, weights=f_bottom)
    qbar_top = np.mean(q)
    pressure_slope = beta * (700.0 * (q - qbar_bottom) - 500.0 * (q - qbar_top)) / 200.0
    term_bottom = float(np.mean(f_bottom * pressure_slope))
    term_top = float(np.mean(pressure_slope))
    expected = omega0 * (
        integrator.zonal_mass_weights.sel(level=700.0) * term_bottom
        + integrator.zonal_mass_weights.sel(level=500.0) * term_top
    ).sum(dim="latitude")
    np.testing.assert_allclose(ck2.values, expected.broadcast_like(ck2).values, rtol=1.0e-12, atol=1.0e-5)


def test_ck2_interface_geopotential_linear_pressure_term_matches_public_oracle():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_row = np.asarray([720.0, 740.0, 760.0, 780.0], dtype=float)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_row[None, :], (latitude.size, longitude.size)),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    beta = 0.04
    q = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float)
    interface_geopotential = _interface_geopotential_field(
        time,
        measure.level_edges,
        latitude,
        longitude,
        beta * measure.level_edges.values[None, :, None, None] * q[None, None, None, :],
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega0 = 0.3
    omega = full_field(time, level, latitude, longitude, omega0, name="omega", units="Pa s-1")

    ck2 = C_K2(u, v, omega, theta, integrator, interface_geopotential=interface_geopotential, ps=ps)

    cell_fraction = (ps_row - 600.0) / 200.0
    qbar_bottom = np.average(q, weights=cell_fraction)
    bottom_mean = np.average(ps_row * q, weights=cell_fraction)
    pressure_slope = beta * (ps_row * q - bottom_mean - 600.0 * (q - qbar_bottom)) / 200.0
    term_bottom = float(np.mean(pressure_slope))
    assert abs(term_bottom) > 1.0e-12
    expected = omega0 * (
        integrator.zonal_mass_weights.sel(level=700.0) * term_bottom
    ).sum(dim="latitude")
    np.testing.assert_allclose(ck2.values, expected.broadcast_like(ck2).values, rtol=1.0e-12, atol=1.0e-5)


def test_ck2_interface_geopotential_uses_provided_faces():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0, 300.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    q = np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=float)
    face_amplitude = np.asarray([100.0, 40.0, 10.0, -20.0], dtype=float)
    interface_values = face_amplitude[None, :, None, None] * q[None, None, None, :]
    interface_geopotential = _interface_geopotential_field(
        time,
        measure.level_edges,
        latitude,
        longitude,
        interface_values,
    )

    top_star, bottom_star, reconstruction = _ck2_interface_geopotential_face_stars(
        interface_geopotential,
        theta,
        measure,
    )

    assert reconstruction == "interface_geopotential_faces_pressure_linear_partial_bottom"
    np.testing.assert_allclose(
        top_star.sel(level=700.0).isel(time=0, latitude=0).values,
        40.0 * q,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        bottom_star.sel(level=700.0).isel(time=0, latitude=0).values,
        100.0 * q,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ck2_interface_geopotential_uses_surface_geopotential_for_partial_bottom():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 650.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    q = np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=float)
    face_amplitude = np.asarray([10.0, 30.0, 90.0], dtype=float)
    interface_geopotential = _interface_geopotential_field(
        time,
        measure.level_edges,
        latitude,
        longitude,
        face_amplitude[None, :, None, None] * q[None, None, None, :],
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        22.0 * q[None, None, :],
    )
    center_geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        np.asarray([24.0, 70.0], dtype=float)[None, :, None, None] * q[None, None, None, :],
        name="geopotential",
        units="m2 s-2",
    )

    _, bottom_from_surface, surface_reconstruction = _ck2_interface_geopotential_face_stars(
        interface_geopotential,
        theta,
        measure,
        phis=phis,
    )
    _, bottom_from_center, center_reconstruction = _ck2_interface_geopotential_face_stars(
        interface_geopotential,
        theta,
        measure,
        center_geopotential=center_geopotential,
    )
    _, bottom_from_interfaces, interface_reconstruction = _ck2_interface_geopotential_face_stars(
        interface_geopotential,
        theta,
        measure,
    )

    assert surface_reconstruction == "interface_geopotential_faces_surface_partial_bottom"
    assert center_reconstruction == "interface_geopotential_faces_center_linear_partial_bottom"
    assert interface_reconstruction == "interface_geopotential_faces_pressure_linear_partial_bottom"
    np.testing.assert_allclose(
        bottom_from_surface.sel(level=700.0).isel(time=0, latitude=0).values,
        22.0 * q,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        bottom_from_center.sel(level=700.0).isel(time=0, latitude=0).values,
        27.0 * q,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        bottom_from_interfaces.sel(level=700.0).isel(time=0, latitude=0).values,
        25.0 * q,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ck2_interface_geopotential_ignores_inactive_below_ground_face_fill():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_row = np.asarray([500.0, 520.0, 540.0, 560.0], dtype=float)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_row[None, :], (latitude.size, longitude.size)),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    q = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float)
    interface_values = np.asarray([1.0e6, 40.0, 10.0], dtype=float)[None, :, None, None] * q[None, None, None, :]
    low_fill = _interface_geopotential_field(time, measure.level_edges, latitude, longitude, interface_values)
    high_values = interface_values.copy()
    high_values[:, 0, :, :] = -1.0e6 * q[None, None, :]
    high_fill = _interface_geopotential_field(time, measure.level_edges, latitude, longitude, high_values)
    u = full_field(time, level, latitude, longitude, 0.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.2, name="omega", units="Pa s-1")

    reference = C_K2(u, v, omega, theta, integrator, interface_geopotential=low_fill, ps=ps)
    perturbed = C_K2(u, v, omega, theta, integrator, interface_geopotential=high_fill, ps=ps)

    assert np.isfinite(reference.values).all()
    assert np.any(np.abs(np.asarray(reference.values, dtype=float)) > 1.0e-12)
    np.testing.assert_allclose(reference.values, perturbed.values, rtol=0.0, atol=1.0e-8)


def test_ck2_finite_volume_terms_apply_sloping_bottom_leibniz_correction():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=8)
    ps_row = 740.0 + 30.0 * np.cos(np.deg2rad(longitude.values))
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_row[None, :], (latitude.size, longitude.size)),
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    lon_wave = np.sin(np.deg2rad(longitude.values))[None, None, None, :]
    phi_star = full_field(
        time,
        level,
        latitude,
        longitude,
        40.0 * np.broadcast_to(lon_wave, (time.size, level.size, latitude.size, longitude.size)),
        name="phi_star",
        units="m2 s-2",
    )

    tendency, gradient_x, gradient_y, pressure_term = _ck2_finite_volume_terms(
        phi_star,
        measure,
        integrator,
        constants=MARS,
    )

    expected_gradient_x = _area_weighted_zonal_mean(
        measure.cell_fraction * _longitude_gradient(phi_star, constants=MARS),
        integrator,
    )
    assert float(abs(expected_gradient_x).max()) > 1.0e-12
    np.testing.assert_allclose(gradient_x.values, expected_gradient_x.values, rtol=1.0e-12, atol=1.0e-18)
    np.testing.assert_allclose(tendency.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(gradient_y.values, 0.0, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(pressure_term.values, 0.0, rtol=0.0, atol=1.0e-12)


def test_ck2_finite_volume_terms_apply_meridional_sloping_bottom_leibniz_correction():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlon=4)
    latitude_radians = np.deg2rad(latitude.values)
    ps_by_latitude = 700.0 + 20.0 * latitude_radians
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_by_latitude[:, None], (latitude.size, longitude.size)),
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    meridional_slope = 80.0
    phi_star = full_field(
        time,
        level,
        latitude,
        longitude,
        meridional_slope * latitude_radians[None, None, :, None],
        name="phi_star",
        units="m2 s-2",
    )

    tendency, gradient_x, gradient_y, pressure_term = _ck2_finite_volume_terms(
        phi_star,
        measure,
        integrator,
        constants=MARS,
    )

    expected_gradient_y = (measure.cell_fraction * meridional_slope / MARS.a).isel(longitude=0, drop=True)
    assert float(abs(expected_gradient_y).max()) > 1.0e-12
    assert float(measure.cell_fraction.sel(level=700.0).min()) < 1.0
    np.testing.assert_allclose(gradient_y.values, expected_gradient_y.values, rtol=1.0e-12, atol=1.0e-18)
    np.testing.assert_allclose(tendency.values, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(gradient_x.values, 0.0, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(pressure_term.values, 0.0, rtol=0.0, atol=1.0e-12)


def test_ck2_finite_volume_terms_apply_moving_bottom_leibniz_correction():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    ps_values = np.asarray([720.0, 740.0, 760.0], dtype=float)[:, None, None]
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_values, (time.size, latitude.size, longitude.size)),
    )
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    tendency_rate = 0.05
    time_seconds = np.asarray(time.values, dtype=float) * 3600.0
    phi_star = full_field(
        time,
        level,
        latitude,
        longitude,
        tendency_rate * time_seconds[:, None, None, None],
        name="phi_star",
        units="m2 s-2",
    )

    tendency, gradient_x, gradient_y, pressure_term = _ck2_finite_volume_terms(
        phi_star,
        measure,
        integrator,
        constants=MARS,
    )

    expected_tendency = _area_weighted_zonal_mean(measure.cell_fraction * tendency_rate, integrator)
    np.testing.assert_allclose(tendency.values, expected_tendency.values, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(gradient_x.values, 0.0, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(gradient_y.values, 0.0, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(pressure_term.values, 0.0, rtol=0.0, atol=1.0e-12)


def test_ck2_finite_volume_terms_cancel_moving_bottom_for_static_pressure_linear_phi():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    ps_values = np.asarray([720.0, 740.0, 760.0], dtype=float)[:, None, None]
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_values, (time.size, latitude.size, longitude.size)),
    )
    pressure = pressure_field(time, level, latitude, longitude)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    beta = 0.04
    q = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float)
    phi_star = full_field(
        time,
        level,
        latitude,
        longitude,
        beta * pressure.values * q[None, None, None, :],
        name="phi_star",
        units="m2 s-2",
    )

    tendency, gradient_x, gradient_y, pressure_term = _ck2_finite_volume_terms(
        phi_star,
        measure,
        integrator,
        constants=MARS,
    )

    np.testing.assert_allclose(tendency.values, 0.0, rtol=0.0, atol=1.0e-12)
    assert float(abs(pressure_term).max()) > 1.0e-12
    np.testing.assert_allclose(gradient_y.values, 0.0, rtol=0.0, atol=1.0e-18)


def test_geopotential_reconstruction_matches_explicit_field_in_ck2():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    policy = surface_pressure_policy_for_case(ps, level)
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
        theta_mask=theta,
    )

    explicit = C_K2(u, v, omega, theta, integrator, geopotential=geopotential, ps=ps, surface_pressure_policy=policy)
    with pytest.raises(ValueError, match="geopotential_mode='hydrostatic'"):
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
            surface_pressure_policy=policy,
        )
    with pytest.raises(ValueError, match="geopotential_mode='hydrostatic'"):
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
            surface_pressure_policy=policy,
        )
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
        surface_pressure_policy=policy,
        geopotential_mode="hydrostatic",
    )
    ck_total = C_K(
        u,
        v,
        omega,
        theta,
        integrator,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
        surface_pressure_policy=policy,
        geopotential_mode="hydrostatic",
    )
    with pytest.warns(FutureWarning, match="allow_geopotential_reconstruction"):
        legacy = C_K2(
            u,
            v,
            omega,
            theta,
            integrator,
            temperature=temperature,
            pressure=pressure,
            ps=ps,
            phis=phis,
            surface_pressure_policy=policy,
            allow_geopotential_reconstruction=True,
        )
    with pytest.warns(FutureWarning, match="allow_geopotential_reconstruction"):
        with pytest.raises(ValueError, match="only one of 'geopotential_mode'"):
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
                surface_pressure_policy=policy,
                geopotential_mode="fill",
                allow_geopotential_reconstruction=True,
            )

    assert explicit.attrs["ck2_geopotential_source"] == "hydrostatically_reconstructed_geopotential"
    assert explicit.attrs["ck2_geopotential_mode"] == "strict"
    assert explicit.attrs["ck2_geopotential_reconstruction_allowed"] is False
    assert explicit.attrs["ck2_geopotential_reconstruction_approximate"] is True
    assert reconstructed.attrs["ck2_geopotential_source"] == "hydrostatically_reconstructed_geopotential"
    assert reconstructed.attrs["ck2_geopotential_mode"] == "hydrostatic"
    assert reconstructed.attrs["ck2_geopotential_reconstruction_allowed"] is True
    assert reconstructed.attrs["ck2_geopotential_reconstruction_approximate"] is True
    assert ck_total.attrs["ck2_geopotential_mode"] == "hydrostatic"
    assert ck_total.attrs["ck2_geopotential_reconstruction_allowed"] is True
    np.testing.assert_allclose(reconstructed.values, explicit.values)
    np.testing.assert_allclose(legacy.values, reconstructed.values)


def test_ck2_geopotential_mode_fill_only_fills_explicit_geopotential_gaps():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    policy = surface_pressure_policy_for_case(ps, level)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(10.0, 40.0, level.size)[None, :, None, None],
        name="geopotential",
        units="m2 s-2",
    )
    geopotential = geopotential.where(~((geopotential.coords["level"] == level.values[1]) & (theta > 0.0)))
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")
    temperature = full_field(time, level, latitude, longitude, 210.0, name="temperature", units="K")
    phis = surface_geopotential(time, latitude, longitude, 0.0)

    with pytest.raises(ValueError, match="non-finite values"):
        C_K2(
            u,
            v,
            omega,
            theta,
            integrator,
            geopotential=geopotential,
            pressure=pressure,
            ps=ps,
            surface_pressure_policy=policy,
        )
    filled = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=geopotential,
        pressure=pressure,
        ps=ps,
        geopotential_mode="fill",
        surface_pressure_policy=policy,
    )
    with pytest.raises(ValueError, match="requires an explicit 'geopotential'"):
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
            geopotential_mode="fill",
            surface_pressure_policy=policy,
        )

    assert filled.attrs["ck2_geopotential_mode"] == "fill"
    assert filled.attrs["ck2_geopotential_source"] == "level_center_geopotential_with_log_pressure_fill"
    assert filled.attrs["ck2_geopotential_reconstruction_allowed"] is True
    assert filled.attrs["ck2_geopotential_reconstruction_approximate"] is True
    assert np.isfinite(filled.values).all()


def test_geopotential_reconstruction_deprecated_theta_keyword_warns_and_matches_theta_mask():
    time, level, latitude, longitude = make_coords(ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(time, latitude, longitude, 125.0)
    theta = make_theta(pressure, ps)
    temperature = full_field(time, level, latitude, longitude, 210.0, name="temperature", units="K")

    expected = reconstruct_hydrostatic_geopotential(
        temperature,
        pressure,
        phis,
        ps=ps,
        theta_mask=theta,
    )

    with pytest.warns(FutureWarning, match="theta.*deprecated"):
        deprecated = reconstruct_hydrostatic_geopotential(
            temperature,
            pressure,
            phis,
            ps=ps,
            theta=theta,
        )

    np.testing.assert_allclose(deprecated.values, expected.values)


def test_ck2_remains_finite_when_surface_crosses_a_level_in_time():
    time, level, latitude, longitude = make_coords(ntime=3, time_dtype="datetime")
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.full((time.size, latitude.size, longitude.size), 900.0)
    ps_values[:, :, 0] = np.asarray([450.0, 650.0, 450.0])[:, None]
    ps = surface_pressure(time, latitude, longitude, ps_values)
    policy = surface_pressure_policy_for_case(ps, level)
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
        theta_mask=theta,
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
        surface_pressure_policy=policy,
        geopotential_mode="hydrostatic",
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
        surface_pressure_policy=policy,
    )

    assert np.isfinite(reconstructed.values).all()
    assert np.isfinite(explicit.values).all()
    assert reconstructed.attrs["ck2_geopotential_source"] == "hydrostatically_reconstructed_geopotential"
    np.testing.assert_allclose(explicit.values, reconstructed.values)


def test_ck2_explicit_geopotential_takes_precedence_over_reconstruction_inputs():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    phis = surface_geopotential(time, latitude, longitude, 100.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        25.0 * np.sin(np.deg2rad(longitude.values))[None, None, None, :],
        name="geopotential",
        units="m2 s-2",
    )
    temperature = full_field(time, level, latitude, longitude, 180.0, name="temperature", units="K")
    hot_temperature = full_field(time, level, latitude, longitude, 280.0, name="temperature", units="K")
    u = full_field(time, level, latitude, longitude, 3.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.1, name="omega", units="Pa s-1")

    reference = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=geopotential,
        temperature=temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
        geopotential_mode="hydrostatic",
    )
    perturbed = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=geopotential,
        temperature=hot_temperature,
        pressure=pressure,
        ps=ps,
        phis=phis,
        geopotential_mode="hydrostatic",
    )

    assert reference.attrs["ck2_geopotential_source"] == "level_center_geopotential"
    assert reference.attrs["ck2_geopotential_reconstruction_approximate"] is False
    np.testing.assert_allclose(reference.values, perturbed.values)


def test_ck2_ignores_below_ground_explicit_geopotential_fill():
    time, level, latitude, longitude = make_coords(ntime=3, time_dtype="datetime")
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.full((time.size, latitude.size, longitude.size), 900.0)
    ps_values[:, :, 0] = np.asarray([450.0, 650.0, 450.0])[:, None]
    ps_values[:, :, 1] = np.asarray([500.0, 500.0, 700.0])[:, None]
    ps = surface_pressure(time, latitude, longitude, ps_values)
    policy = surface_pressure_policy_for_case(ps, level)
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
        theta_mask=theta,
    )
    explicit_low = reconstructed_phi.where(theta > 0.0, -1.0e3)
    explicit_high = reconstructed_phi.where(theta > 0.0, 1.0e6)

    low_fill = C_K2(u, v, omega, theta, integrator, geopotential=explicit_low, ps=ps, surface_pressure_policy=policy)
    high_fill = C_K2(u, v, omega, theta, integrator, geopotential=explicit_high, ps=ps, surface_pressure_policy=policy)

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
    policy = surface_pressure_policy_for_case(ps, level)
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
        surface_pressure_policy=policy,
        geopotential_mode="hydrostatic",
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
        surface_pressure_policy=policy,
        geopotential_mode="hydrostatic",
    )

    assert np.isfinite(base.values).all()
    assert np.isfinite(perturbed.values).all()
    np.testing.assert_allclose(base.values, perturbed.values)


def test_ck2_is_invariant_under_longitude_cyclic_shift_with_wraparound_segment():
    time, level, latitude, longitude = make_coords(ntime=3, nlon=8)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_row = np.asarray([760.0, 755.0, 650.0, 650.0, 650.0, 650.0, 650.0, 770.0], dtype=float)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_row[None, :], (latitude.size, longitude.size)),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    policy = surface_pressure_policy_for_case(ps, level)

    geopotential_values = np.zeros((time.size, level.size, latitude.size, longitude.size), dtype=float)
    lat_scale = np.linspace(1.0, 2.5, latitude.size, dtype=float)[:, None]
    lon_pattern = np.asarray([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0], dtype=float)[None, :]
    geopotential_values[:, 0, :, :] = 200.0 + 25.0 * lat_scale * lon_pattern
    geopotential_values[:, 1, :, :] = 120.0
    geopotential_values[:, 2, :, :] = 80.0
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        geopotential_values,
        name="geopotential",
        units="m2 s-2",
    )

    u = full_field(time, level, latitude, longitude, 12.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega", units="Pa s-1")

    reference = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=geopotential,
        ps=ps,
        surface_pressure_policy=policy,
    )

    ps_shifted = ps.roll(longitude=2, roll_coords=False)
    geopotential_shifted = geopotential.roll(longitude=2, roll_coords=False)
    theta_shifted = make_theta(pressure, ps_shifted)
    shifted = C_K2(
        u,
        v,
        omega,
        theta_shifted,
        integrator,
        geopotential=geopotential_shifted,
        ps=ps_shifted,
        surface_pressure_policy=surface_pressure_policy_for_case(ps_shifted, level),
    )

    assert np.isfinite(reference.values).all()
    assert np.isfinite(shifted.values).all()
    assert np.any(np.abs(np.asarray(reference.values, dtype=float)) > 1.0e-12)
    np.testing.assert_allclose(reference.values, shifted.values, rtol=0.0, atol=1.0e-12)


def test_ck2_handles_short_longitude_cut_cell_segments_without_boundary_fill_dependence():
    time, level, latitude, longitude = make_coords(ntime=3, nlat=2, nlon=8)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_values = np.asarray(
        [
            [760.0, 650.0, 650.0, 650.0, 650.0, 650.0, 650.0, 650.0],
            [650.0, 650.0, 760.0, 760.0, 650.0, 650.0, 650.0, 650.0],
        ],
        dtype=float,
    )
    ps = surface_pressure(time, latitude, longitude, ps_values)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    policy = surface_pressure_policy_for_case(ps, level)

    geopotential_values = np.zeros((time.size, level.size, latitude.size, longitude.size), dtype=float)
    geopotential_values[:, 0, 0, 0] = 100.0
    geopotential_values[:, 0, 1, 2] = 10.0
    geopotential_values[:, 0, 1, 3] = 20.0
    geopotential_values[:, 1, :, :] = 120.0
    geopotential_values[:, 2, :, :] = 80.0
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        geopotential_values,
        name="geopotential",
        units="m2 s-2",
    )
    geopotential_singleton_perturbed = geopotential.copy(deep=True)
    geopotential_singleton_perturbed.loc[
        dict(level=level.values[0], latitude=latitude.values[0], longitude=longitude.values[0])
    ] = 1.0e6

    explicit_low = geopotential.where(theta > 0.0, -1.0e4)
    explicit_high = geopotential.where(theta > 0.0, 1.0e6)
    explicit_singleton = geopotential_singleton_perturbed.where(theta > 0.0, -1.0e4)

    u = full_field(time, level, latitude, longitude, 15.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega", units="Pa s-1")

    low_fill = C_K2(u, v, omega, theta, integrator, geopotential=explicit_low, ps=ps, surface_pressure_policy=policy)
    high_fill = C_K2(u, v, omega, theta, integrator, geopotential=explicit_high, ps=ps, surface_pressure_policy=policy)
    singleton_perturbed = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=explicit_singleton,
        ps=ps,
        surface_pressure_policy=policy,
    )

    assert np.isfinite(low_fill.values).all()
    assert np.isfinite(high_fill.values).all()
    assert np.isfinite(singleton_perturbed.values).all()
    assert np.any(np.abs(np.asarray(low_fill.values, dtype=float)) > 1.0e-12)
    np.testing.assert_allclose(low_fill.values, high_fill.values, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(low_fill.values, singleton_perturbed.values, rtol=1.0e-9, atol=1.0e-6)


def test_ck2_meridional_fragmented_mask_is_stable_for_latitude_segment_boundaries():
    time, level, latitude, longitude = make_coords(ntime=3, nlat=6, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps_by_latitude = np.asarray([650.0, 740.0, 760.0, 780.0, 650.0, 650.0], dtype=float)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.broadcast_to(ps_by_latitude[:, None], (latitude.size, longitude.size)),
    )
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    policy = surface_pressure_policy_for_case(ps, level)

    geopotential_values = np.zeros((time.size, level.size, latitude.size, longitude.size), dtype=float)
    lat_profile = np.asarray([-3.0, -1.0, 0.5, 2.0, 3.0, 4.0], dtype=float)[:, None]
    lon_profile = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)[None, :]
    geopotential_values[:, 0, :, :] = 150.0 + 30.0 * lat_profile * lon_profile
    geopotential_values[:, 1, :, :] = 120.0
    geopotential_values[:, 2, :, :] = 80.0
    geopotential = full_field(
        time,
        level,
        latitude,
        longitude,
        geopotential_values,
        name="geopotential",
        units="m2 s-2",
    )

    u = full_field(time, level, latitude, longitude, 0.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 5.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega", units="Pa s-1")

    reference = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        geopotential=geopotential,
        ps=ps,
        surface_pressure_policy=policy,
    )

    ps_mirror = ps.copy(data=np.flip(ps.values, axis=1))
    theta_mirror = make_theta(pressure, ps_mirror)
    geopotential_mirror = geopotential.copy(data=np.flip(geopotential.values, axis=2))
    v_mirror = v.copy(data=-np.flip(v.values, axis=2))
    mirrored = C_K2(
        u,
        v_mirror,
        omega,
        theta_mirror,
        integrator,
        geopotential=geopotential_mirror,
        ps=ps_mirror,
        surface_pressure_policy=surface_pressure_policy_for_case(ps_mirror, level),
    )

    truncated_columns = np.count_nonzero(np.asarray(theta.isel(time=0, level=0).values, dtype=bool), axis=0)
    assert np.any((truncated_columns > 0) & (truncated_columns < latitude.size))
    assert np.isfinite(reference.values).all()
    assert np.isfinite(mirrored.values).all()
    np.testing.assert_allclose(reference.values, 0.0, rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(mirrored.values, 0.0, rtol=0.0, atol=1.0e-4)


def test_flat_surface_topographic_terms_and_totals_reduce_to_phase2():
    time, level, latitude, longitude = reference_case_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 925.0)
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
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

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
            - A_Z1(temperature, pressure, theta, integrator, reference_state=solution, ps=ps)
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
            - A_E1(temperature, pressure, theta, integrator, reference_state=solution, ps=ps)
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
            - A1(temperature, pressure, theta, integrator, reference_state=solution, ps=ps)
        ).values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(C_Z2(ps, phis, integrator).values, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        (C_Z(omega, alpha, theta, integrator, ps=ps, phis=phis) - C_Z1(omega, alpha, theta, integrator, ps=ps)).values,
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
            geopotential_mode="hydrostatic",
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
                geopotential_mode="hydrostatic",
            )
            - C_K1(u, v, omega, theta, integrator, ps=ps)
        ).values,
        0.0,
        atol=1.0e-12,
    )


def test_flat_surface_limit_with_nonzero_constant_phis_still_zeroes_surface_terms():
    case = build_shared_flat_reference_case(
        ntime=3,
        level_values=(900.0, 700.0, 500.0),
        ps_value=1000.0,
        phis_value=2000.0,
        theta_profile=(210.0, 230.0, 250.0),
    )
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    ps = case["ps"]
    phis = case["phis"]
    theta = case["theta_mask"]
    temperature = case["temperature"]
    integrator = case["integrator"]
    measure = case["measure"]
    solution = case["solution"]
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.3, 0.3, time.size)[:, None, None, None],
        name="omega",
        units="Pa s-1",
    )
    alpha = full_field(time, level, latitude, longitude, 0.8, name="alpha", units="m3 kg-1")

    np.testing.assert_allclose(solution.pi_s.values, ps.values, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(solution.pi_sZ.values, ps.values, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(A_Z2(ps, phis, integrator, reference_state=solution, measure=measure).values, 0.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(A_E2(ps, phis, integrator, reference_state=solution, measure=measure).values, 0.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(A2(ps, phis, integrator, reference_state=solution, measure=measure).values, 0.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(C_Z2(ps, phis, integrator, measure=measure).values, 0.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        A_Z(temperature, pressure, theta, integrator, reference_state=solution, measure=measure, ps=ps, phis=phis).values,
        A_Z1(temperature, pressure, theta, integrator, reference_state=solution, measure=measure, ps=ps).values,
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        A_E(temperature, pressure, theta, integrator, reference_state=solution, measure=measure, ps=ps, phis=phis).values,
        A_E1(temperature, pressure, theta, integrator, reference_state=solution, measure=measure, ps=ps).values,
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        A(temperature, pressure, theta, integrator, reference_state=solution, measure=measure, ps=ps, phis=phis).values,
        A1(temperature, pressure, theta, integrator, reference_state=solution, measure=measure, ps=ps).values,
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        C_Z(omega, alpha, theta, integrator, measure=measure, ps=ps, phis=phis).values,
        C_Z1(omega, alpha, theta, integrator, measure=measure, ps=ps).values,
        rtol=1.0e-12,
        atol=1.0e-10,
    )


def test_flat_surface_limit_ck2_vanishes_with_nonzero_zonal_mean_flow():
    case = build_shared_flat_reference_case(
        ntime=3,
        level_values=(900.0, 700.0, 500.0),
        ps_value=1000.0,
        phis_value=2000.0,
        theta_profile=(210.0, 230.0, 250.0),
    )
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    ps = case["ps"]
    phis = case["phis"]
    theta = case["theta_mask"]
    temperature = case["temperature"]
    integrator = case["integrator"]
    measure = case["measure"]
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(4.0, 16.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-2.0, 5.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.2, 0.3, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="omega",
        units="Pa s-1",
    )
    geopotential = full_field(time, level, latitude, longitude, 0.0, name="geopotential", units="m2 s-2")

    ck2 = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        measure=measure,
        geopotential=geopotential,
        ps=ps,
    )
    ck = C_K(
        u,
        v,
        omega,
        theta,
        integrator,
        measure=measure,
        geopotential=geopotential,
        ps=ps,
    )
    ck1 = C_K1(u, v, omega, theta, integrator, measure=measure, ps=ps)

    np.testing.assert_allclose(ck2.values, 0.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(ck.values, ck1.values, rtol=1.0e-12, atol=1.0e-10)


def test_ck2_and_ck_carry_boundary_scheme_attrs():
    case = build_shared_flat_reference_case(
        ntime=3,
        level_values=(900.0, 700.0, 500.0),
        ps_value=1000.0,
        phis_value=2000.0,
        theta_profile=(210.0, 230.0, 250.0),
    )
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    ps = case["ps"]
    phis = case["phis"]
    theta = case["theta_mask"]
    temperature = case["temperature"]
    integrator = case["integrator"]
    measure = case["measure"]
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(4.0, 16.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-2.0, 5.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.2, 0.3, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="omega",
        units="Pa s-1",
    )
    geopotential = full_field(time, level, latitude, longitude, 0.0, name="geopotential", units="m2 s-2")

    ck2 = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        measure=measure,
        geopotential=geopotential,
        ps=ps,
    )
    ck = C_K(
        u,
        v,
        omega,
        theta,
        integrator,
        measure=measure,
        geopotential=geopotential,
        ps=ps,
    )

    _assert_ck2_boundary_attrs(ck2)
    _assert_ck2_boundary_attrs(ck)


def test_ck2_and_ck_accept_interface_geopotential_and_carry_source_attrs():
    time, level, latitude, longitude = make_coords(ntime=3, level_values=[700.0, 500.0], nlat=2, nlon=4)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 800.0)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator)
    q = np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=float)
    interface_geopotential = _interface_geopotential_field(
        time,
        measure.level_edges,
        latitude,
        longitude,
        np.asarray([100.0, 40.0, 10.0], dtype=float)[None, :, None, None] * q[None, None, None, :],
    )
    u = full_field(time, level, latitude, longitude, 3.0, name="u", units="m s-1")
    v = full_field(time, level, latitude, longitude, 0.0, name="v", units="m s-1")
    omega = full_field(time, level, latitude, longitude, 0.1, name="omega", units="Pa s-1")

    ck2 = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        interface_geopotential=interface_geopotential,
        ps=ps,
    )
    ck = C_K(
        u,
        v,
        omega,
        theta,
        integrator,
        interface_geopotential=interface_geopotential,
        ps=ps,
    )

    assert np.isfinite(ck2.values).all()
    assert np.isfinite(ck.values).all()
    assert ck2.attrs["ck2_geopotential_source"] == "interface_geopotential"
    assert ck.attrs["ck2_geopotential_source"] == "interface_geopotential"
    assert ck2.attrs["ck2_reconstruction"] == "interface_geopotential_faces_pressure_linear_partial_bottom"
    assert ck.attrs["ck2_reconstruction"] == "interface_geopotential_faces_pressure_linear_partial_bottom"


def test_ck2_per_area_preserves_boundary_scheme_attrs():
    case = build_shared_flat_reference_case(
        ntime=3,
        level_values=(900.0, 700.0, 500.0),
        ps_value=1000.0,
        phis_value=2000.0,
        theta_profile=(210.0, 230.0, 250.0),
    )
    time = case["time"]
    level = case["level"]
    latitude = case["latitude"]
    longitude = case["longitude"]
    pressure = case["pressure"]
    ps = case["ps"]
    phis = case["phis"]
    theta = case["theta_mask"]
    temperature = case["temperature"]
    integrator = case["integrator"]
    measure = case["measure"]
    u = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(4.0, 16.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="u",
        units="m s-1",
    )
    v = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-2.0, 5.0, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="v",
        units="m s-1",
    )
    omega = full_field(
        time,
        level,
        latitude,
        longitude,
        np.linspace(-0.2, 0.3, time.size * level.size * latitude.size).reshape(
            time.size,
            level.size,
            latitude.size,
        )[..., None],
        name="omega",
        units="Pa s-1",
    )
    geopotential = full_field(time, level, latitude, longitude, 0.0, name="geopotential", units="m2 s-2")

    ck2 = C_K2(
        u,
        v,
        omega,
        theta,
        integrator,
        measure=measure,
        geopotential=geopotential,
        ps=ps,
    )
    ck = C_K(
        u,
        v,
        omega,
        theta,
        integrator,
        measure=measure,
        geopotential=geopotential,
        ps=ps,
    )

    ck2_per_area = to_per_area(ck2)
    ck_per_area = to_per_area(ck)

    _assert_ck2_boundary_attrs(ck2_per_area)
    _assert_ck2_boundary_attrs(ck_per_area)
    assert ck2_per_area.attrs["units"] == "W m-2"
    assert ck_per_area.attrs["units"] == "W m-2"


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
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

    az_body = A_Z1(
        temperature,
        pressure,
        theta,
        integrator,
        reference_state=solution,
        ps=ps,
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
        time_derivative(az_body.assign_coords(time=time)) + C_Z1(omega, alpha, theta, integrator, ps=ps)
    )
    residual_total = np.abs(
        time_derivative(az_total.assign_coords(time=time)) + C_Z(omega, alpha, theta, integrator, ps=ps, phis=phis)
    )

    assert np.isfinite(residual_body.values).all()
    assert np.isfinite(residual_total.values).all()
    residual_body_norm = float(np.linalg.norm(np.asarray(residual_body.values, dtype=float).ravel()))
    residual_total_norm = float(np.linalg.norm(np.asarray(residual_total.values, dtype=float).ravel()))

    assert residual_total_norm <= 1.01 * residual_body_norm + 1.0e-12
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
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

    direct_az2 = A_Z2(ps, phis, integrator, reference_state=solution)
    explicit_az2 = A_Z2(ps, phis, integrator, pi_sZ=solution.pi_sZ)
    direct_ae2 = A_E2(ps, phis, integrator, reference_state=solution)
    explicit_ae2 = A_E2(ps, phis, integrator, pi_s=solution.pi_s, pi_sZ=solution.pi_sZ)
    direct_a2 = A2(ps, phis, integrator, reference_state=solution)
    explicit_a2 = A2(ps, phis, integrator, pi_s=solution.pi_s)

    np.testing.assert_allclose(direct_az2.values, explicit_az2.values)
    np.testing.assert_allclose(direct_ae2.values, explicit_ae2.values)
    np.testing.assert_allclose(direct_a2.values, explicit_a2.values)


def test_reference_state_convergence_guard_blocks_full_family_consumers_only():
    case = _reference_convergence_guard_case()
    reference_state = _fake_reference_state_for_convergence_guard(case, full=False, zonal=True)

    with pytest.raises(ValueError, match="full solve did not converge"):
        A(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            phis=case["phis"],
        )
    with pytest.raises(ValueError, match="full solve did not converge"):
        A_E(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            phis=case["phis"],
        )

    az = A_Z(
        case["temperature"],
        case["pressure"],
        case["theta"],
        case["integrator"],
        reference_state=reference_state,
        measure=case["measure"],
        ps=case["ps"],
        phis=case["phis"],
    )
    assert np.isfinite(az.values).all()


def test_reference_state_convergence_guard_blocks_zonal_family_consumers_only():
    case = _reference_convergence_guard_case()
    reference_state = _fake_reference_state_for_convergence_guard(case, full=True, zonal=False)

    with pytest.raises(ValueError, match="zonal solve did not converge"):
        A_Z(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            phis=case["phis"],
        )
    with pytest.raises(ValueError, match="zonal solve did not converge"):
        A_E(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            phis=case["phis"],
        )

    total = A(
        case["temperature"],
        case["pressure"],
        case["theta"],
        case["integrator"],
        reference_state=reference_state,
        measure=case["measure"],
        ps=case["ps"],
        phis=case["phis"],
    )
    assert np.isfinite(total.values).all()


def test_reference_state_convergence_guard_treats_missing_values_as_failed():
    case = _reference_convergence_guard_case()
    reference_state = _fake_reference_state_for_convergence_guard(case, full=np.nan, zonal=True)

    with pytest.raises(ValueError, match="full solve did not converge"):
        A1(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
        )


def test_reference_state_convergence_guard_requires_status_when_reference_state_is_used():
    case = _reference_convergence_guard_case()
    reference_state = _fake_reference_state_for_convergence_guard(case, full=True, zonal=True)
    reference_state.pop("converged")

    with pytest.raises(ValueError, match="'converged' is missing"):
        A1(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
        )


def test_reference_state_convergence_guard_respects_explicit_overrides():
    case = _reference_convergence_guard_case()
    reference_state = _fake_reference_state_for_convergence_guard(case, full=False, zonal=False)

    terms = [
        A1(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            n=reference_state["n"],
        ),
        A_Z1(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            n_z=reference_state["n_z"],
        ),
        A_E1(
            case["temperature"],
            case["pressure"],
            case["theta"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            ps=case["ps"],
            n=reference_state["n"],
            n_z=reference_state["n_z"],
        ),
        A2(
            case["ps"],
            case["phis"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            pi_s=reference_state["pi_s"],
        ),
        A_Z2(
            case["ps"],
            case["phis"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            pi_sZ=reference_state["pi_sZ"],
        ),
        A_E2(
            case["ps"],
            case["phis"],
            case["integrator"],
            reference_state=reference_state,
            measure=case["measure"],
            pi_s=reference_state["pi_s"],
            pi_sZ=reference_state["pi_sZ"],
        ),
    ]
    for term in terms:
        assert np.isfinite(term.values).all()


def test_measure_aware_surface_energy_terms_use_effective_surface_pressure_and_preserve_partition():
    time, level, latitude, longitude = make_coords(ntime=1)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.linspace(100.0, 400.0, latitude.size * longitude.size).reshape(latitude.size, longitude.size),
    )
    pi_s = surface_pressure(time, latitude, longitude, 760.0)
    pi_sZ = surface_pressure(time, latitude, longitude, 780.0)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        surface_pressure_policy="clip",
    )

    az2 = A_Z2(ps, phis, integrator, pi_sZ=pi_sZ, measure=measure)
    ae2 = A_E2(ps, phis, integrator, pi_s=pi_s, pi_sZ=pi_sZ, measure=measure)
    a2 = A2(ps, phis, integrator, pi_s=pi_s, measure=measure)
    expected_a2 = integrator.integrate_surface((measure.effective_surface_pressure - pi_s) * phis)

    np.testing.assert_allclose((az2 + ae2).values, a2.values)
    np.testing.assert_allclose(a2.values, expected_a2.values)
    assert az2.attrs["surface_pressure_policy"] == "clip"
    assert ae2.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert a2.attrs["not_exact_full_atmosphere"] is True


def test_clip_policy_auto_measure_surface_terms_match_explicit_measure():
    time, level, latitude, longitude = make_coords(ntime=3)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray([820.0, 900.0, 780.0])[:, None, None],
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        np.linspace(100.0, 400.0, latitude.size * longitude.size).reshape(latitude.size, longitude.size),
    )
    pi_s = surface_pressure(time, latitude, longitude, 760.0)
    pi_sZ = surface_pressure(time, latitude, longitude, 780.0)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        surface_pressure_policy="clip",
    )

    auto_az2 = A_Z2(ps, phis, integrator, pi_sZ=pi_sZ, surface_pressure_policy="clip")
    explicit_az2 = A_Z2(ps, phis, integrator, pi_sZ=pi_sZ, measure=measure)
    auto_ae2 = A_E2(ps, phis, integrator, pi_s=pi_s, pi_sZ=pi_sZ, surface_pressure_policy="clip")
    explicit_ae2 = A_E2(ps, phis, integrator, pi_s=pi_s, pi_sZ=pi_sZ, measure=measure)
    auto_a2 = A2(ps, phis, integrator, pi_s=pi_s, surface_pressure_policy="clip")
    explicit_a2 = A2(ps, phis, integrator, pi_s=pi_s, measure=measure)
    auto_cz2 = C_Z2(ps, phis, integrator, surface_pressure_policy="clip")
    explicit_cz2 = C_Z2(ps, phis, integrator, measure=measure)

    np.testing.assert_allclose(auto_az2.values, explicit_az2.values)
    np.testing.assert_allclose(auto_ae2.values, explicit_ae2.values)
    np.testing.assert_allclose(auto_a2.values, explicit_a2.values)
    np.testing.assert_allclose(auto_cz2.values, explicit_cz2.values)


def test_measure_aware_cz2_differentiates_effective_surface_pressure_under_clip_policy():
    time, level, latitude, longitude = make_coords(ntime=3)
    integrator = build_mass_integrator(level, latitude, longitude)
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        np.asarray([820.0, 900.0, 780.0])[:, None, None],
    )
    phis = surface_geopotential(time, latitude, longitude, 200.0)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        ps,
        integrator,
        surface_pressure_policy="clip",
    )

    clipped = C_Z2(ps, phis, integrator, measure=measure)
    expected = -integrator.integrate_surface(time_derivative(measure.effective_surface_pressure) * phis)

    np.testing.assert_allclose(clipped.values, expected.values)
    assert clipped.attrs["surface_pressure_policy"] == "clip"
    assert clipped.attrs["domain"] == "truncated_to_model_pressure_domain"
    assert clipped.attrs["not_exact_full_atmosphere"] is True
    with pytest.raises(ValueError, match="Surface pressure extends below the deepest model pressure interface"):
        C_Z2(ps, phis, integrator)


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
    solution = _solver_for_case(ps, level).solve(pt, pressure, ps, phis=phis)

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
    policy = surface_pressure_policy_for_case(ps, level)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    geopotential = full_field(time, level, latitude, longitude, 0.0, name="geopotential").assign_coords(
        longitude=longitude.values + 0.5
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    with pytest.raises(ValueError, match="Coordinate 'longitude'"):
        C_K2(u, v, omega, theta, integrator, geopotential=geopotential, ps=ps, surface_pressure_policy=policy)


def test_ck2_rejects_interface_geopotential_coordinate_mismatch():
    time, level, latitude, longitude = make_coords(ntime=3)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 900.0)
    policy = surface_pressure_policy_for_case(ps, level)
    theta = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(level, ps, integrator, surface_pressure_policy=policy)
    interface_geopotential = _interface_geopotential_field(
        time,
        measure.level_edges,
        latitude,
        longitude,
        0.0,
    )
    u = full_field(time, level, latitude, longitude, 0.0, name="u")
    v = full_field(time, level, latitude, longitude, 0.0, name="v")
    omega = full_field(time, level, latitude, longitude, 0.0, name="omega")

    with pytest.raises(ValueError, match="Coordinate 'longitude'"):
        C_K2(
            u,
            v,
            omega,
            theta,
            integrator,
            interface_geopotential=interface_geopotential.assign_coords(longitude=longitude.values + 0.5),
            ps=ps,
            surface_pressure_policy=policy,
        )

    short_edges = measure.level_edges.isel(level_edge=slice(0, -1))
    short_interface = _interface_geopotential_field(time, short_edges, latitude, longitude, 0.0)
    with pytest.raises(ValueError, match="one more level_edge"):
        C_K2(
            u,
            v,
            omega,
            theta,
            integrator,
            interface_geopotential=short_interface,
            ps=ps,
            surface_pressure_policy=policy,
        )

    wrong_edge_values = interface_geopotential.assign_coords(level_edge=measure.level_edges.values + 10.0)
    with pytest.raises(ValueError, match="level_edge coordinate"):
        C_K2(
            u,
            v,
            omega,
            theta,
            integrator,
            interface_geopotential=wrong_edge_values,
            ps=ps,
            surface_pressure_policy=policy,
        )
