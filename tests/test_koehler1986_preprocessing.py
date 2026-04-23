from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.common.topography_measure import TopographyAwareMeasure
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import (
    build_theta_levels,
    interpolate_pressure_to_koehler_isentropes,
    koehler_isentropic_layer_mass_statistics,
    resolve_surface_potential_temperature,
)

from .helpers import make_coords, pressure_field, surface_pressure, temperature_from_theta_values


def _surface_theta_field(ps: xr.DataArray, value: float) -> xr.DataArray:
    field = xr.full_like(ps, float(value), dtype=float)
    field.name = "surface_potential_temperature"
    field.attrs["units"] = "K"
    return field


def _surface_temperature_for_theta(ps: xr.DataArray, theta_value: float) -> xr.DataArray:
    field = xr.full_like(ps, float(theta_value), dtype=float)
    field = field * (ps / MARS.p00) ** MARS.kappa
    field.name = "surface_temperature"
    field.attrs["units"] = "K"
    return field


def _build_stable_case(
    *,
    theta_profile: np.ndarray | list[float] = (190.0, 210.0, 230.0),
    ps_value: float = 650.0,
    surface_theta_value: float = 180.0,
):
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[700.0, 500.0, 300.0])
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, ps_value)
    pt = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        np.asarray(theta_profile, dtype=float)[None, :, None, None],
    )
    pt = pt * (MARS.p00 / pressure) ** MARS.kappa
    pt.name = "potential_temperature"
    surface_theta = _surface_theta_field(ps, surface_theta_value)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    return {
        "pressure": pressure,
        "ps": ps,
        "pt": pt,
        "surface_theta": surface_theta,
        "theta_mask": theta_mask,
        "integrator": integrator,
    }


def _column_sample(field: xr.DataArray) -> xr.DataArray:
    sample = field.isel(time=0, latitude=0, longitude=0)
    if ISENTROPIC_LAYER_DIM in sample.dims:
        return sample.transpose(ISENTROPIC_LAYER_DIM)
    if ISENTROPIC_DIM in sample.dims:
        return sample.transpose(ISENTROPIC_DIM)
    return sample


from mars_exact_lec.reference_state.interpolate_isentropes import ISENTROPIC_DIM, ISENTROPIC_LAYER_DIM


def test_build_theta_levels_public_exports_exist():
    assert build_theta_levels is not None
    assert resolve_surface_potential_temperature is not None
    assert interpolate_pressure_to_koehler_isentropes is not None
    assert koehler_isentropic_layer_mass_statistics is not None


def test_build_theta_levels_accepts_explicit_fixed_surfaces():
    case = _build_stable_case()

    levels = build_theta_levels(
        case["pt"],
        case["surface_theta"],
        theta_levels=[160.0, 180.0, 200.0, 220.0],
    )

    np.testing.assert_allclose(levels.values, np.asarray([160.0, 180.0, 200.0, 220.0]))
    assert levels.attrs["theta_level_semantics"] == "fixed_isentropic_surfaces"
    assert levels.attrs["construction"] == "explicit"


def test_build_theta_levels_from_increment_uses_surface_theta_and_above_ground_mask():
    case = _build_stable_case(surface_theta_value=175.0)

    levels = build_theta_levels(
        case["pt"],
        case["surface_theta"],
        theta_increment=20.0,
        theta_mask=case["theta_mask"],
    )

    np.testing.assert_allclose(levels.values, np.asarray([160.0, 180.0, 200.0, 220.0, 240.0]))


@pytest.mark.parametrize(
    ("theta_levels", "theta_increment", "message"),
    [
        ([160.0, 180.0], 20.0, "exactly one"),
        (None, None, "exactly one"),
    ],
)
def test_build_theta_levels_rejects_ambiguous_specification(theta_levels, theta_increment, message):
    case = _build_stable_case()

    with pytest.raises(ValueError, match=message):
        build_theta_levels(
            case["pt"],
            case["surface_theta"],
            theta_levels=theta_levels,
            theta_increment=theta_increment,
            theta_mask=case["theta_mask"],
        )


def test_resolve_surface_potential_temperature_prefers_explicit_field():
    case = _build_stable_case()

    surface_theta = resolve_surface_potential_temperature(
        pressure=case["pressure"],
        surface_pressure=case["ps"],
        surface_potential_temperature=case["surface_theta"],
        surface_temperature=_surface_temperature_for_theta(case["ps"], 999.0),
        template=case["ps"],
    )

    np.testing.assert_allclose(surface_theta.values, case["surface_theta"].values)


def test_resolve_surface_potential_temperature_uses_effective_surface_pressure_in_clip_mode():
    time, level, latitude, longitude = make_coords(ntime=1, level_values=[700.0, 500.0, 300.0])
    pressure = pressure_field(time, level, latitude, longitude)
    raw_ps = surface_pressure(time, latitude, longitude, 850.0)
    surface_temperature = xr.full_like(raw_ps, 200.0, dtype=float)
    surface_temperature.name = "surface_temperature"
    integrator = build_mass_integrator(level, latitude, longitude)
    measure = TopographyAwareMeasure.from_surface_pressure(
        level,
        raw_ps,
        integrator,
        surface_pressure_policy="clip",
    )

    surface_theta = resolve_surface_potential_temperature(
        pressure=pressure,
        surface_pressure=measure.effective_surface_pressure,
        surface_temperature=surface_temperature,
        template=raw_ps,
    )

    expected = 200.0 * (MARS.p00 / 800.0) ** MARS.kappa
    np.testing.assert_allclose(surface_theta.values, expected)


def test_resolve_surface_potential_temperature_requires_surface_theta_or_temperature():
    case = _build_stable_case()

    with pytest.raises(ValueError, match="requires 'surface_potential_temperature' or 'surface_temperature'"):
        resolve_surface_potential_temperature(
            pressure=case["pressure"],
            surface_pressure=case["ps"],
            template=case["ps"],
        )


def test_interpolate_pressure_to_koehler_isentropes_is_surface_aware():
    case = _build_stable_case()

    interpolation = interpolate_pressure_to_koehler_isentropes(
        case["pt"],
        case["pressure"],
        case["ps"],
        case["surface_theta"],
        [160.0, 180.0, 200.0, 220.0, 240.0],
        theta_mask=case["theta_mask"],
        interpolation_space="pressure",
    )

    np.testing.assert_allclose(
        _column_sample(interpolation["pressure_on_theta"]).values,
        np.asarray([650.0, 650.0, 550.0, 400.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        _column_sample(interpolation["is_below_surface"]).values,
        np.asarray([True, True, False, False, False]),
    )
    np.testing.assert_array_equal(
        _column_sample(interpolation["is_free_atmosphere"]).values,
        np.asarray([False, False, True, True, False]),
    )
    np.testing.assert_array_equal(
        _column_sample(interpolation["is_above_model_top"]).values,
        np.asarray([False, False, False, False, True]),
    )
    assert bool(interpolation["column_interpolation_valid"].isel(time=0, latitude=0, longitude=0))
    assert float(interpolation["column_theta_min"].isel(time=0, latitude=0, longitude=0)) == 180.0
    assert float(interpolation["column_theta_max"].isel(time=0, latitude=0, longitude=0)) == 230.0
    assert float(interpolation["column_top_pressure"].isel(time=0, latitude=0, longitude=0)) == 300.0
    assert float(interpolation["column_bottom_pressure"].isel(time=0, latitude=0, longitude=0)) == 500.0
    assert float(interpolation["column_top_edge_pressure"].isel(time=0, latitude=0, longitude=0)) == 200.0
    assert int(interpolation["valid_level_count"].isel(time=0, latitude=0, longitude=0)) == 2


def test_interpolate_pressure_to_koehler_isentropes_reports_monotonic_repair_and_reject():
    case = _build_stable_case(theta_profile=[190.0, 185.0, 230.0], surface_theta_value=170.0, ps_value=750.0)
    targets = [160.0, 180.0, 200.0, 220.0]

    repaired = interpolate_pressure_to_koehler_isentropes(
        case["pt"],
        case["pressure"],
        case["ps"],
        case["surface_theta"],
        targets,
        theta_mask=case["theta_mask"],
        monotonic_policy="repair",
        interpolation_space="pressure",
    )
    rejected = interpolate_pressure_to_koehler_isentropes(
        case["pt"],
        case["pressure"],
        case["ps"],
        case["surface_theta"],
        targets,
        theta_mask=case["theta_mask"],
        monotonic_policy="reject",
        interpolation_space="pressure",
    )

    assert int(repaired["monotonic_violations"].isel(time=0, latitude=0, longitude=0)) == 1
    assert int(repaired["monotonic_repairs"].isel(time=0, latitude=0, longitude=0)) >= 1
    assert bool(repaired["column_interpolation_valid"].isel(time=0, latitude=0, longitude=0))

    assert int(rejected["monotonic_violations"].isel(time=0, latitude=0, longitude=0)) == 1
    assert int(rejected["monotonic_repairs"].isel(time=0, latitude=0, longitude=0)) == 0
    assert not bool(rejected["column_interpolation_valid"].isel(time=0, latitude=0, longitude=0))
    assert np.isnan(_column_sample(rejected["pressure_on_theta"]).values).all()


def test_koehler_isentropic_layer_mass_statistics_close_layers_with_surface_and_top_edge():
    case = _build_stable_case()
    interpolation = interpolate_pressure_to_koehler_isentropes(
        case["pt"],
        case["pressure"],
        case["ps"],
        case["surface_theta"],
        [160.0, 180.0, 200.0, 220.0, 240.0],
        theta_mask=case["theta_mask"],
        interpolation_space="pressure",
    )

    stats = koehler_isentropic_layer_mass_statistics(
        interpolation,
        integrator=case["integrator"],
    )

    np.testing.assert_allclose(
        _column_sample(stats["interface_pressure"]).values,
        np.asarray([650.0, 650.0, 550.0, 400.0, 200.0]),
    )
    np.testing.assert_allclose(
        _column_sample(stats["layer_pressure_thickness"]).values,
        np.asarray([0.0, 100.0, 150.0, 200.0]),
    )
    np.testing.assert_allclose(
        stats["mean_pressure_on_theta"].isel(time=0).values,
        np.asarray([650.0, 650.0, 550.0, 400.0, 200.0]),
    )
    np.testing.assert_allclose(
        stats["below_surface_area_fraction"].isel(time=0).values,
        np.asarray([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        stats["free_atmosphere_area_fraction"].isel(time=0).values,
        np.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(
        stats["above_model_top_area_fraction"].isel(time=0).values,
        np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
    )

    total_area = float(case["integrator"].cell_area.sum())
    expected_total_mass = total_area * (650.0 - 200.0) / MARS.g
    np.testing.assert_allclose(stats["layer_mass"].sum(dim=ISENTROPIC_LAYER_DIM).values, expected_total_mass)
    np.testing.assert_allclose(stats.coords["lower_theta"].values, np.asarray([160.0, 180.0, 200.0, 220.0]))
    np.testing.assert_allclose(stats.coords["upper_theta"].values, np.asarray([180.0, 200.0, 220.0, 240.0]))
    assert stats.attrs["mass_mode"] == "koehler1986_observed_state"
    assert stats.attrs["surface_pressure_behavior"] == "surface_aware"
