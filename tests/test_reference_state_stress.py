from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mars_exact_lec.boer.reservoirs import A
from mars_exact_lec.common.integrals import build_mass_integrator
from mars_exact_lec.constants_mars import MARS
from mars_exact_lec.io.mask_below_ground import make_theta
from mars_exact_lec.reference_state import KoehlerReferenceState, potential_temperature

from .helpers import (
    finite_reference_profile,
    pressure_field,
    pressure_inside_reference_layer,
    reference_case_coords,
    reference_case_surface_geopotential_values,
    reference_case_surface_pressure_values,
    reference_case_surface_time_series,
    reference_case_theta_field_values,
    reference_layer_mass_from_interfaces,
    surface_geopotential,
    surface_mass_from_pi_s,
    surface_pressure,
    temperature_from_theta_values,
)


pytestmark = pytest.mark.slow_reference


def _build_stress_case(
    *,
    grid: str,
    ntime: int,
    ps_base: float,
    ps_lon_drop: float,
    ps_lat_drop: float,
    ps_time_offsets,
    phis_lat_range: float,
    phis_lon_range: float,
    phis_time_offsets,
    theta_base: float = 175.0,
    theta_step: float = 5.0,
    theta_time_offsets=0.0,
    theta_lat_amplitude: float = 8.0,
    theta_lon_amplitude: float = 12.0,
    pressure_tolerance: float = 1.0e-6,
    max_iterations: int = 64,
):
    time, level, latitude, longitude = reference_case_coords(grid=grid, ntime=ntime)
    pressure = pressure_field(time, level, latitude, longitude)

    ps_2d = reference_case_surface_pressure_values(
        latitude,
        longitude,
        base=ps_base,
        lon_drop=ps_lon_drop,
        lat_drop=ps_lat_drop,
    )
    phis_2d = reference_case_surface_geopotential_values(
        latitude,
        longitude,
        lat_range=phis_lat_range,
        lon_range=phis_lon_range,
    )
    ps = surface_pressure(
        time,
        latitude,
        longitude,
        reference_case_surface_time_series(ps_2d, ps_time_offsets),
    )
    phis = surface_geopotential(
        time,
        latitude,
        longitude,
        reference_case_surface_time_series(phis_2d, phis_time_offsets),
    )
    theta_values = reference_case_theta_field_values(
        time,
        level,
        latitude,
        longitude,
        base=theta_base,
        step=theta_step,
        time_offsets=theta_time_offsets,
        lat_amplitude=theta_lat_amplitude,
        lon_amplitude=theta_lon_amplitude,
    )
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_values,
    )
    pt = potential_temperature(temperature, pressure)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    solver = KoehlerReferenceState(
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
    )
    solution = solver.solve(pt, pressure, ps, phis=phis)
    return {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "pt": pt,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "temperature": temperature,
        "solver": solver,
        "solution": solution,
    }


def _build_flat_partial_bottom_case(*, pressure_tolerance: float, max_iterations: int = 128):
    time, level, latitude, longitude = reference_case_coords(grid="gaussian", ntime=1)
    pressure = pressure_field(time, level, latitude, longitude)
    ps = surface_pressure(time, latitude, longitude, 875.0)
    phis = surface_geopotential(time, latitude, longitude, 0.0)
    theta_values = reference_case_theta_field_values(
        time,
        level,
        latitude,
        longitude,
        base=180.0,
        step=5.0,
        time_offsets=0.0,
        lat_amplitude=0.0,
        lon_amplitude=0.0,
    )
    temperature = temperature_from_theta_values(
        time,
        level,
        latitude,
        longitude,
        theta_values,
    )
    pt = potential_temperature(temperature, pressure)
    theta_mask = make_theta(pressure, ps)
    integrator = build_mass_integrator(level, latitude, longitude)
    solver = KoehlerReferenceState(
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
    )
    solution = solver.solve(pt, pressure, ps, phis=phis)
    return {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
        "pressure": pressure,
        "ps": ps,
        "phis": phis,
        "pt": pt,
        "theta_mask": theta_mask,
        "integrator": integrator,
        "temperature": temperature,
        "solver": solver,
        "solution": solution,
    }


def _upper_half_mass(
    profile: dict[str, np.ndarray],
    layer_index: int,
    phis_2d,
    cell_area_2d,
) -> float:
    phis_values = np.asarray(phis_2d.values, dtype=float)
    area_values = np.asarray(cell_area_2d.values, dtype=float)
    theta_layer = float(profile["theta_reference"][layer_index])
    p_bottom = float(profile["reference_interface_pressure"][layer_index])
    p_top = float(profile["reference_interface_pressure"][layer_index + 1])
    phi_bottom = float(profile["reference_interface_geopotential"][layer_index])
    phi_top = float(profile["reference_interface_geopotential"][layer_index + 1])
    pi_sample = float(profile["pi_reference"][layer_index])

    local_bottom = np.full_like(phis_values, p_top, dtype=float)
    full_mask = phis_values <= phi_bottom
    if np.any(full_mask):
        local_bottom[full_mask] = p_bottom

    partial_mask = (phis_values > phi_bottom) & (phis_values < phi_top)
    if np.any(partial_mask):
        local_bottom[partial_mask] = pressure_inside_reference_layer(
            phis_values[partial_mask],
            phi_bottom,
            p_bottom,
            theta_layer,
        )

    return float(
        np.sum(np.maximum(np.minimum(local_bottom, pi_sample) - p_top, 0.0) * area_values) / MARS.g
    )


def _reference_error_report(case, solution, *, compute_flat_ape: bool = False) -> dict[str, float]:
    total_mass = np.asarray(solution.total_mass.values, dtype=float)
    surface_mass = np.asarray(
        surface_mass_from_pi_s(
            solution.pi_s,
            solution.reference_top_pressure,
            case["integrator"].cell_area,
        ).values,
        dtype=float,
    )
    surface_mass_ratio = float(np.max(np.abs(surface_mass - total_mass) / np.maximum(total_mass, 1.0)))

    layer_mass_ratio = 0.0
    half_mass_ratio = 0.0
    top_interface_ratio = 0.0

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        total_mass_t = max(float(solution.total_mass.isel(time=time_index)), 1.0)
        phis_relative = case["phis"].isel(time=time_index) - case["phis"].isel(time=time_index).min(
            dim=("latitude", "longitude")
        )
        top_interface_ratio = max(
            top_interface_ratio,
            abs(
                float(profile["reference_interface_pressure"][-1])
                - float(solution.reference_top_pressure.isel(time=time_index))
            )
            / max(float(solution.reference_top_pressure.isel(time=time_index)), 1.0),
        )

        for layer_index, theta_layer in enumerate(profile["theta_reference"]):
            reconstructed_mass = reference_layer_mass_from_interfaces(
                profile["reference_interface_pressure"][layer_index],
                profile["reference_interface_pressure"][layer_index + 1],
                profile["reference_interface_geopotential"][layer_index],
                profile["reference_interface_geopotential"][layer_index + 1],
                theta_layer,
                phis_relative,
                case["integrator"].cell_area,
            )
            layer_mass_ratio = max(
                layer_mass_ratio,
                abs(reconstructed_mass - float(profile["mass_reference"][layer_index])) / total_mass_t,
            )
            half_mass_ratio = max(
                half_mass_ratio,
                abs(
                    _upper_half_mass(
                        profile,
                        layer_index,
                        phis_relative,
                        case["integrator"].cell_area,
                    )
                    - 0.5 * float(profile["mass_reference"][layer_index])
                )
                / total_mass_t,
            )

    flat_ape_ratio = float("nan")
    if compute_flat_ape:
        total_ape = np.asarray(
            A(
                case["temperature"],
                case["pressure"],
                case["theta_mask"],
                case["integrator"],
                reference_state=solution,
                ps=case["ps"],
                phis=case["phis"],
            ).values,
            dtype=float,
        )
        theta_max = float(np.nanmax(np.asarray(case["pt"].values, dtype=float)))
        scale = np.maximum(total_mass * solution.constants.cp * theta_max, 1.0)
        flat_ape_ratio = float(np.max(np.abs(total_ape) / scale))

    return {
        "surface_mass_ratio": surface_mass_ratio,
        "layer_mass_ratio": layer_mass_ratio,
        "half_mass_ratio": half_mass_ratio,
        "top_interface_ratio": top_interface_ratio,
        "flat_ape_ratio": flat_ape_ratio,
    }


def _assert_reference_stress_contracts(case, *, compute_flat_ape: bool = False) -> dict[str, float]:
    solution = case["solution"]
    pressure_tolerance = case["solver"].pressure_tolerance

    assert solution.converged is not None
    assert solution.converged_zonal is not None
    assert solution.converged.values.all()
    assert solution.converged_zonal.values.all()
    assert np.all(solution.iterations.values <= case["solver"].max_iterations)
    assert np.all(solution.iterations_zonal.values <= case["solver"].max_iterations)

    pi = solution.reference_pressure(case["pt"], pressure=case["pressure"])
    n = solution.efficiency(case["pt"], case["pressure"])

    assert np.isfinite(pi.values).all()
    assert np.isfinite(n.values).all()
    assert np.isfinite(solution.pi_reference.values).any()
    assert np.isfinite(solution.reference_interface_pressure.values).any()
    assert np.isfinite(solution.reference_interface_geopotential.values).any()
    assert np.isfinite(solution.pi_s.values).all()
    assert np.isfinite(solution.pi_sZ.values).all()

    for time_index in range(case["time"].size):
        profile = finite_reference_profile(solution, time_index=time_index)
        assert profile["reference_interface_pressure"].size == profile["theta_reference"].size + 1
        assert np.all(np.diff(profile["theta_reference"]) > 0.0)
        assert np.all(np.diff(profile["pi_reference"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_pressure"]) < 0.0)
        assert np.all(np.diff(profile["reference_interface_geopotential"]) > 0.0)

    report = _reference_error_report(case, solution, compute_flat_ape=compute_flat_ape)
    assert report["surface_mass_ratio"] <= 20.0 * pressure_tolerance
    assert report["layer_mass_ratio"] <= 20.0 * pressure_tolerance
    assert report["half_mass_ratio"] <= 20.0 * pressure_tolerance
    assert report["top_interface_ratio"] <= 10.0 * pressure_tolerance
    if compute_flat_ape:
        assert report["flat_ape_ratio"] <= 1.0e-15
    return report


def test_reference_state_stress_regular_extreme_low_ps_long_run():
    case = _build_stress_case(
        grid="regular",
        ntime=6,
        ps_base=980.0,
        ps_lon_drop=640.0,
        ps_lat_drop=180.0,
        ps_time_offsets=[40.0, 0.0, -40.0, -60.0, -20.0, 20.0],
        phis_lat_range=1200.0,
        phis_lon_range=1800.0,
        phis_time_offsets=[0.0, 200.0, 400.0, 600.0, 400.0, 200.0],
        theta_time_offsets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        theta_lat_amplitude=2.0,
        theta_lon_amplitude=4.0,
    )

    _assert_reference_stress_contracts(case)


def test_reference_state_stress_gaussian_extreme_low_ps_long_run():
    case = _build_stress_case(
        grid="gaussian",
        ntime=6,
        ps_base=970.0,
        ps_lon_drop=600.0,
        ps_lat_drop=160.0,
        ps_time_offsets=[20.0, 0.0, -20.0, -40.0, -20.0, 0.0],
        phis_lat_range=900.0,
        phis_lon_range=1500.0,
        phis_time_offsets=[0.0, 150.0, 300.0, 450.0, 300.0, 150.0],
        theta_time_offsets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        theta_lat_amplitude=8.0,
        theta_lon_amplitude=12.0,
    )

    _assert_reference_stress_contracts(case)


def test_reference_state_stress_time_varying_surface_crosses_many_levels():
    case = _build_stress_case(
        grid="regular",
        ntime=8,
        ps_base=930.0,
        ps_lon_drop=540.0,
        ps_lat_drop=120.0,
        ps_time_offsets=[100.0, 40.0, -20.0, -80.0, -40.0, 20.0, 80.0, 40.0],
        phis_lat_range=1000.0,
        phis_lon_range=1600.0,
        phis_time_offsets=[0.0, 80.0, 160.0, 240.0, 320.0, 240.0, 160.0, 80.0],
        theta_time_offsets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        theta_lat_amplitude=4.0,
        theta_lon_amplitude=8.0,
    )
    level_count_span = (
        case["theta_mask"].sum(dim="level").max(dim="time") - case["theta_mask"].sum(dim="level").min(dim="time")
    )
    assert np.any(np.asarray(level_count_span.values, dtype=float) >= 4.0)

    _assert_reference_stress_contracts(case)


def test_reference_state_tolerance_ladder_scales_residuals():
    tolerances = [1.0e-4, 1.0e-6, 1.0e-8]
    reports = {}

    for pressure_tolerance in tolerances:
        case = _build_stress_case(
            grid="regular",
            ntime=4,
            ps_base=950.0,
            ps_lon_drop=520.0,
            ps_lat_drop=120.0,
            ps_time_offsets=[20.0, 0.0, -20.0, 10.0],
            phis_lat_range=650.0,
            phis_lon_range=1000.0,
            phis_time_offsets=[0.0, 100.0, 200.0, 100.0],
            theta_time_offsets=[0.0, 0.0, 0.0, 0.0],
            theta_lat_amplitude=2.0,
            theta_lon_amplitude=4.0,
            pressure_tolerance=pressure_tolerance,
            max_iterations=128,
        )
        reports[pressure_tolerance] = _assert_reference_stress_contracts(case)

    floor = 5.0e-16
    ladder_metrics = (
        "surface_mass_ratio",
        "layer_mass_ratio",
        "half_mass_ratio",
        "top_interface_ratio",
    )
    for metric in ladder_metrics:
        assert reports[1.0e-6][metric] <= reports[1.0e-4][metric] + floor
        assert reports[1.0e-8][metric] <= reports[1.0e-6][metric] + floor

    assert any(
        reports[1.0e-8][metric] <= 0.1 * reports[1.0e-4][metric] + floor
        for metric in ladder_metrics
    )

    flat_default = _build_flat_partial_bottom_case(pressure_tolerance=1.0e-6)
    flat_default_report = _assert_reference_stress_contracts(flat_default, compute_flat_ape=True)
    flat_strict = _build_flat_partial_bottom_case(pressure_tolerance=1.0e-8)
    flat_strict_report = _assert_reference_stress_contracts(flat_strict, compute_flat_ape=True)
    assert flat_strict_report["flat_ape_ratio"] <= flat_default_report["flat_ape_ratio"] + 5.0e-16
