from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from .test_reference_state_stress import (
    _assert_reference_stress_contracts,
    _build_flat_partial_bottom_case,
    _build_stress_case,
    _upper_half_mass,
)


pytestmark = [pytest.mark.slow_reference, pytest.mark.legacy_reference_internal]


def test_reference_state_finite_volume_stress_helpers_cover_half_mass_contract():
    case = _build_stress_case(
        grid="regular",
        ntime=2,
        ps_base=950.0,
        ps_lon_drop=520.0,
        ps_lat_drop=120.0,
        ps_time_offsets=[20.0, -20.0],
        phis_lat_range=650.0,
        phis_lon_range=1000.0,
        phis_time_offsets=[0.0, 100.0],
        theta_time_offsets=[0.0, 0.0],
        theta_lat_amplitude=2.0,
        theta_lon_amplitude=4.0,
        max_iterations=128,
    )

    report = _assert_reference_stress_contracts(case)
    assert report["layer_mass_ratio"] <= 20.0 * case["solver"].pressure_tolerance
    assert report["half_mass_ratio"] <= 20.0 * case["solver"].pressure_tolerance
    assert _upper_half_mass is not None


def test_reference_state_finite_volume_flat_partial_bottom_stress_zero_ape():
    case = _build_flat_partial_bottom_case(pressure_tolerance=1.0e-6)
    report = _assert_reference_stress_contracts(case, compute_flat_ape=True)
    assert report["flat_ape_ratio"] <= 1.0e-15
