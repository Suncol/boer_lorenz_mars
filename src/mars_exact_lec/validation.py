"""Validation helpers for live SEBA spectral component cross-checks.

These helpers expose SEBA spectral energy components on pressure levels for
cross-validation. They intentionally do not claim that SEBA APE/VKE/RKE/DKE are
the same mathematical objects as Boer exact four-reservoir diagnostics.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import xarray as xr

from .constants_mars import MARS

_REPO_SRC = Path(__file__).resolve().parents[1]


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_repo_src_path(entry: str) -> bool:
    try:
        return Path(entry).resolve() == _REPO_SRC
    except (OSError, RuntimeError):
        return False


def _purge_seba_modules() -> dict[str, object]:
    removed: dict[str, object] = {}
    for name in list(sys.modules):
        if name == "seba" or name.startswith("seba."):
            removed[name] = sys.modules.pop(name)
    return removed


def _classify_live_seba_import_error(exc: BaseException) -> str | None:
    for error in _iter_exception_chain(exc):
        if isinstance(error, ModuleNotFoundError):
            missing = getattr(error, "name", "") or ""
            if missing in {"seba", "seba.seba"}:
                return (
                    "Installed SEBA runtime not found. Install this repository into the active "
                    "environment before running live SEBA cross-validation."
                )
            if missing in {"seba.numeric_tools", "numeric_tools"}:
                return (
                    "Installed SEBA runtime is missing its compiled numeric_tools extension. "
                    "Reinstall SEBA in the active environment so the f2py extension is built."
                )
            if missing in {"shtns", "_shtns"}:
                return (
                    "Installed SEBA runtime is missing shtns. Install FFTW first, then install "
                    "shtns into the active environment before running live cross-validation."
                )

        if isinstance(error, ImportError):
            message = str(error)
            lower = message.lower()

            if "numeric_tools" in message:
                return (
                    "Installed SEBA runtime could not import its numeric_tools extension. "
                    "Reinstall SEBA in the active environment so the compiled backend is present."
                )
            if "shtns" in lower or "_shtns" in lower:
                return (
                    "Installed SEBA runtime could not import shtns. Install FFTW first, then "
                    "install shtns into the active environment before running live cross-validation."
                )
            if any(token in lower for token in ("fftw", "libfftw", "fftw3")):
                return (
                    "Installed SEBA runtime could not load FFTW-backed dependencies. Ensure FFTW "
                    "is installed and discoverable before installing shtns."
                )
            if any(token in lower for token in ("libgomp", "libomp", "openmp", "-lgomp")):
                return (
                    "Installed SEBA runtime could not load OpenMP-compatible compiled "
                    "extensions. Install an OpenMP-capable toolchain or rebuild SEBA with "
                    "openmp disabled."
                )
            if any(token in lower for token in ("library not loaded", "image not found", "dlopen")):
                return (
                    "Installed SEBA runtime failed to load one of its compiled dependencies. "
                    "Reinstall shtns and the numeric_tools extension after fixing FFTW/OpenMP "
                    "library discovery."
                )

    return None


def _resolve_live_seba_energy_budget():
    """Import ``seba.seba.EnergyBudget`` from the installed runtime, not repo ``src``."""

    original_path = list(sys.path)
    filtered_path = [entry for entry in original_path if not _is_repo_src_path(entry)]
    removed_modules = _purge_seba_modules()
    imported = False

    try:
        importlib.invalidate_caches()
        sys.path[:] = filtered_path
        module = importlib.import_module("seba.seba")
        imported = True
        return module.EnergyBudget
    except Exception as exc:
        message = _classify_live_seba_import_error(exc)
        if message is not None:
            raise ImportError(message) from exc
        raise
    finally:
        sys.path[:] = original_path
        if not imported:
            sys.modules.update(removed_modules)


def seba_energy_components_per_level(
    dataset: xr.Dataset,
    *,
    p_levels,
    ps: xr.DataArray,
    variables: dict[str, str] | None = None,
    truncation: int | None = None,
    rsphere: float = MARS.a,
) -> xr.Dataset:
    """Return SEBA RKE/DKE/HKE/VKE/APE per level by summing degree spectra.

    These are SEBA spectral components for validation and regression checks.
    They are not Boer exact reservoir terms and should not be interpreted as a
    replacement for the exact topographic Lorenz/Boer cycle.
    """

    EnergyBudget = _resolve_live_seba_energy_budget()

    budget = EnergyBudget(
        dataset,
        variables=variables,
        p_levels=p_levels,
        ps=ps,
        truncation=truncation,
        rsphere=rsphere,
    )
    rke, dke, hke = budget.horizontal_kinetic_energy()
    components = {
        "rke": rke,
        "dke": dke,
        "hke": hke,
        "vke": budget.vertical_kinetic_energy(),
        "ape": budget.available_potential_energy(),
    }
    data_vars = {}
    for component_name, spectrum in components.items():
        per_level = spectrum.sum(dim="kappa")
        per_level.name = f"seba_{component_name}_per_level"
        per_level.attrs["units"] = spectrum.attrs.get("units", "m**2 s**-2")
        per_level.attrs["validation_role"] = "live_seba_spectral_component"
        per_level.attrs["component"] = component_name
        data_vars[per_level.name] = per_level
    result = xr.Dataset(data_vars=data_vars)
    result.attrs["validation_role"] = "live_seba_spectral_components_per_level"
    result.attrs["boer_exact_equivalence"] = "none"
    return result


def seba_total_hke_per_level(
    dataset: xr.Dataset,
    *,
    p_levels,
    ps: xr.DataArray,
    variables: dict[str, str] | None = None,
    truncation: int | None = None,
    rsphere: float = MARS.a,
) -> xr.DataArray:
    """Return SEBA total horizontal KE per level by summing the degree spectrum."""

    components = seba_energy_components_per_level(
        dataset,
        p_levels=p_levels,
        ps=ps,
        variables=variables,
        truncation=truncation,
        rsphere=rsphere,
    )
    total = components["seba_hke_per_level"].copy(deep=False)
    total.name = "seba_total_hke_per_level"
    return total


__all__ = ["seba_energy_components_per_level", "seba_total_hke_per_level"]
