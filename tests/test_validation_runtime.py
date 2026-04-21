from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import mars_exact_lec.validation as validation


REPO_SRC = (Path(__file__).resolve().parents[1] / "src").resolve()


def test_live_resolver_drops_repo_src_from_sys_path(monkeypatch):
    observed: dict[str, list[str]] = {}

    fake_module = types.SimpleNamespace(EnergyBudget=type("EnergyBudget", (), {}))

    def fake_import_module(name: str):
        observed["name"] = name
        observed["path"] = list(sys.path)
        return fake_module

    original_path = list(sys.path)
    monkeypatch.setattr(validation.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sys, "path", [str(REPO_SRC), *original_path])

    energy_budget = validation._resolve_live_seba_energy_budget()

    assert energy_budget is fake_module.EnergyBudget
    assert observed["name"] == "seba.seba"
    assert str(REPO_SRC) not in observed["path"]
    assert sys.path[0] == str(REPO_SRC)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ModuleNotFoundError("No module named 'seba.numeric_tools'", name="seba.numeric_tools"),
            "numeric_tools extension",
        ),
        (
            ModuleNotFoundError("No module named 'shtns'", name="shtns"),
            "missing shtns",
        ),
    ],
)
def test_live_resolver_reports_environment_blockers(monkeypatch, error, expected):
    def fake_import_module(name: str):
        raise error

    monkeypatch.setattr(validation.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=expected):
        validation._resolve_live_seba_energy_budget()
