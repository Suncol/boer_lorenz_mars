"""Legacy finite-volume compatibility entrypoint for reference-state solvers.

This module preserves the historical import path used by tests and downstream
code. It re-exports the legacy parcel-sorted finite-volume implementation and
its shared solution object, but it is not the full Koehler (1986) fixed-
isentrope solver.
"""

import warnings

from .finite_volume_solver import FiniteVolumeReferenceState, _solve_reference_family
from .solution import REFERENCE_INTERFACE_DIM, REFERENCE_SAMPLE_DIM, ReferenceStateSolution


def __getattr__(name: str):
    if name == "KoehlerReferenceState":
        warnings.warn(
            "'KoehlerReferenceState' is a deprecated legacy alias for "
            "'FiniteVolumeReferenceState'. Use 'Koehler1986ReferenceState' for the "
            "fixed-isentrope Koehler (1986) solver.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FiniteVolumeReferenceState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REFERENCE_SAMPLE_DIM",
    "REFERENCE_INTERFACE_DIM",
    "ReferenceStateSolution",
    "FiniteVolumeReferenceState",
    "_solve_reference_family",
]
