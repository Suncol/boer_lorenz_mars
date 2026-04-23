"""Legacy finite-volume compatibility entrypoint for reference-state solvers.

This module preserves the historical import path used by tests and downstream
code. It re-exports the legacy parcel-sorted finite-volume implementation and
its shared solution object, but it is not the full Koehler (1986) fixed-
isentrope solver.
"""

from .finite_volume_solver import FiniteVolumeReferenceState, _solve_reference_family
from .solution import REFERENCE_INTERFACE_DIM, REFERENCE_SAMPLE_DIM, ReferenceStateSolution


KoehlerReferenceState = FiniteVolumeReferenceState


__all__ = [
    "REFERENCE_SAMPLE_DIM",
    "REFERENCE_INTERFACE_DIM",
    "ReferenceStateSolution",
    "FiniteVolumeReferenceState",
    "KoehlerReferenceState",
    "_solve_reference_family",
]
