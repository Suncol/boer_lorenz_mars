"""Mars exact Lorenz energy-cycle diagnostics.

Phase 1 provides the shared topography-aware foundation used by the exact
Lorenz/Boer branch: Mars constants, pressure-mask helpers, geometric weights,
mass integrators, representative zonal operators, and the first kinetic-energy
reservoir/conversion terms that do not depend on a reference state.
"""

from .constants_mars import MARS, MarsConstants, Omega, Rd, a, cp, g, kappa, p00

__all__ = [
    "MARS",
    "MarsConstants",
    "a",
    "g",
    "Omega",
    "Rd",
    "cp",
    "kappa",
    "p00",
]
