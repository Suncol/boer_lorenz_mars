"""Mars physical constants used by the exact Lorenz/Boer branch.

The values below follow the dry-CO2 Mars PCM / MCD-style constants used in the
LMD Mars modeling stack. In particular, they match the Mars branch of
``module_model_constants.F`` in the LMD/Planeto Mars PCM interfaces:

    g       = 3.72 m s-2
    r_d     = 191.0 J kg-1 K-1
    cp      = 744.5 J kg-1 K-1
    p0      = 610.0 Pa
    reradius= 3397200.0 m
    EOMEG   = 7.0721e-5 s-1

These constants are intentionally centralized so that the exact Boer branch and
any future SEBA Mars adapter can share one authoritative source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarsConstants:
    """Read-only container for Mars physical constants."""

    a: float
    g: float
    Omega: float
    Rd: float
    cp: float
    p00: float

    @property
    def kappa(self) -> float:
        return self.Rd / self.cp


MARS = MarsConstants(
    a=3_397_200.0,
    g=3.72,
    Omega=7.0721e-5,
    Rd=191.0,
    cp=744.5,
    p00=610.0,
)

a = MARS.a
g = MARS.g
Omega = MARS.Omega
Rd = MARS.Rd
cp = MARS.cp
kappa = MARS.kappa
p00 = MARS.p00

__all__ = ["MarsConstants", "MARS", "a", "g", "Omega", "Rd", "cp", "kappa", "p00"]
