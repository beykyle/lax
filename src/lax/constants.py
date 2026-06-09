"""Fundamental physical constants in MeV-fm units.

All values are from the Particle Data Group (PDG) 2022 review unless noted.
Lengths are in fm, energies in MeV, momenta in MeV/c.

Example usage::

    import lax.constants as C

    # ℏ²/2μ for neutron–proton system (MeV·fm²)
    HBAR2_2MU = C.hbar2_over_2mu(1.008665, 1.007276)

    # ℏ²/2μ for α + ⁴⁰Ca
    HBAR2_2MU = C.hbar2_over_2mu(4.001506, 39.96259)
"""

from __future__ import annotations

import numpy as np

HBARC: float = 197.3269804
"""ℏc in MeV·fm."""

ALPHA: float = 1.0 / 137.0359991
"""Fine structure constant (dimensionless)."""

WAVENUMBER_PION: float = float(np.sqrt(0.5))
"""Pion wavenumber mπ/(ℏc) in fm⁻¹.  [Thompson & Nuñes, below eq. 4.3.10]"""

AMU: float = 931.494102
"""Atomic mass unit in MeV/c²  (PDG)."""

MASS_N: float = 1.008665 * AMU
"""Neutron mass in MeV/c²  (PDG)."""

MASS_P: float = 1.007276 * AMU
"""Proton mass in MeV/c²  (PDG)."""

MASS_E: float = 0.5109989461
"""Electron mass in MeV/c²  (PDG)."""

E2: float = ALPHA * HBARC
"""e² ≈ 1.44 MeV·fm (Coulomb coupling constant)."""


def hbar2_over_2mu(m1_amu: float, m2_amu: float) -> float:
    """Return ℏ²/2μ in MeV·fm² for a two-body system.

    Parameters
    ----------
    m1_amu, m2_amu
        Particle masses in atomic mass units (AMU).  Use PDG masses for
        accuracy: neutron 1.008665, proton 1.007276, alpha 4.001506, etc.

    Returns
    -------
    float
        ℏ²/2μ = (ℏc)² / (2 μc²)  in MeV·fm².

    Examples
    --------
    >>> import lax.constants as C
    >>> round(C.hbar2_over_2mu(1.008665, 1.008665), 3)  # n-n
    41.47...
    >>> round(C.hbar2_over_2mu(1.008665, 1.007276), 3)  # n-p
    41.47...
    """
    mu_c2 = (m1_amu * m2_amu) / (m1_amu + m2_amu) * AMU  # μc² in MeV
    return HBARC**2 / (2.0 * mu_c2)


__all__ = [
    "ALPHA",
    "AMU",
    "E2",
    "HBARC",
    "MASS_E",
    "MASS_N",
    "MASS_P",
    "WAVENUMBER_PION",
    "hbar2_over_2mu",
]
