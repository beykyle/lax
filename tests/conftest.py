"""Shared test fixtures and hypothesis profile registration."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

import lax.constants

# Register the CI hypothesis profile.  GitHub Actions sets CI=true which makes
# hypothesis look for this profile; defining it explicitly here ensures the
# settings are applied consistently across all CI runs.
settings.register_profile(
    "ci",
    database=None,
    deadline=None,
    print_blob=True,
    derandomize=True,
    suppress_health_check=(HealthCheck.too_slow,),
)

# Conventional rounded e² (MeV·fm) used by the published Descouvemont / optical-model
# benchmark references.  The library default (lax.constants.E2 = ALPHA*HBARC ≈ 1.43996)
# is the exact physical value; only the benchmarks that compare against 1.44-prepared
# data override it via the legacy_coulomb_constant fixture below.
LEGACY_COULOMB_E2 = 1.44


@pytest.fixture
def legacy_coulomb_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the Coulomb constant to the rounded 1.44 of the benchmark references.

    Both library Coulomb sites (``boundary.coulomb._sommerfeld`` and
    ``models.optical.uniform_sphere_coulomb_potential``) read ``lax.constants.E2`` at
    call time, so this single patch covers the boundary Sommerfeld parameter and the
    rotor-model Coulomb potential for the duration of a test.
    """

    monkeypatch.setattr(lax.constants, "E2", LEGACY_COULOMB_E2)
