"""Shared test fixtures and hypothesis profile registration."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

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
