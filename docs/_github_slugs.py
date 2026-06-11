"""GitHub-style heading slugs for MyST (importable for Sphinx env pickling).

DESIGN.md's table of contents uses GitHub anchor links such as
``#1-overview-and-scope``; MyST's default slugger drops the leading digits,
so we mirror GitHub's rule: lowercase, strip punctuation, spaces → dashes.
"""

from __future__ import annotations

import re


def github_heading_slug(title: str) -> str:
    """Return the GitHub anchor slug for one heading title."""

    slug = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return slug.replace(" ", "-")
