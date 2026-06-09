"""Sphinx configuration for lax."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# pypandoc-binary installs pandoc into pypandoc/files/, not into PATH.
# nbconvert discovers pandoc via shutil.which(), so we prepend that directory here.
try:
    import pypandoc as _pypandoc

    _pandoc_dir = os.path.join(os.path.dirname(os.path.realpath(_pypandoc.__file__)), "files")
    os.environ["PATH"] = _pandoc_dir + os.pathsep + os.environ.get("PATH", "")
    del _pypandoc, _pandoc_dir
except ImportError:
    pass

# Ensure the installed package is importable (needed for autodoc).
# When running via `uv run sphinx-build`, lax is already installed in the env.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Copy example notebooks into docs/notebooks/ (a real directory, not a symlink).
# A real directory keeps path resolution within docs/ so Sphinx can locate
# notebook output images without following symlinks into the project root.
_notebooks_dir = Path(__file__).parent / "notebooks"
_examples_dir = Path(__file__).parent.parent / "examples"
if _notebooks_dir.is_symlink():
    _notebooks_dir.unlink()
_notebooks_dir.mkdir(exist_ok=True)
for _nb in _examples_dir.glob("*.ipynb"):
    shutil.copy2(_nb, _notebooks_dir / _nb.name)

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "lax"
copyright = "2024, Kyle Beyer"
author = "Kyle Beyer"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "nbsphinx",
]

# ---------------------------------------------------------------------------
# Source / output
# ---------------------------------------------------------------------------

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# lax re-exports types from internal submodules (e.g. lax.Solver is defined in
# lax.boundary._types).  Sphinx sees these as duplicate Python domain objects;
# suppress the warning because the duplication is intentional.
suppress_warnings = ["py.duplicate"]

# ---------------------------------------------------------------------------
# HTML output — Furo theme
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = "lax"
html_theme_options = {
    "source_repository": "https://github.com/beykyle/lax",
    "source_branch": "main",
    "source_directory": "docs/",
}

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# ---------------------------------------------------------------------------
# napoleon (numpy-style docstrings)
# ---------------------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
# Sphinx 9 autodoc auto-documents @dataclass fields from annotations.
# napoleon_use_ivar=True makes napoleon emit :ivar: fields instead of
# .. py:attribute:: directives, avoiding duplicate domain entries.
napoleon_use_ivar = True

# ---------------------------------------------------------------------------
# intersphinx — cross-reference to upstream docs
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "mpmath": ("https://mpmath.org/doc/current", None),
}

# ---------------------------------------------------------------------------
# nbsphinx — Jupyter notebook rendering
# ---------------------------------------------------------------------------

# Render committed notebook outputs without re-executing.
# Execution correctness is verified separately by tests/benchmarks/test_examples_notebooks.py.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
