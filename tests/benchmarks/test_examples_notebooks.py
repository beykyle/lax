from __future__ import annotations

import warnings
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient
MissingIDFieldWarning = nbformat.validator.MissingIDFieldWarning

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_NOTEBOOKS = sorted((REPO_ROOT / "examples").glob("*.ipynb"))


@pytest.mark.benchmark
@pytest.mark.parametrize("notebook_path", EXAMPLE_NOTEBOOKS, ids=lambda path: path.name)
def test_example_notebooks_execute(notebook_path: Path) -> None:
    """All example notebooks execute successfully under nbclient."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
