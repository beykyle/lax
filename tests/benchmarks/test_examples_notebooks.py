from __future__ import annotations

import warnings
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient
MissingIDFieldWarning = nbformat.validator.MissingIDFieldWarning
NotebookNode = nbformat.NotebookNode

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_NOTEBOOKS = sorted((REPO_ROOT / "examples").glob("*.ipynb"))
SKIP_BENCHMARK_TAG = "skip-benchmark"


@pytest.mark.benchmark
@pytest.mark.parametrize("notebook_path", EXAMPLE_NOTEBOOKS, ids=lambda path: path.name)
def test_example_notebooks_execute(notebook_path: Path) -> None:
    """All example notebooks execute successfully under nbclient."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
    notebook = _without_skipped_benchmark_cells(notebook)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def _without_skipped_benchmark_cells(notebook: NotebookNode) -> NotebookNode:
    """Drop notebook cells tagged to be excluded from benchmark execution."""

    notebook.cells = [
        cell
        for cell in notebook.cells
        if SKIP_BENCHMARK_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    return notebook
