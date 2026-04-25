"""Smoke tests — every Streamlit page renders without uncaught exceptions.

Uses Streamlit's official AppTest framework, which simulates a full page
render in-process (no browser needed). If a page top-level raises, this
test catches it.

These tests don't depend on backend credentials or generated data — they
just verify the import + initial-render path is clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    Path("app/Home.py"),
    Path("app/pages/2_Gallery.py"),
    Path("app/pages/3_Strategy_Lab.py"),
    Path("app/pages/4_Training_Studio.py"),
    Path("app/pages/5_Experiments.py"),
    Path("app/pages/6_About.py"),
]


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each smoke test gets a private DuckDB + data dir.

    Without this, the test would try to open `data/runs/runs.duckdb` — which
    is locked whenever a real Streamlit dev server is also running. DuckDB is
    single-writer per file, so the test would 'IO Error: Conflicting lock'.
    Also resets the DB singleton so the override actually takes effect.
    """
    from image_generator import db as db_pkg
    from image_generator.config import settings

    monkeypatch.setattr(settings, "imggen_data_dir", tmp_path)
    monkeypatch.setattr(settings, "imggen_db_path", tmp_path / "runs.duckdb")
    monkeypatch.setattr(db_pkg.connection, "_database_singleton", None)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders(page: Path) -> None:
    """Run the page top-down and assert no uncaught exception was raised.

    `at.exception` is Streamlit's collected exception list; empty = clean run.
    """
    at = AppTest.from_file(str(page), default_timeout=30)
    at.run()
    if at.exception:
        details = "\n".join(str(e.value) for e in at.exception)
        pytest.fail(f"Page {page} raised during initial render:\n{details}")
