"""Unit tests for the idea-pool helpers."""

from __future__ import annotations

import pytest

from utils import idea_pool
from utils.idea_pool import add_idea, load_index, remove_idea, update_idea


def test_full_lifecycle(tmp_path):
    ideas_dir = tmp_path / "ideas"
    ideas_dir.mkdir()

    # add: monotonic ids, slugified filenames, title as H1
    assert [add_idea(ideas_dir, t, f"body-{t}") for t in ("alpha", "beta", "gamma")] == [1, 2, 3]
    assert (ideas_dir / "001_alpha.md").read_text() == "# alpha\n\nbody-alpha\n"

    # render_index wrote INDEX.md in id order
    index = load_index(ideas_dir)
    assert "- [001] alpha" in index
    assert index.index("[001]") < index.index("[002]") < index.index("[003]")

    # update: title stays, body replaced
    update_idea(ideas_dir, 1, "new body")
    assert (ideas_dir / "001_alpha.md").read_text() == "# alpha\n\nnew body\n"

    # remove: file gone, index updated, subsequent add takes max+1 (no id reuse)
    remove_idea(ideas_dir, 2)
    assert not (ideas_dir / "002_beta.md").exists()
    assert "beta" not in load_index(ideas_dir)
    assert add_idea(ideas_dir, "delta", "d") == 4

    # missing id on remove/update raises ValueError (unpacking zero-length glob)
    with pytest.raises(ValueError):
        remove_idea(ideas_dir, 99)
    with pytest.raises(ValueError):
        update_idea(ideas_dir, 99, "nope")


def test_load_index_truncation_warning(monkeypatch, tmp_path):
    ideas_dir = tmp_path / "ideas"
    ideas_dir.mkdir()
    monkeypatch.setattr(idea_pool, "MAX_INDEX_LINES", 5)

    for i in range(10):
        add_idea(ideas_dir, f"idea-{i}", "body")

    loaded = load_index(ideas_dir)
    assert "WARNING" in loaded
    assert loaded.rstrip().endswith("Prune the pool.")
