"""Unit tests for utils.skills — skill loader and catalog renderer."""

from __future__ import annotations

import pytest

from utils.skills import load_agent_skills, render_skill_catalog


def _write_skill(root, name, description="A skill.", body="# Body\n\nDetails."):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_discovered_skills_reach_the_rendered_catalog(tmp_path):
    _write_skill(tmp_path, "beta-skill", "Second skill")
    _write_skill(tmp_path, "alpha-skill", "First skill")

    skills = load_agent_skills(tmp_path)
    assert len(skills) == 2
    assert skills[0]["skill_md"].endswith("alpha-skill/SKILL.md")

    catalog = render_skill_catalog(skills)
    assert "alpha-skill" in catalog
    assert "beta-skill" in catalog


def test_missing_skills_dir_returns_empty_rather_than_raising(tmp_path):
    assert load_agent_skills(tmp_path / "does-not-exist") == []


def test_malformed_no_frontmatter(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "SKILL.md").write_text("no frontmatter here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_agent_skills(tmp_path)


def test_build_system_includes_skill_catalog():
    from prompts.main_agent import build_system

    catalog = "Available local skills:\n- test-skill: does stuff"
    out = build_system(
        slug="s",
        goal_text="win",
        index_md="# ideas",
        writable_root="/tmp/test",
        skill_catalog=catalog,
    )
    assert "test-skill: does stuff" in out
