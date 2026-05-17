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


def test_load_two_skills_sorted(tmp_path):
    _write_skill(tmp_path, "beta-skill", "Second skill")
    _write_skill(tmp_path, "alpha-skill", "First skill")

    skills = load_agent_skills(tmp_path)
    assert len(skills) == 2
    assert skills[0]["name"] == "alpha-skill"
    assert skills[1]["name"] == "beta-skill"
    assert skills[0]["description"] == "First skill"
    assert skills[0]["path"].endswith("alpha-skill")
    assert skills[0]["skill_md"].endswith("SKILL.md")
    assert "Body" in skills[0]["body"]


def test_load_empty_directory(tmp_path):
    assert load_agent_skills(tmp_path) == []


def test_malformed_no_frontmatter(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "SKILL.md").write_text("no frontmatter here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_agent_skills(tmp_path)


def test_missing_name_key(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\ndescription: oops\n---\n\nbody\n", encoding="utf-8"
    )
    with pytest.raises(KeyError):
        load_agent_skills(tmp_path)


def test_render_catalog_includes_names_and_rules(tmp_path):
    _write_skill(tmp_path, "my-skill", "Does things")
    skills = load_agent_skills(tmp_path)
    catalog = render_skill_catalog(skills)

    assert "my-skill" in catalog
    assert "Does things" in catalog
    assert "SKILL.md" in catalog
    assert "Skill rules:" in catalog
    assert "read its full SKILL.md" in catalog


def test_render_catalog_empty():
    assert render_skill_catalog([]) == ""


def test_live_agents_directory():
    skills = load_agent_skills()
    names = {s["name"] for s in skills}
    assert "neurogolf-submit" in names
    assert "neurogolf-sweep" in names
    assert "neurogolf-sweep-forward" in names
    assert "neurogolf-qgentic-sweep" in names
    assert len(skills) >= 4


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
