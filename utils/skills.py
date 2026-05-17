from __future__ import annotations

from pathlib import Path

import yaml


def load_agent_skills(root: str | Path = ".agents/skills") -> list[dict]:
    skills = []

    for skill_md in sorted(Path(root).glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        _, frontmatter, body = text.split("---", 2)
        meta = yaml.safe_load(frontmatter)

        skills.append(
            {
                "name": meta["name"],
                "description": meta["description"].strip(),
                "path": str(skill_md.parent.resolve()),
                "skill_md": str(skill_md.resolve()),
                "body": body.strip(),
            }
        )

    return skills


def render_skill_catalog(skills: list[dict]) -> str:
    if not skills:
        return ""
    lines = ["Available local skills:"]
    for s in skills:
        lines.append(f"- {s['name']}: {s['description']}")
        lines.append(f"  SKILL.md: {s['skill_md']}")
    lines.append("")
    lines.append("Skill rules:")
    lines.append("- If the user names a skill, use it.")
    lines.append("- If the task clearly matches a skill description, use it.")
    lines.append("- Before using a skill, read its full SKILL.md from the listed path.")
    lines.append("- Follow only the minimal set of skills needed for the task.")
    lines.append(
        "- Resolve referenced scripts, assets, and helper files relative to the skill directory."
    )
    lines.append(
        "- Do not carry a skill across turns unless the current task still matches it."
    )
    return "\n".join(lines)
