import base64
from pathlib import Path

import pytest
import sys

# Ensure project root is importable when running tests from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_min_png(path: Path) -> None:
    # 1x1 transparent PNG
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)


def test_ingest_new_media_appends_multimodal_message(tmp_path, monkeypatch):
    # Arrange: create agent bound to a temp slug under real task root
    from agents.researcher import ResearcherAgent
    import uuid

    # Use unique slug to avoid test pollution
    slug = f"test-media-ingest-{uuid.uuid4().hex[:8]}"
    base_dir = Path("task") / slug
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create description.md file required by ResearcherAgent
    (base_dir / "description.md").write_text("Test competition description")

    agent = ResearcherAgent(slug=slug, iteration=1)

    # Use the agent's media_dir (which includes run_id)
    media_dir = agent.media_dir

    before = agent._list_media_files()
    img_path = media_dir / "chart_1.png"
    _write_min_png(img_path)

    # Act: ingest the newly created media
    messages = agent._ingest_new_media(before)

    # Assert: a message was returned with correct structure (Gemini format)
    assert messages is not None
    assert len(messages) == 1
    m = messages[0]
    assert m.get("role") == "user"
    parts = m.get("parts")
    assert isinstance(parts, list) and len(parts) >= 1
    # At least one Part with inline_data present (Gemini format)
    assert any(hasattr(p, "inline_data") and p.inline_data for p in parts)


def test_ingest_media_respects_max_images(tmp_path):
    from agents.researcher import ResearcherAgent, MAX_IMAGES_PER_STEP
    import uuid

    # Use unique slug to avoid test pollution
    slug = f"test-media-cap-{uuid.uuid4().hex[:8]}"
    base_dir = Path("task") / slug
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create description.md file required by ResearcherAgent
    (base_dir / "description.md").write_text("Test competition description")

    agent = ResearcherAgent(slug=slug, iteration=1)

    # Use the agent's media_dir (which includes run_id)
    media_dir = agent.media_dir

    before = agent._list_media_files()
    # Create more images than the cap
    for i in range(MAX_IMAGES_PER_STEP + 3):
        _write_min_png(media_dir / f"img_{i}.png")

    messages = agent._ingest_new_media(before)

    assert messages is not None
    assert len(messages) == 1
    parts = messages[0]["parts"]
    attached = [p for p in parts if hasattr(p, "inline_data") and p.inline_data]
    assert len(attached) <= MAX_IMAGES_PER_STEP


