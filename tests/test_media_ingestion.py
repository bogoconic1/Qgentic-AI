import base64
from pathlib import Path

import pytest

from agents.researcher import ResearcherAgent, MAX_IMAGES_PER_STEP


def _write_min_png(path: Path) -> None:
    """Write a 1x1 transparent PNG."""
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)


def test_ingest_new_media_appends_multimodal_message(tmp_path, monkeypatch):
    """Test that new media files are ingested as multimodal messages."""
    import uuid

    slug = f"test-media-ingest-{uuid.uuid4().hex[:8]}"
    base_dir = Path("task") / slug
    base_dir.mkdir(parents=True, exist_ok=True)

    (base_dir / "description.md").write_text("Test competition description")

    agent = ResearcherAgent(slug=slug, iteration=1)

    media_dir = agent.media_dir

    before = agent._list_media_files()
    img_path = media_dir / "chart_1.png"
    _write_min_png(img_path)

    messages = agent._ingest_new_media(before)

    assert messages is not None
    assert len(messages) == 1
    m = messages[0]
    assert m.get("role") == "user"
    parts = m.get("parts")
    assert isinstance(parts, list) and len(parts) >= 1
    assert any(hasattr(p, "inline_data") and p.inline_data for p in parts)


def test_ingest_media_respects_max_images(tmp_path):
    """Test that media ingestion respects MAX_IMAGES_PER_STEP cap."""
    import uuid

    slug = f"test-media-cap-{uuid.uuid4().hex[:8]}"
    base_dir = Path("task") / slug
    base_dir.mkdir(parents=True, exist_ok=True)

    (base_dir / "description.md").write_text("Test competition description")

    agent = ResearcherAgent(slug=slug, iteration=1)

    media_dir = agent.media_dir

    before = agent._list_media_files()
    for i in range(MAX_IMAGES_PER_STEP + 3):
        _write_min_png(media_dir / f"img_{i}.png")

    messages = agent._ingest_new_media(before)

    assert messages is not None
    assert len(messages) == 1
    parts = messages[0]["parts"]
    attached = [p for p in parts if hasattr(p, "inline_data") and p.inline_data]
    assert len(attached) <= MAX_IMAGES_PER_STEP
