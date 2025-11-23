import os
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


class _StubMsg:
    def __init__(self, content: str):
        self.content = content


class _StubCompletion:
    def __init__(self, content: str):
        self.choices = [type("Ch", (), {"message": _StubMsg(content)})()]
        self.output_text = content
        self.output = [{"role": "assistant", "content": content}]


def test_ask_eda_sets_media_dir_and_code_can_write(monkeypatch, tmp_path):
    # Arrange: patch call_llm_with_retry to return trivial python that prints MEDIA_DIR and writes a file
    import tools.researcher as tr

    code = (
        """
```python
import os
from pathlib import Path
media = os.environ.get("MEDIA_DIR")
print("MEDIA_DIR:", media)
f = Path(media) / "unit_test_plot.png"
Path(media).mkdir(parents=True, exist_ok=True)
open(f, "wb").write(b"PNG")
print("Chart successfully saved to:", str(f.resolve()))
```
        """
    )

    def _fake_call_llm_with_retry(model=None, instructions=None, tools=None, messages=None, **kwargs):
        return _StubCompletion(code)

    # Force OpenAI provider to avoid real API calls
    monkeypatch.setattr(tr, "detect_provider", lambda x: "openai")
    monkeypatch.setattr(tr, "call_llm_with_retry", _fake_call_llm_with_retry)

    # Act: run ask_eda pointing to a temporary data path
    out = tr.ask_eda(
        question="Generate a tiny chart",
        description="desc",
        data_path=str(tmp_path),
    )

    # Assert: MEDIA_DIR is set and file path echoed
    assert "MEDIA_DIR:" in out or "Chart successfully saved" in out
    media_dir = os.environ.get("MEDIA_DIR")
    assert media_dir is not None
    assert Path(media_dir).exists()


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

    # Assert: a message was returned with correct structure
    assert messages is not None
    assert len(messages) == 1
    m = messages[0]
    assert m.get("role") == "user"
    content = m.get("content")
    assert isinstance(content, list) and len(content) >= 1
    # At least one input_image item present (Responses API format)
    assert any(item.get("type") == "input_image" for item in content)


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
    content = messages[0]["content"]
    attached = [c for c in content if c.get("type") == "input_image"]
    assert len(attached) <= MAX_IMAGES_PER_STEP


