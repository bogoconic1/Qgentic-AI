"""Route-level tests for scripts.viewer.server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.viewer import server


def _seed_run(root: Path) -> tuple[str, str]:
    slug, run_id = "demo", "20260101_000000"
    run = root / slug / run_id
    run.mkdir(parents=True)
    (run / "MAIN.md").write_text("# Plan\n\nhello world.")
    (root / slug / "GOAL.md").write_text("# goal")
    (run / "main_agent_chat.jsonl").write_text(
        "\n".join([
            json.dumps({
                "role": "assistant",
                "content": {"parts": [{"text": "hi from MainAgent"}]},
                "ts": "2026-01-01T00:00:00+00:00",
            }),
            json.dumps({
                "role": "tool",
                "name": "read_file",
                "args": {"path": "/x"},
                "result": "{\"content\": \"ok\"}",
                "ts": "2026-01-01T00:00:01+00:00",
            }),
        ]) + "\n"
    )

    research = run / "research_1"
    research.mkdir()
    (research / "RESEARCH.md").write_text("# research")
    (research / "web_research").mkdir()
    (research / "web_research" / "001.md").write_text("# web result 1")
    return slug, run_id


@pytest.fixture
def client(tmp_path: Path):
    _seed_run(tmp_path)
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def test_index_lists_slug(client):
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"demo" in rv.data


def test_slug_lists_run(client):
    rv = client.get("/task/demo")
    assert rv.status_code == 200
    assert b"20260101_000000" in rv.data
    # GOAL.md link should appear since it's at slug root
    assert b"GOAL.md" in rv.data


def test_slug_404_unknown(client):
    assert client.get("/task/ghost").status_code == 404


def test_run_renders_main_transcript(client):
    rv = client.get("/task/demo/20260101_000000")
    assert rv.status_code == 200
    assert b"hi from MainAgent" in rv.data
    assert b"read_file" in rv.data
    assert b"research_1" in rv.data


def test_bash_command_rendered_directly(tmp_path: Path):
    slug, run_id = "bashy", "20260102_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    cmd = 'grep -n -A 20 "class Node" /venv/main/lib/python3.12/site-packages/onnx_tool/node.py'
    (run / "main_agent_chat.jsonl").write_text(
        "\n".join([
            json.dumps({
                "role": "assistant",
                "content": {"parts": [
                    {"function_call": {"id": "c1", "name": "bash", "args": {"command": cmd}}}
                ]},
                "ts": "2026-01-02T00:00:00+00:00",
            }),
            json.dumps({
                "role": "tool",
                "name": "bash",
                "args": {"command": cmd},
                "result": "ok",
                "ts": "2026-01-02T00:00:01+00:00",
            }),
        ]) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    # The command renders once, inside the function_call's <pre class="bash-cmd">.
    # Quotes get HTML-escaped to &#34; (the browser shows them as ").
    assert "bash-cmd" in body
    assert "&#34;class Node&#34;" in body
    # No JSON-wrapped form: no `"command":` key, no JSON-escaped \" sequence.
    assert '"command":' not in body
    assert '\\"class Node\\"' not in body
    # Tool record no longer repeats the command — bash-cmd appears exactly once.
    assert body.count('class="bash-cmd"') == 1


def test_tool_result_unwraps_bash_output(tmp_path: Path):
    slug, run_id = "wrap", "20260103_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    payload = json.dumps({"output": "hello\nworld", "returncode": 0, "truncated": False})
    (run / "main_agent_chat.jsonl").write_text(
        json.dumps({
            "role": "tool",
            "name": "bash",
            "args": {"command": "echo hi"},
            "result": payload,
            "ts": "2026-01-03T00:00:00+00:00",
        }) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    # Primary text shows up directly, not as JSON-escaped \"output\": \"hello\\nworld\".
    assert "hello\nworld" in body
    assert '\\"output\\":' not in body
    # Metadata shown as key=value pairs.
    assert "returncode" in body and "truncated" in body
    # The wrapping JSON braces don't show up in the rendered body anymore.
    assert '"output":' not in body


def test_tool_result_error_styled(tmp_path: Path):
    slug, run_id = "errfile", "20260104_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    (run / "main_agent_chat.jsonl").write_text(
        json.dumps({
            "role": "tool",
            "name": "write_file",
            "args": {"path": "/tmp/x"},
            "result": json.dumps({"error": "Blocked by safety judge"}),
            "ts": "2026-01-04T00:00:00+00:00",
        }) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    assert "Blocked by safety judge" in body
    assert "result-error" in body


def test_tool_result_no_primary_renders_meta_only(tmp_path: Path):
    slug, run_id = "wrap2", "20260105_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    (run / "main_agent_chat.jsonl").write_text(
        json.dumps({
            "role": "tool",
            "name": "add_idea",
            "args": {"body": "x"},
            "result": json.dumps({"idea_id": 1}),
            "ts": "2026-01-05T00:00:00+00:00",
        }) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    # No textual primary key → meta pills only, no body <pre>.
    assert "meta-row" in body
    assert "idea_id" in body
    # Result body is empty so no result-* <pre> in the result section.
    result_section = body.split('<div class="result">', 1)[1].split("</div>", 1)[0]
    assert "<pre" not in result_section
    # Sanity: no leftover JSON-fallback class.
    assert "result-json" not in body


def test_tool_result_list_dir_entries_unwrapped(tmp_path: Path):
    slug, run_id = "lst", "20260106_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    payload = json.dumps({
        "entries": ["a.py", "b.py", "README.md"],
        "showing": 3,
        "total": 3,
        "truncated": False,
    })
    (run / "main_agent_chat.jsonl").write_text(
        json.dumps({
            "role": "tool",
            "name": "list_dir",
            "args": {"path": "."},
            "result": payload,
            "ts": "2026-01-06T00:00:00+00:00",
        }) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    assert "a.py\nb.py\nREADME.md" in body
    # Surrounding fields appear in meta row, not body.
    assert "showing" in body and "total" in body and "truncated" in body
    # No raw JSON wrapper.
    assert '"entries":' not in body


def test_tool_result_grep_matches_with_dict_items(tmp_path: Path):
    slug, run_id = "grp", "20260107_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    payload = json.dumps({
        "matches": [{"file": "x.py", "line": 1, "text": "hi"}],
        "showing": 1,
        "total_matches": 1,
    })
    (run / "main_agent_chat.jsonl").write_text(
        json.dumps({
            "role": "tool",
            "name": "grep_code",
            "args": {"pattern": "hi"},
            "result": payload,
            "ts": "2026-01-07T00:00:00+00:00",
        }) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    # Each match item rendered as JSON on its own line in the body block.
    assert '{&#34;file&#34;: &#34;x.py&#34;, &#34;line&#34;: 1, &#34;text&#34;: &#34;hi&#34;}' in body
    assert "total_matches" in body and "showing" in body


def test_tool_result_run_solution_output_tail_unwrapped(tmp_path: Path):
    slug, run_id = "rs", "20260108_000000"
    run = tmp_path / slug / run_id
    run.mkdir(parents=True)
    payload = json.dumps({
        "elapsed_seconds": 132.5,
        "output_tail": "Final score: 0.87",
        "score": 0.87,
        "stats": {"train_loss": 0.1, "val_score": 0.87},
        "success": True,
    })
    (run / "main_agent_chat.jsonl").write_text(
        json.dumps({
            "role": "tool",
            "name": "run_solution",
            "args": {},
            "result": payload,
            "ts": "2026-01-08T00:00:00+00:00",
        }) + "\n"
    )
    app = server.create_app(task_root_override=tmp_path)
    app.config["TESTING"] = True
    rv = app.test_client().get(f"/task/{slug}/{run_id}")
    assert rv.status_code == 200
    body = rv.data.decode()
    assert "Final score: 0.87" in body
    # The other fields appear as meta pills (not the body <pre>).
    assert "elapsed_seconds" in body
    assert "score" in body
    assert "stats" in body
    assert "success" in body
    # No raw JSON wrapper of the full record.
    assert '"output_tail":' not in body


def test_run_404_missing(client):
    assert client.get("/task/demo/19990101_000000").status_code == 404


def test_research_subagent(client):
    rv = client.get("/task/demo/20260101_000000/research_1")
    assert rv.status_code == 200
    assert b"RESEARCH.md" in rv.data
    assert b"web_research/001.md" in rv.data


def test_file_renders_markdown(client):
    rv = client.get("/file?path=demo/20260101_000000/MAIN.md")
    assert rv.status_code == 200
    # markdown converted to <h1>
    assert b"<h1>Plan</h1>" in rv.data


def test_file_path_traversal_400(client):
    rv = client.get("/file?path=../../etc/passwd")
    assert rv.status_code == 400


def test_file_missing_404(client):
    rv = client.get("/file?path=demo/20260101_000000/nope.md")
    assert rv.status_code == 404


def test_file_unknown_extension_415(client, tmp_path: Path):
    # Drop a .bin file into the seeded tree under the task root.
    (tmp_path / "demo" / "20260101_000000" / "blob.bin").write_text("x")
    rv = client.get("/file?path=demo/20260101_000000/blob.bin")
    assert rv.status_code == 415


def test_file_size_limit_413(client, tmp_path: Path):
    big = tmp_path / "demo" / "20260101_000000" / "big.txt"
    big.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    rv = client.get("/file?path=demo/20260101_000000/big.txt")
    assert rv.status_code == 413


def test_file_empty_path_400(client):
    rv = client.get("/file?path=")
    assert rv.status_code == 400


def test_export_self_contained(client, tmp_path: Path):
    research_log = tmp_path / "demo" / "20260101_000000" / "research_1" / "researcher_chat.jsonl"
    research_log.write_text(json.dumps({
        "role": "assistant",
        "content": {"parts": [{"text": "hi from researcher"}]},
        "ts": "2026-01-01T00:00:02+00:00",
    }) + "\n")
    rv = client.get("/export/demo/20260101_000000")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    assert "<style>" in body
    assert "/static/style.css" not in body
    assert "hi from MainAgent" in body
    assert "research_1" in body
    assert "hi from researcher" in body


def test_export_404_unknown_run(client):
    assert client.get("/export/demo/19990101_000000").status_code == 404
