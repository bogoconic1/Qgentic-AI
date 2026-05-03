"""Flask app exposing agent run transcripts."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, abort, render_template, request

from . import discovery, parser, render

logger = logging.getLogger(__name__)

_MD_EXT = {".md"}
_PRE_EXT = {".py", ".json", ".txt", ".jsonl", ".log"}
_FILE_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB

_SLUG_FILES: tuple[tuple[str, str], ...] = (
    ("GOAL.md", "GOAL.md"),
    ("RESEARCHER_INSTRUCTIONS.md", "RESEARCHER_INSTRUCTIONS.md"),
)
_RUN_FILES: tuple[tuple[str, str], ...] = (
    ("MAIN.md", "MAIN.md"),
    ("ideas/INDEX.md", "ideas/INDEX.md"),
)


def create_app(task_root_override: Path | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.jinja_env.filters["render_markdown"] = render.render_markdown
    app.jinja_env.filters["format_args_json"] = render.format_args_json
    app.jinja_env.filters["truncate_body"] = render.truncate_body
    app.jinja_env.filters["classify_result"] = render.classify_result
    app.jinja_env.filters["fmt_bytes"] = render.fmt_bytes
    app.jinja_env.filters["fmt_mtime"] = render.fmt_mtime
    app.jinja_env.filters["fmt_meta_value"] = render.fmt_meta_value
    app.jinja_env.globals["bash_command"] = render.bash_command
    app.jinja_env.globals["args_without_command"] = render.args_without_command

    resolved_root = (task_root_override or discovery.task_root()).resolve()
    app.config["TASK_ROOT"] = resolved_root

    @app.route("/")
    def index():
        slugs = discovery.list_slugs(resolved_root)
        return render_template("index.html", slugs=slugs, task_root=resolved_root, live=False)

    @app.route("/task/<slug>")
    def slug_page(slug: str):
        slug_dir = resolved_root / slug
        if not slug_dir.is_dir():
            abort(404)
        runs = discovery.list_runs(resolved_root, slug)
        slug_files = _existing_links(slug_dir, _SLUG_FILES, prefix=f"{slug}/")
        return render_template(
            "slug.html",
            slug=slug,
            runs=runs,
            slug_files=slug_files,
            live=False,
        )

    @app.route("/task/<slug>/<run_id>")
    def run_page(slug: str, run_id: str):
        run = _get_run_or_404(slug, run_id)
        run_dir = resolved_root / slug / run_id
        records = list(parser.iter_records(run_dir / "main_agent_chat.jsonl"))
        run_files = _existing_links(run_dir, _RUN_FILES, prefix=f"{slug}/{run_id}/")
        live = request.args.get("live") == "1"
        return render_template(
            "run.html",
            run=run,
            records=records,
            run_files=run_files,
            live=live,
        )

    @app.route("/task/<slug>/<run_id>/research_<int:n>")
    def research_page(slug: str, run_id: str, n: int):
        run = _get_run_or_404(slug, run_id)
        if n not in run.research_indices:
            abort(404)
        sub_dir = resolved_root / slug / run_id / f"research_{n}"
        log_path = sub_dir / "researcher_chat.jsonl"
        records = list(parser.iter_records(log_path))
        artifacts: list[tuple[str, str]] = []
        if (sub_dir / "RESEARCH.md").is_file():
            artifacts.append(("RESEARCH.md", f"{slug}/{run_id}/research_{n}/RESEARCH.md"))
        for label_dir in ("web_research", "web_fetch"):
            d = sub_dir / label_dir
            if d.is_dir():
                for md in sorted(d.glob("*.md")):
                    artifacts.append(
                        (f"{label_dir}/{md.name}",
                         f"{slug}/{run_id}/research_{n}/{label_dir}/{md.name}")
                    )
        live = request.args.get("live") == "1"
        return render_template(
            "subagent.html",
            kind="researcher",
            n=n,
            subagent_label=f"research_{n}",
            run=run,
            records=records,
            artifacts=artifacts,
            log_missing=not log_path.is_file(),
            live=live,
        )

    @app.route("/export/<slug>/<run_id>")
    def export_page(slug: str, run_id: str):
        run = _get_run_or_404(slug, run_id)
        run_dir = resolved_root / slug / run_id
        main_records = list(parser.iter_records(run_dir / "main_agent_chat.jsonl"))
        subagents: list[tuple[str, list]] = []
        for n in run.research_indices:
            log = run_dir / f"research_{n}" / "researcher_chat.jsonl"
            if log.is_file():
                subagents.append((f"research_{n}", list(parser.iter_records(log))))
        css_path = Path(__file__).parent / "static" / "style.css"
        inline_css = css_path.read_text(encoding="utf-8")
        return render_template(
            "export.html",
            run=run,
            main_records=main_records,
            subagents=subagents,
            inline_css=inline_css,
        )

    @app.route("/file")
    def file_page():
        rel = request.args.get("path", "")
        resolved = discovery.is_safe_path(resolved_root, rel)
        if resolved is None:
            abort(400)
        if not resolved.is_file():
            abort(404)
        size = resolved.stat().st_size
        if size > _FILE_SIZE_LIMIT:
            abort(413)
        suffix = resolved.suffix.lower()
        if suffix in _MD_EXT:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            rendered = render.render_markdown(text)
            return render_template("file.html", rel_path=rel, rendered_html=rendered, raw_text=None)
        if suffix in _PRE_EXT:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            return render_template("file.html", rel_path=rel, rendered_html=None, raw_text=text)
        abort(415)

    def _get_run_or_404(slug: str, run_id: str):
        try:
            return discovery.get_run(resolved_root, slug, run_id)
        except FileNotFoundError:
            abort(404)

    return app


def _existing_links(
    base: Path,
    candidates: tuple[tuple[str, str], ...],
    *,
    prefix: str,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for label, rel in candidates:
        if (base / rel).is_file():
            out.append((label, prefix + rel))
    return out
