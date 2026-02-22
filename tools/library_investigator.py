"""Library source code investigator sub-agent.

Explores installed Python packages in site-packages to produce accurate API
reports. Called by the DeveloperAgent as a tool during code generation.
"""

import json
import logging
import subprocess
from pathlib import Path

from google.genai import types
from project_config import get_config
from tools.helpers import call_llm
from utils.llm_utils import append_message, get_library_investigator_tools
from schemas.library_investigator import LibraryInvestigatorReport
from prompts.library_investigator import build_system, build_user, SITE_PACKAGES_PATH

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_RUNTIME_CFG = _CONFIG["runtime"]
_LIBRARY_INVESTIGATOR_MODEL = _LLM_CFG["library_investigator_model"]
_MAX_TOOL_STEPS = _RUNTIME_CFG["library_investigator_max_steps"]

_SITE_PACKAGES = Path(SITE_PACKAGES_PATH).resolve()


# ---------------------------------------------------------------------------
# Tool implementations (read-only, scoped to site-packages)
# ---------------------------------------------------------------------------


def _validate_site_packages_path(path: Path) -> str | None:
    """Return an error string if the resolved path escapes site-packages."""
    resolved = path.resolve()
    if not str(resolved).startswith(str(_SITE_PACKAGES)):
        return "Path escapes site-packages directory"
    return None


def _tool_list_packages() -> str:
    if not _SITE_PACKAGES.exists():
        return json.dumps({"error": f"Site-packages not found at {SITE_PACKAGES_PATH}"})

    packages = sorted(
        d.name
        for d in _SITE_PACKAGES.iterdir()
        if d.is_dir()
        and not d.name.startswith("_")
        and not d.name.startswith(".")
        and not d.name.endswith(".dist-info")
        and not d.name.endswith(".egg-info")
    )
    return json.dumps({"packages": packages, "count": len(packages)})


def _tool_read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    full_path = _SITE_PACKAGES / path
    error = _validate_site_packages_path(full_path)
    if error:
        return json.dumps({"error": error})

    resolved = full_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not resolved.is_file():
        return json.dumps({"error": f"Not a file: {path}"})

    try:
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return json.dumps({"error": f"Failed to read file: {e}"})

    total_lines = len(lines)

    if start_line is not None or end_line is not None:
        s = (start_line or 1) - 1
        e = end_line or total_lines
        lines = lines[s:e]
        line_offset = s + 1
    else:
        line_offset = 1
        if total_lines > 300:
            lines = lines[:300]
            numbered = "\n".join(
                f"{i + line_offset}: {line}" for i, line in enumerate(lines)
            )
            return json.dumps(
                {
                    "content": numbered,
                    "total_lines": total_lines,
                    "truncated": True,
                    "showing": "1-300",
                }
            )

    numbered = "\n".join(
        f"{i + line_offset}: {line}" for i, line in enumerate(lines)
    )
    return json.dumps({"content": numbered, "total_lines": total_lines})


def _tool_glob_files(package: str, pattern: str) -> str:
    package_dir = _SITE_PACKAGES / package
    error = _validate_site_packages_path(package_dir)
    if error:
        return json.dumps({"error": error})

    resolved = package_dir.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Package directory not found: {package}"})

    matches = sorted(str(p.relative_to(resolved)) for p in resolved.glob(pattern))
    total = len(matches)
    if total > 50:
        matches = matches[:50]

    return json.dumps({"matches": matches, "total": total, "truncated": total > 50})


def _tool_grep_code(package: str, pattern: str, max_results: int = 20) -> str:
    package_dir = _SITE_PACKAGES / package
    error = _validate_site_packages_path(package_dir)
    if error:
        return json.dumps({"error": error})

    resolved = package_dir.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Package directory not found: {package}"})

    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "-E", pattern, str(resolved)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Search timed out"})
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})

    site_packages_str = str(_SITE_PACKAGES)
    cleaned = [
        line.replace(site_packages_str + "/", "") for line in lines[:max_results]
    ]

    return json.dumps(
        {
            "matches": cleaned,
            "total_matches": len(lines),
            "showing": min(max_results, len(lines)),
        }
    )


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _execute_tool_call(item) -> str:
    args = dict(item.args)

    if item.name == "list_packages":
        return _tool_list_packages()
    elif item.name == "read_file":
        return _tool_read_file(
            args["path"],
            args.get("start_line"),
            args.get("end_line"),
        )
    elif item.name == "glob_files":
        return _tool_glob_files(args["package"], args["pattern"])
    elif item.name == "grep_code":
        return _tool_grep_code(
            args["package"], args["pattern"], args.get("max_results", 20)
        )
    else:
        return json.dumps({"error": f"Unknown tool: {item.name}"})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def investigate_library(query: str) -> str:
    """Run the library investigator sub-agent and return a formatted report.

    Args:
        query: Natural language query about library APIs.

    Returns:
        Formatted string report of the investigation findings.
    """
    logger.info("Starting library investigation: %s", query[:100])

    system_prompt = build_system()
    user_prompt = build_user(query)
    tools = get_library_investigator_tools()
    input_list = [append_message("user", user_prompt)]

    for step in range(_MAX_TOOL_STEPS):
        is_last_step = step == _MAX_TOOL_STEPS - 1

        logger.info("Library investigator step %d/%d", step + 1, _MAX_TOOL_STEPS)

        response = call_llm(
            model=_LIBRARY_INVESTIGATOR_MODEL,
            system_instruction=system_prompt,
            function_declarations=tools if not is_last_step else [],
            messages=input_list,
            text_format=LibraryInvestigatorReport if is_last_step else None,
        )

        # Structured output on last step
        if hasattr(response, "packages_examined"):
            logger.info("Library investigation completed at step %d", step + 1)
            return _format_report(response)

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call for part in parts if hasattr(part, "function_call")
        )

        if not has_function_calls:
            logger.info(
                "No function calls at step %d, requesting structured output", step + 1
            )
            response = call_llm(
                model=_LIBRARY_INVESTIGATOR_MODEL,
                system_instruction=system_prompt,
                messages=input_list,
                text_format=LibraryInvestigatorReport,
            )
            return _format_report(response)

        function_responses = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_result_str = _execute_tool_call(part.function_call)
                function_responses.append(
                    types.Part.from_function_response(
                        name=part.function_call.name,
                        response={"result": tool_result_str},
                    )
                )

        input_list.append(response.candidates[0].content)
        if function_responses:
            input_list.append(
                types.Content(role="function", parts=function_responses)
            )

    # Exhausted steps
    logger.warning(
        "Library investigator exhausted %d steps, forcing final report",
        _MAX_TOOL_STEPS,
    )
    response = call_llm(
        model=_LIBRARY_INVESTIGATOR_MODEL,
        system_instruction=system_prompt
        + "\n\nReturn your findings now based on what you have gathered.",
        messages=input_list,
        text_format=LibraryInvestigatorReport,
    )
    return _format_report(response)


def _format_report(report: LibraryInvestigatorReport) -> str:
    return (
        f"## Library Investigation Report\n\n"
        f"**Packages examined:** {report.packages_examined}\n\n"
        f"### API Surface\n{report.api_surface}\n\n"
        f"### Constructor Signatures\n{report.constructor_signatures}\n\n"
        f"### Key Methods\n{report.key_methods}\n\n"
        f"### Usage Patterns\n{report.usage_patterns}\n\n"
        f"### Caveats\n{report.caveats}"
    )
