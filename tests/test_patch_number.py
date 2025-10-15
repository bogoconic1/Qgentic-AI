import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.developer import DeveloperAgent


SIMPLE_BASE = """print('hello world')\nA=1\nB=2\n"""

# Minimal diff that changes a constant and adds a new line
SIMPLE_DIFF_TEMPLATE = """--- {base}
+++ {base}
@@ -1,3 +1,4 @@
 print('hello world')
-A=1
+A=42
 B=2
+print('patched')
"""


# -----------------
# Pytest fixtures
# -----------------

import pytest


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    slug = "dummy-slug"
    iteration = 3
    monkeypatch.setattr(DeveloperAgent, "_load_benchmark_info", lambda self: None)
    a = DeveloperAgent(slug, iteration)
    a.outputs_dir = tmp_path
    a.developer_log_path = tmp_path / "developer_patch.log"
    a.plan_path = tmp_path / "developer_patch_plan.md"
    return a


@pytest.fixture()
def base_version():
    return 22


@pytest.fixture()
def target_version():
    return 43


@pytest.fixture()
def base_filename(agent, base_version):
    return agent._code_filename(base_version)


@pytest.fixture()
def diff_text(base_filename):
    return SIMPLE_DIFF_TEMPLATE.format(base=base_filename)


def test_append_patch_directive_references_given_base_version(agent, base_version, base_filename):
    instr = "Fix the issue, write a diff."
    out = agent._append_patch_directive(instr, base_version)
    assert base_filename in out
    assert "```diff" in out
    assert "@@" in out


def test_apply_patch_uses_specified_base_version(agent, base_version, target_version, diff_text, tmp_path):
    target_filename = agent._code_filename(target_version)
    # Ensure base file exists via fixture usage in previous test or create here explicitly
    base_filename = agent._code_filename(base_version)
    if not (tmp_path / base_filename).exists():
        (tmp_path / base_filename).write_text(SIMPLE_BASE)

    print("diff_text ", diff_text)
    print("base_version ", base_version)
    print("target_version ", target_version)

    updated = agent._apply_patch(base_version=base_version, diff_payload=diff_text, target_version=target_version)
    print("updated ", updated)
    assert (tmp_path / target_filename).exists(), "Patched target file not created"
    assert updated is not None
    content = (tmp_path / target_filename).read_text()
    assert "A=42" in content
    assert "print('patched')" in content


