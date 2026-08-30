"""The official harness must actually run our agent.

`local_evaluator.py` does `from starter.agent import Agent` at module load and
never takes an override. If that file still holds the organiser's BM25 baseline,
every score we report locally comes from a code path the graded command never
takes. This test is the guard against shipping that mistake.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"


def test_starter_agent_re_exports_our_implementation():
    source = (KIT / "starter/agent.py").read_text(encoding="utf-8")
    assert "from src.agent import Agent" in source, "the kit entry point is not bridged to src/"


def test_entry_point_imports_the_same_class_as_src():
    """Imported the way the evaluator imports it, from the kit directory."""
    probe = (
        "import sys; sys.path.insert(0, %r); "
        "from starter.agent import Agent as Entry; "
        "sys.path.insert(0, %r); "
        "from src.agent import Agent as Ours; "
        "assert Entry is Ours, 'entry point resolves to a different class'; "
        "print('ok')" % (str(KIT), str(REPO_ROOT))
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(KIT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_evaluator_file_is_unmodified():
    """The rules forbid editing evaluator files. Fail loudly if anyone did."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "techjam-conversational-search/evaluator/"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.stdout.strip() == "", f"evaluator files modified: {result.stdout}"
