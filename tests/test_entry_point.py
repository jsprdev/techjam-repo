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


# Hashes taken from the pristine participant-kit release. Pinning content rather
# than diffing against git is deliberate: `git diff HEAD` compares the working
# tree to the last commit, so it reports clean for an edit that was committed,
# and reports clean again if git itself is unavailable. A content hash is
# tamper-evident regardless of history.
PRISTINE_EVALUATOR_SHA256 = {
    "local_evaluator.py": "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564",
    "__init__.py": "c597e982409b24fe5411298cfe033aeb287eafcf26c33e34b8c43294cff0a917",
}


def test_evaluator_files_are_byte_identical_to_the_kit():
    """The rules forbid editing evaluator files and treat a modified run as
    invalid, so this has to catch a committed edit, not just a dirty tree."""
    import hashlib

    for name, expected in PRISTINE_EVALUATOR_SHA256.items():
        path = KIT / "evaluator" / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{path} has been modified. The submission rules forbid editing "
            f"evaluator files. Expected {expected}, got {actual}."
        )


def test_uncommitted_evaluator_edits_are_also_caught():
    """Cheap second signal that names the file, for a dirty working tree."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "techjam-conversational-search/evaluator/"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, "git unavailable, so this check proved nothing"
    assert result.stdout.strip() == "", f"evaluator files dirty: {result.stdout}"
