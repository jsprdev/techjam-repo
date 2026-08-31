"""Official entry point. Bridges the organiser's harness to our implementation.

The evaluator does `from starter.agent import Agent` at module load and
constructs `Agent(args.catalog)`. That import path is hardcoded and the rules
forbid editing evaluator files, so this file has to stay here, keep the class
name, and keep the one positional constructor argument.

It is a bridge, not an implementation. All logic lives in `src/`, one directory
above the kit. Run the official harness exactly as the organiser documents it:

    cd techjam-conversational-search
    python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl

The previous contents of this file, the organiser's weak BM25 baseline scoring
0.107, are preserved in git history.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Under `python3 -m evaluator.local_evaluator` the process CWD is the kit root,
# so sys.path[0] is the kit and `src/` one level above is invisible. Add the
# repo root explicitly rather than relying on how the harness was launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agent import Agent  # noqa: E402,F401  re-exported as the entry symbol

__all__ = ["Agent"]
