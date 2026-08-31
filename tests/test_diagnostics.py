"""The diagnostic scripts must actually run, and the trace must carry what they read.

`evaluation/rank_diagnostics.py` shipped broken in two independent ways and
nobody noticed, because nothing imports it and running it costs six minutes. It
called `run()` with an argument that does not exist, and it read two trace keys
the agent never emitted. Both are the same class of bug: a script committed
without being executed.

These tests are cheap because they exercise the seam rather than the full run:
the call signature, and the trace contract the diagnostics depend on.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from evaluation import rank_diagnostics, run_eval
from src.trace import SINK


def test_rank_diagnostics_calls_run_with_arguments_run_accepts():
    """A signature drift here is a TypeError six minutes into a run.

    Parsed rather than pattern matched, so reformatting the call does not
    silently turn this check off.
    """
    accepted = set(inspect.signature(run_eval.run).parameters)
    tree = ast.parse(textwrap.dedent(inspect.getsource(rank_diagnostics.main)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run"
    ]
    assert calls, "rank_diagnostics.main no longer calls run(), so this test is checking nothing"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords if kw.arg}
        unknown = keywords - accepted
        assert not unknown, f"rank_diagnostics passes {sorted(unknown)}, which run() does not accept"
        positional = len(call.args)
        assert positional + len(keywords) <= len(accepted), (
            f"rank_diagnostics passes {positional + len(keywords)} arguments "
            f"but run() takes {len(accepted)}"
        )


def test_the_trace_carries_every_key_the_diagnostics_read(agent, profile):
    """The diagnostics join on these two keys. Renaming one breaks them silently."""
    SINK.clear()
    SINK.enable()
    try:
        agent.reset("diag", profile)
        agent.respond("diag", "I am looking for a leather belt", 1, 10)
        rows = SINK.rows()
    finally:
        SINK.disable()
        SINK.clear()

    assert rows, "tracing was enabled but nothing was recorded"
    extra = rows[0]["extra"]
    for key in ("retrieval_shortlist", "ranking"):
        assert key in extra, f"the trace no longer carries {key!r}"

    assert extra["retrieval_shortlist"], "shortlist was traced but empty"
    assert "parent_asin" in extra["retrieval_shortlist"][0]
    for name in rank_diagnostics.COMPONENTS:
        assert name in extra["ranking"][0], f"score component {name!r} missing from the trace"


def test_score_components_are_not_built_when_tracing_is_off(agent, profile):
    """The breakdown costs a dict per candidate per turn, so a scored run must not pay it."""
    SINK.disable()
    SINK.clear()
    agent.reset("nodiag", profile)
    agent.respond("nodiag", "I am looking for a leather belt", 1, 10)
    assert not SINK.rows(), "tracing is off, so nothing should have been recorded"
