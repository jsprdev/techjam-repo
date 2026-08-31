"""Per-session turn traces.

Owned by role 4. Logged from day one because a judge asking why one specific
session failed is answerable only from a trace, and reconstructing them later
never happens.

The agent writes here; `evaluation/run_eval.py` drains it after a run. Keeping
the sink a module-level singleton avoids threading a logger through every
module, which is acceptable because the rules define each session as an isolated
single-user interaction with no concurrency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TurnTrace:
    session_id: str
    turn: int
    user_message: str
    query: str
    ask_attribute: str | None
    candidate_count: int
    top_recommendations: list[str]
    elapsed_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


class TraceSink:
    """Collects turn traces in memory, grouped by session."""

    def __init__(self) -> None:
        self._enabled = False
        self._detailed = False
        self._traces: list[TurnTrace] = []

    def enable(self, detailed: bool = False) -> None:
        self._enabled = True
        self._detailed = detailed

    def disable(self) -> None:
        self._enabled = False
        self._detailed = False

    @property
    def enabled(self) -> bool:
        """Whether callers should assemble optional, potentially large traces."""
        return self._enabled

    @property
    def detailed(self) -> bool:
        """Whether callers should assemble diagnostic score breakdowns."""
        return self._enabled and self._detailed

    def clear(self) -> None:
        self._traces.clear()

    def record(self, trace: TurnTrace) -> None:
        if self._enabled:
            self._traces.append(trace)

    def rows(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self._traces]

    def by_session(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for trace in self._traces:
            grouped.setdefault(trace.session_id, []).append(asdict(trace))
        return grouped


SINK = TraceSink()
