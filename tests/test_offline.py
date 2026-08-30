"""The agent must complete a full session with no network access.

The submission rules warn that organiser policy may disable network access for
official scoring. This test is the enforcement: it makes every socket
construction raise, then runs a real multi turn session. If anyone later puts an
LLM call on the critical path, this test fails before the graded run does.
"""

from __future__ import annotations

import socket

import pytest

from src import response
from src.interfaces import MAX_TURNS


class NetworkTouched(BaseException):
    """Deliberately not an Exception.

    Agent.respond wraps its work in a blanket `except Exception` so a failed
    turn degrades to a popularity list instead of forfeiting the session. An
    Exception-derived sentinel is swallowed by that handler, the fallback comes
    back contract-legal, and the test passes on an agent that just opened a
    socket. Only a BaseException gets past it.
    """


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise NetworkTouched("network access attempted during a graded turn")

    for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, name, blocked)
    return blocked


def test_the_sentinel_is_not_swallowed_by_the_fallback(agent, profile, no_network):
    """Guards the guard. If this ever fails, every other test in this file has
    silently stopped testing anything."""
    import src.retrieval.baseline as baseline

    def touch(self, query, k):
        raise NetworkTouched("pretending to call out")

    original = baseline.TfidfRetriever.retrieve
    baseline.TfidfRetriever.retrieve = touch
    try:
        agent.reset("sentinel", profile)
        with pytest.raises(NetworkTouched):
            agent.respond("sentinel", "a watch", 1, 10)
    finally:
        baseline.TfidfRetriever.retrieve = original


def test_full_session_runs_with_sockets_blocked(agent, profile, no_network):
    agent.reset("offline", profile)
    for turn in range(1, MAX_TURNS + 1):
        result = agent.respond("offline", "stainless steel wrist watch", turn, 10)
        assert response.violations(result) == []
        assert result["recommendations"]


def test_agent_construction_needs_no_network(fake_catalog_path, no_network):
    from src.agent import Agent

    built = Agent(fake_catalog_path)
    built.reset("s", {})
    assert built.respond("s", "leather belt", 1, 10)["recommendations"]
