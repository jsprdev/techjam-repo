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


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted during a graded turn")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    return blocked


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
