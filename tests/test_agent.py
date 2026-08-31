"""Agent robustness, limited to failures that actually cost score.

The evaluator swallows exceptions from `respond` and substitutes an empty
response, so a crash is silent and forfeits every remaining turn of that
session. It calls `reset` OUTSIDE any try/except, so a raise there aborts all
200 sessions. Those two facts are why this file exists.

Note the trap that makes naive tests here worthless: `respond` degrades to a
popularity-ordered guess on any failure, so asserting only "the response was
well formed" passes on a completely dead pipeline. Anything checking that the
agent WORKS uses `strict_agent`, which re-raises.
"""

from __future__ import annotations

import pytest

from src import response
from src.interfaces import MAX_TURNS


# -- reset: a raise here kills all 200 sessions, not one turn ----------------


@pytest.mark.parametrize("bad_profile", [None, "a string", {"average_prior_rating": None}])
def test_reset_never_raises_on_a_degenerate_profile(agent, bad_profile):
    agent.reset("degenerate", bad_profile)
    assert response.violations(agent.respond("degenerate", "a watch", 1, 10)) == []


# -- respond: a raise costs the rest of the session --------------------------


def test_respond_without_reset_does_not_raise(agent):
    """The starter agent raises here. Raising in the harness forfeits the run."""
    result = agent.respond("never-reset-session", "a watch", 1, 10)
    assert response.violations(result) == []
    assert result["recommendations"]


@pytest.mark.parametrize("message", ["", "?" * 5000, "🙂🙂🙂"])
def test_hostile_messages_do_not_raise(agent, profile, message):
    agent.reset("hostile", profile)
    assert response.violations(agent.respond("hostile", message, 1, 10)) == []


def test_production_agent_degrades_rather_than_raising(agent, profile, monkeypatch):
    import src.retrieval.baseline as baseline

    monkeypatch.setattr(
        baseline.TfidfRetriever,
        "retrieve",
        lambda self, query, k: (_ for _ in ()).throw(RuntimeError("dead")),
    )
    agent.reset("degrade", profile)
    result = agent.respond("degrade", "leather belt", 1, 10)
    assert result["recommendations"], "a degraded turn must still guess"


def test_a_full_ten_turn_session_never_raises(agent, profile):
    agent.reset("long", profile)
    messages = [
        "I'm looking for Accessories Belts. Buckle closure",
        "For that, what matters is: leather; 100% Leather.",
        "Actually, ignore my earlier preference. What I need is: Cork sole.",
    ]
    for turn in range(1, MAX_TURNS + 1):
        result = agent.respond("long", messages[(turn - 1) % len(messages)], turn, 10)
        assert response.violations(result) == []


def test_turn_ten_still_returns_recommendations(agent, profile):
    """Turn 10 is a full scoring turn: the hit check runs before the loop
    breaks, so a terminal message with an empty list forfeits it."""
    agent.reset("last", profile)
    assert agent.respond("last", "stainless steel wrist watch", 10, 10)["recommendations"]


# -- session isolation: one Agent serves all 200 sessions in sequence --------


def test_sessions_do_not_leak_into_each_other(agent, profile):
    agent.reset("a", profile)
    agent.reset("b", profile)
    agent.respond("a", "leather belt buckle closure", 1, 10)
    result = agent.respond("b", "stainless steel wrist watch", 1, 10)
    assert result["recommendations"][0]["parent_asin"] == "B000000002"


# -- the pipeline actually works, not just returns something -----------------


def test_a_dead_pipeline_is_actually_detected(strict_agent, profile, monkeypatch):
    """Guards the guard. If this fails, the two tests below stopped testing."""
    import src.retrieval.baseline as baseline

    monkeypatch.setattr(
        baseline.TfidfRetriever,
        "retrieve",
        lambda self, query, k: (_ for _ in ()).throw(RuntimeError("dead")),
    )
    strict_agent.reset("dead", profile)
    with pytest.raises(RuntimeError):
        strict_agent.respond("dead", "leather belt", 1, 10)


@pytest.mark.parametrize(
    "query, expected_top",
    [
        ("stainless steel wrist watch water resistant", "B000000002"),
        ("cork sole adjustable flat thong sandals", "B000000004"),
    ],
)
def test_retrieval_finds_the_right_product(strict_agent, profile, query, expected_top):
    """The fallback returns popularity order regardless of query, so a
    query-specific winner is what separates a working pipeline from a dead one."""
    strict_agent.reset("finds", profile)
    result = strict_agent.respond("finds", query, 1, 10)
    assert result["recommendations"][0]["parent_asin"] == expected_top
