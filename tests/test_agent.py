"""Agent robustness. Every failure here costs whole sessions.

The evaluator swallows exceptions and substitutes an empty response, so a crash
is silent and expensive: it forfeits every remaining turn of that session. These
tests exist because that failure mode leaves no trace in the score breakdown.
"""

from __future__ import annotations

import pytest

from src import response
from src.interfaces import MAX_TURNS


def test_reset_then_respond_returns_recommendations(agent, profile):
    agent.reset("s", profile)
    result = agent.respond("s", "leather belt with buckle", 1, 10)
    assert result["recommendations"], "never return an empty list, a guess can still hit"


def test_respond_without_reset_does_not_raise(agent):
    """The starter agent raises here. We recover instead, because raising in the
    real harness would forfeit the session."""
    result = agent.respond("never-reset-session", "a watch", 1, 10)
    assert response.violations(result) == []
    assert result["recommendations"]


@pytest.mark.parametrize(
    "message",
    ["", "   ", "\n\t", "?" * 5000, "🙂🙂🙂", "SELECT * FROM products", 'x" OR "1"="1'],
)
def test_hostile_messages_do_not_raise(agent, profile, message):
    agent.reset("hostile", profile)
    result = agent.respond("hostile", message, 1, 10)
    assert response.violations(result) == []


def test_turn_past_the_cap_is_handled(agent, profile):
    """Exceeding ten turns is a forced zero, so we must not depend on the
    harness stopping us."""
    agent.reset("capped", profile)
    result = agent.respond("capped", "a watch", MAX_TURNS + 5, 10)
    assert response.violations(result) == []
    assert result["recommendations"]


def test_sessions_are_isolated(agent, profile):
    agent.reset("a", profile)
    agent.reset("b", profile)
    agent.respond("a", "leather belt buckle closure", 1, 10)
    result_b = agent.respond("b", "stainless steel wrist watch", 1, 10)
    # Session b must not have inherited session a's belt constraint.
    top_b = result_b["recommendations"][0]["parent_asin"]
    assert top_b == "B000000002", "session b should find the watch, not a's belt"


def test_reset_clears_prior_state(agent, profile):
    agent.reset("reused", profile)
    agent.respond("reused", "leather belt", 1, 10)
    agent.reset("reused", profile)
    result = agent.respond("reused", "stainless steel wrist watch", 1, 10)
    assert result["recommendations"][0]["parent_asin"] == "B000000002"


def test_a_full_ten_turn_session_never_raises(agent, profile):
    agent.reset("long", profile)
    messages = [
        "I'm looking for Accessories Belts. Buckle closure",
        "For that, what matters is: leather; 100% Leather.",
        "I don't have an additional preference for color.",
        "Actually, ignore my earlier preference. What I need is: Cork sole.",
        "For that, what matters is: Adjustable Flat Thong.",
    ]
    for turn in range(1, MAX_TURNS + 1):
        message = messages[(turn - 1) % len(messages)]
        result = agent.respond("long", message, turn, 10)
        assert response.violations(result) == []


# -- reset is the one method the evaluator does NOT wrap ----------------------
# local_evaluator.py calls agent.reset(...) bare, then wraps agent.respond(...)
# in try/except. So a raise in reset aborts all 200 sessions, not one turn.


@pytest.mark.parametrize(
    "bad_profile",
    [None, {}, [], "a string", 42, {"average_prior_rating": None}, {"preference_tags": None}],
)
def test_reset_never_raises_on_a_degenerate_profile(agent, bad_profile):
    agent.reset("degenerate", bad_profile)
    result = agent.respond("degenerate", "a watch", 1, 10)
    assert response.violations(result) == []


def test_null_average_prior_rating_survives_a_whole_session(agent):
    """Never null in the public 200, but the contract types it number-or-null,
    so the private 800 may send it."""
    agent.reset("null-rating", {"average_prior_rating": None, "preference_tags": ["fit"]})
    for turn in range(1, 4):
        assert response.violations(agent.respond("null-rating", "leather belt", turn, 10)) == []


def test_turn_ten_still_returns_a_full_list(agent, profile):
    """Turn 10 is a full scoring turn: the evaluator checks for a hit before it
    breaks out of the loop, so a terminal 'sorry' with no items forfeits it."""
    agent.reset("last", profile)
    result = agent.respond("last", "stainless steel wrist watch", 10, 10)
    assert len(result["recommendations"]) >= 1
