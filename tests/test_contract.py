"""The response envelope must satisfy docs/agent_api_contract.json.

Why these exist: if `respond` returns a dict whose `message` is not a string,
the evaluator throws away the ENTIRE response including the recommendations and
scores that turn empty. Nothing in the metric breakdown says why. That failure
is silent and expensive, which is the bar for a test earning its place here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import response
from src.interfaces import ASK_ATTRIBUTES

CONTRACT = json.loads(
    (
        Path(__file__).resolve().parent.parent
        / "techjam-conversational-search/docs/agent_api_contract.json"
    ).read_text(encoding="utf-8")
)


def test_our_enum_matches_the_shipped_contract():
    """Catches the organiser's enum and ours drifting apart."""
    shipped = CONTRACT["turn_response"]["properties"]["ask_attribute"]["enum"]
    assert set(ASK_ATTRIBUTES) == {v for v in shipped if v is not None}


def test_build_produces_a_legal_response():
    built = response.build("hello", "material", [("B1", 0.5), ("B2", 0.25)])
    assert response.violations(built) == []
    assert built["recommendations"][0] == {"parent_asin": "B1", "score": 0.5}


def test_build_drops_duplicates_preserving_order():
    """Rank is array position, and the evaluator drops duplicates itself, so a
    duplicate silently wastes one of only ten scored slots."""
    built = response.build("x", None, ["B1", "B2", "B1", "B3"])
    assert [i["parent_asin"] for i in built["recommendations"]] == ["B1", "B2", "B3"]


def test_build_coerces_an_illegal_attribute_to_null():
    """The local evaluator silently rewrites an unknown attribute to 'other',
    but the shipped enum is closed and the private harness may validate it."""
    assert response.build("x", "not_an_attribute", ["B1"])["ask_attribute"] is None


def test_non_finite_scores_are_never_emitted():
    """NaN serialises as bare NaN, which is not valid JSON. Any future ranker
    doing a division can produce one."""
    built = response.build("x", None, [("B1", float("nan"))])
    json.dumps(built)
    assert "score" not in built["recommendations"][0]


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"message": 1, "ask_attribute": None, "recommendations": []}, "message must be a string"),
        ({"message": "x", "ask_attribute": "nope", "recommendations": []}, "not in enum"),
        ({"message": "x", "ask_attribute": None, "recommendations": {}}, "must be a list"),
        ({"message": "x", "ask_attribute": None, "recommendations": [], "junk": 1}, "additionalProperties"),
        ({"message": "x", "ask_attribute": None}, "missing required field"),
    ],
)
def test_violations_catches_each_contract_break(bad, fragment):
    assert any(fragment in problem for problem in response.violations(bad))


def test_every_agent_turn_is_contract_legal(agent, profile):
    """The one that would actually catch a regression in the shipped agent."""
    agent.reset("contract-session", profile)
    for turn in range(1, 11):
        result = agent.respond("contract-session", "I want a leather belt", turn, 10)
        assert response.violations(result) == [], f"turn {turn}"
