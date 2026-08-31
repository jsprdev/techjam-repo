"""The response envelope must satisfy docs/agent_api_contract.json.

These are the fatal-severity tests. A malformed response costs a whole session,
and the contract sets additionalProperties: false in three places, so an extra
key is a violation even though the local evaluator happens to tolerate it. We
test against the contract, not against the evaluator's leniency.
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
    """Guards against the organiser's enum and ours drifting apart."""
    shipped = CONTRACT["turn_response"]["properties"]["ask_attribute"]["enum"]
    assert set(ASK_ATTRIBUTES) == {value for value in shipped if value is not None}
    assert None in shipped


def test_response_allowed_keys_match_contract():
    allowed = set(CONTRACT["turn_response"]["properties"])
    built = response.build("hi", "color", ["B1"])
    assert set(built) <= allowed
    assert CONTRACT["turn_response"]["additionalProperties"] is False


def test_build_produces_a_legal_response():
    built = response.build("hello", "material", [("B1", 0.5), ("B2", 0.25)])
    assert response.violations(built) == []
    assert built["recommendations"][0] == {"parent_asin": "B1", "score": 0.5}


def test_build_drops_duplicates_preserving_order():
    built = response.build("x", None, ["B1", "B2", "B1", "B3"])
    assert [item["parent_asin"] for item in built["recommendations"]] == ["B1", "B2", "B3"]


def test_build_coerces_an_illegal_attribute_to_null():
    built = response.build("x", "not_an_attribute", ["B1"])
    assert built["ask_attribute"] is None
    assert response.violations(built) == []


def test_build_respects_the_hundred_item_cap():
    built = response.build("x", None, [f"B{i}" for i in range(250)])
    assert len(built["recommendations"]) == CONTRACT["turn_response"]["properties"]["recommendations"]["maxItems"]


def test_build_never_emits_negative_token_counts():
    built = response.build("x", None, ["B1"], prompt_tokens=-5, completion_tokens=-1)
    assert built["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


@pytest.mark.parametrize(
    "bad, expected_fragment",
    [
        ({"message": 1, "ask_attribute": None, "recommendations": []}, "message must be a string"),
        ({"message": "x", "ask_attribute": "nope", "recommendations": []}, "ask_attribute not in enum"),
        ({"message": "x", "ask_attribute": None, "recommendations": {}}, "recommendations must be a list"),
        ({"message": "x", "ask_attribute": None, "recommendations": [], "junk": 1}, "additionalProperties"),
        ({"message": "x", "ask_attribute": None}, "missing required field: recommendations"),
        (
            {"message": "x", "ask_attribute": None, "recommendations": [{"parent_asin": "B1", "oops": 2}]},
            "extra keys",
        ),
        (
            {"message": "x", "ask_attribute": None, "recommendations": [], "usage": {"prompt_tokens": -1, "completion_tokens": 0}},
            "non-negative",
        ),
        ("not a dict", "must be a dict"),
    ],
)
def test_violations_catches_each_contract_break(bad, expected_fragment):
    found = response.violations(bad)
    assert any(expected_fragment in problem for problem in found), found


def test_every_agent_turn_is_contract_legal(agent, profile):
    """The real thing, end to end, across a multi turn session."""
    agent.reset("contract-session", profile)
    for turn in range(1, 11):
        result = agent.respond("contract-session", "I want a leather belt", turn, 10)
        assert response.violations(result) == [], f"turn {turn}: {response.violations(result)}"
