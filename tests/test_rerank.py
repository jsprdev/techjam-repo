"""The live reranker must never be able to break a graded turn.

Official scoring may run with network access disabled, so every failure path
has to return the input order unchanged rather than raise. These tests inject a
fake client, so they need no key and no network.
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.language import rerank


class FakeResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.usage = type("Usage", (), {"input_tokens": 120, "output_tokens": 30})()


class FakeClient:
    """Returns a canned reply, or raises, depending on construction."""

    def __init__(self, text: str = "", error: Exception | None = None, stop_reason: str = "end_turn") -> None:
        self._text, self._error, self._stop = text, error, stop_reason
        self.messages = self

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return FakeResponse(self._text, self._stop)


CANDIDATES = [("A", "leather belt"), ("B", "steel watch"), ("C", "cork sandals")]
ON = Config().with_overrides(use_llm=True)


def test_disabled_by_default_and_returns_input_order():
    """The default must be off: a model on the turn path is a way to score zero
    if the graded run has no network."""
    assert Config().use_llm is False
    result = rerank.rerank(["leather"], CANDIDATES, Config())
    assert result.order == ["A", "B", "C"]
    assert result.used_llm is False


def test_a_valid_reply_actually_reorders():
    result = rerank.rerank(["a watch"], CANDIDATES, ON, client=FakeClient('{"order": [2, 3, 1]}'))
    assert result.order == ["B", "C", "A"]
    assert result.used_llm is True
    assert (result.prompt_tokens, result.completion_tokens) == (120, 30)


@pytest.mark.parametrize(
    "client, why",
    [
        (FakeClient(error=RuntimeError("boom")), "call failed"),
        (FakeClient(error=TimeoutError("slow")), "timed out"),
        (FakeClient("not json at all"), "unparseable"),
        (FakeClient('{"wrong_key": []}'), "missing key"),
        (FakeClient('{"order": "not a list"}'), "wrong type"),
        (FakeClient("", stop_reason="refusal"), "safety refusal"),
    ],
)
def test_every_failure_path_returns_the_input_order(client, why):
    result = rerank.rerank(["a watch"], CANDIDATES, ON, client=client)
    assert result.order == ["A", "B", "C"], f"{why} did not fall back cleanly"
    assert result.used_llm is False


@pytest.mark.parametrize(
    "reply",
    ['{"order": [1]}', '{"order": [2, 2, 2]}', '{"order": [9, 1, 2, 3]}', '{"order": []}'],
)
def test_a_dropped_duplicated_or_invented_index_never_loses_a_candidate(reply):
    """The model is not trusted. Whatever it returns, every candidate must
    survive exactly once or the agent silently drops recommendations."""
    result = rerank.rerank(["a watch"], CANDIDATES, ON, client=FakeClient(reply))
    assert sorted(result.order) == ["A", "B", "C"]
    assert len(result.order) == len(set(result.order))


def test_candidates_beyond_the_cap_are_preserved():
    many = [(f"P{i}", f"product {i}") for i in range(rerank.MAX_CANDIDATES + 5)]
    result = rerank.rerank(["x"], many, ON, client=FakeClient('{"order": [1]}'))
    assert sorted(result.order) == sorted(a for a, _ in many)


def test_no_key_means_no_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = rerank.rerank(["x"], CANDIDATES, ON)
    assert result.used_llm is False
    assert result.reason == "no api key"
