"""The v0 modules must satisfy the frozen protocols.

Roles 1 to 3 will each replace their module. These tests are what tells them
their replacement still fits the seam the other two build against.
"""

from __future__ import annotations

from src.catalog import load
from src.config import Config
from src.interfaces import Ranker, Retriever, SlotState
from src.rank import PriorRanker
from src.retrieval import TfidfRetriever
from src.state import Slots


def test_retriever_satisfies_the_protocol(fake_catalog_path):
    assert isinstance(TfidfRetriever(load(fake_catalog_path), Config()), Retriever)


def test_ranker_satisfies_the_protocol(fake_catalog_path):
    assert isinstance(PriorRanker(load(fake_catalog_path), Config()), Ranker)


def test_slots_satisfies_the_protocol():
    assert isinstance(Slots(), SlotState)


def test_retrieve_respects_k_and_orders_by_score(fake_catalog_path):
    retriever = TfidfRetriever(load(fake_catalog_path), Config())
    results = retriever.retrieve("leather belt buckle closure", 3)
    assert len(results) <= 3
    assert [score for _, score in results] == sorted(
        (score for _, score in results), reverse=True
    )
    assert results[0][0] == "B000000001"


def test_retrieve_on_empty_query_returns_empty(fake_catalog_path):
    assert TfidfRetriever(load(fake_catalog_path), Config()).retrieve("   ", 5) == []


def test_to_query_includes_profile_tags():
    slots = Slots()
    slots.profile = {"preference_tags": ["comfort"]}
    slots.observe("leather belt", 1)
    assert "comfort" in slots.to_query()
    assert "leather belt" in slots.to_query()


def test_pick_attribute_never_repeats_before_exhausting():
    slots = Slots()
    picked = [slots.pick_attribute() for _ in range(9)]
    assert len(set(picked)) == 9, "a repeated ask wastes the information"


def test_rank_returns_bare_asins_best_first(fake_catalog_path):
    catalog = load(fake_catalog_path)
    ranker = PriorRanker(catalog, Config())
    ranked = ranker.rank([("B000000001", 1.0), ("B000000003", 0.4)], Slots(), {})
    assert ranked == ["B000000001", "B000000003"]
    assert all(isinstance(asin, str) for asin in ranked)


