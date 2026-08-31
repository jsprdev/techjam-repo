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


def test_to_query_carries_the_category_and_the_constraints():
    slots = Slots()
    slots.observe("I'm looking for Accessories Belts. Buckle closure", 1)
    slots.observe("For that, what matters is: leather; 100% Leather.", 2)
    query = slots.to_query()
    assert "Accessories Belts" in query
    assert "leather" in query and "Buckle closure" in query
    assert "looking for" not in query, "conversational framing leaked into the query"


def test_no_information_replies_are_dropped_when_the_flag_is_on():
    """Off by default: measured, dropping them scores 0.7369 against 0.7422 for
    keeping them. The flag stays so the decision can be re-tested after any
    retrieval change, rather than inherited as folklore."""
    from src.config import Config

    slots = Slots(config=Config().with_overrides(drop_no_information=True))
    slots.observe("I'm looking for Accessories Belts. Buckle closure", 1)
    before = slots.to_query()
    slots.observe("I don't have an additional preference for brand.", 2)
    slots.observe("Those options are not quite right yet. Ask me about one specific attribute.", 3)
    assert slots.to_query() == before


def test_multiple_constraints_are_split_apart():
    """The simulator joins them with '; ' and each is lifted verbatim from the
    target's own record, so splitting gives ranking whole phrases to match."""
    slots = Slots()
    slots.observe("For that, what matters is: Water Resistant; 3 Year Battery.", 1)
    assert slots.constraints() == ["Water Resistant", "3 Year Battery"]


def test_pick_attribute_never_asks_an_unanswerable_attribute():
    """category, brand and budget are never returned by the evaluator's
    constraint classifier, so asking them is a guaranteed wasted turn. The
    first version of the policy spent turns 2 and 3 on two of them."""
    from src.state.slots import UNANSWERABLE

    slots = Slots()
    picked = [slots.pick_attribute() for _ in range(8)]
    assert not (set(picked) & set(UNANSWERABLE)), f"asked an unanswerable attribute: {picked}"


def test_pick_attribute_orders_by_measured_yield():
    slots = Slots()
    assert [slots.pick_attribute() for _ in range(3)] == ["feature", "material", "color"]


def test_an_empty_answer_retires_that_attribute():
    """The customer saying the bucket is empty is information. Asking again
    spends a turn to be told the same thing twice."""
    slots = Slots()
    first = slots.pick_attribute()
    slots.observe(f"I don't have an additional preference for {first}.", 1)
    later = [slots.pick_attribute() for _ in range(6)]
    assert first not in later


def test_rank_returns_bare_asins_best_first(fake_catalog_path):
    catalog = load(fake_catalog_path)
    ranker = PriorRanker(catalog, Config())
    ranked = ranker.rank([("B000000001", 1.0), ("B000000003", 0.4)], Slots(), {})
    assert ranked == ["B000000001", "B000000003"]
    assert all(isinstance(asin, str) for asin in ranked)


