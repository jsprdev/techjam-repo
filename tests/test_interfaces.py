"""The retrieval, state, and ranking modules must satisfy the frozen protocols.

Roles 1 to 3 will each replace their module. These tests are what tells them
their replacement still fits the seam the other two build against.
"""

from __future__ import annotations

import json

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


def write_catalog(tmp_path, products):
    path = tmp_path / "field-catalog.jsonl"
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    return path


def field_products():
    """Pairs keep distinguishing tokens above the retriever's min_df threshold."""
    common = {
        "features": [],
        "categories": [],
        "store": "",
        "average_rating": 4.0,
        "rating_number": 1,
    }
    return [
        {**common, "parent_asin": "title-match", "title": "cerulean jacket", "description": "plain apparel", "details": {}},
        {**common, "parent_asin": "description-match", "title": "basic apparel", "description": "cerulean jacket", "details": {}},
        {**common, "parent_asin": "title-neighbour", "title": "cerulean accessory", "description": "other apparel", "details": {"Motif": "celestial ornament"}},
        {**common, "parent_asin": "description-neighbour", "title": "other apparel", "description": "cerulean accessory", "details": {"Motif": "celestial"}},
        {**common, "parent_asin": "details-match", "title": "plain apparel", "description": "plain apparel", "details": {"Motif": "celestial dragon"}},
        {**common, "parent_asin": "details-dragon-neighbour", "title": "plain apparel", "description": "plain apparel", "details": {"Motif": "plain dragon"}},
    ]


def test_field_weights_change_which_field_match_wins(tmp_path):
    catalog = load(write_catalog(tmp_path, field_products()))
    title_first = TfidfRetriever(
        catalog,
        Config().with_overrides(
            weight_title=10.0,
            weight_features=0.0,
            weight_categories=0.0,
            weight_description=0.0,
            weight_store=0.0,
            weight_details=0.0,
        ),
    )
    description_first = TfidfRetriever(
        catalog,
        Config().with_overrides(
            weight_title=0.0,
            weight_features=0.0,
            weight_categories=0.0,
            weight_description=10.0,
            weight_store=0.0,
            weight_details=0.0,
        ),
    )
    assert title_first.retrieve("cerulean jacket", 1)[0][0] == "title-match"
    assert description_first.retrieve("cerulean jacket", 1)[0][0] == "description-match"


def test_details_weight_promotes_a_details_only_match(tmp_path):
    catalog = load(write_catalog(tmp_path, field_products()))
    retriever = TfidfRetriever(
        catalog,
        Config().with_overrides(
            weight_title=0.0,
            weight_features=0.0,
            weight_categories=0.0,
            weight_description=0.0,
            weight_store=0.0,
            weight_details=10.0,
        ),
    )
    assert retriever.retrieve("celestial dragon", 1)[0][0] == "details-match"


def test_phrase_evidence_changes_selection_before_candidate_truncation(tmp_path):
    common = {
        "features": [],
        "description": [],
        "categories": [],
        "store": "",
        "details": {},
        "rating_number": 0,
        "average_rating": 0.0,
    }
    catalog = load(write_catalog(tmp_path, [
        {**common, "parent_asin": "lexical-neighbour", "title": "moon pendant"},
        {
            **common,
            "parent_asin": "phrase-target",
            "title": "moon pendant",
            "features": ["triple constellation phrase"],
        },
    ]))
    retriever = TfidfRetriever(catalog, Config())
    plain = retriever.retrieve("moon pendant", 1)
    boosted = retriever.retrieve(
        "moon pendant",
        1,
        phrases=("triple constellation phrase",),
    )
    assert plain[0][0] == "lexical-neighbour"
    assert boosted[0][0] == "phrase-target"


def test_omitted_phrase_evidence_matches_an_empty_sequence(fake_catalog_path):
    retriever = TfidfRetriever(load(fake_catalog_path), Config())
    assert retriever.retrieve("leather belt", 3) == retriever.retrieve(
        "leather belt", 3, phrases=()
    )


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


def test_rank_explanation_matches_the_returned_order(fake_catalog_path):
    catalog = load(fake_catalog_path)
    ranker = PriorRanker(catalog, Config())
    candidates = [("B000000001", 1.0), ("B000000003", 0.4)]
    explanation = ranker.explain(candidates, Slots(), {})
    assert ranker.rank(candidates, Slots(), {}) == [
        item["parent_asin"] for item in explanation
    ]
    for item in explanation:
        expected = (
            item["retrieval"]
            + item["popularity"]
            + item["rating"]
            + item["phrase"]
        )
        assert item["total"] == expected
