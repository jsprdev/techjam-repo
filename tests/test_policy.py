"""Belief, routing, override and the commit policy.

These are the parts where a silent bug is expensive: an override that erases
instead of demoting, a decay that underflows to zero, an entropy that is always
1.0 and so makes the cutoff a dead branch. Each test below is written to fail if
the behaviour it names stops happening, not merely to exercise the code.
"""

from __future__ import annotations

import pytest

from src.catalog import load
from src.config import Config
from src.policy.commit import decide
from src.policy.intent import BROWSING, BUYING, buying_score, route
from src.rank import PriorRanker
from src.state import Belief, Slots

BUYING_OPENER = "I'm looking for Accessories Belts. A key requirement is: leather."
BROWSING_OPENER = "I'm looking for Accessories Belts, but I'm still exploring."
OVERRIDE_MESSAGE = "Actually, ignore my earlier preference. What I need is: Rubber sole."


# -- belief ------------------------------------------------------------------


def test_entropy_is_one_when_nothing_separates():
    assert Belief(asins=list("abcd"), scores=[1.0] * 4).entropy() == pytest.approx(1.0)


def test_entropy_is_zero_when_one_candidate_dominates():
    # Not exactly zero: `mass` adds an epsilon so an all-identical distribution
    # cannot divide by zero, and that epsilon leaves a residue here.
    entropy = Belief(asins=list("abcd"), scores=[9.0, 0.0, 0.0, 0.0]).entropy()
    assert entropy == pytest.approx(0.0, abs=1e-9)


def test_entropy_ignores_a_shared_offset():
    """Every candidate carries the same popularity floor. If that floor moved the
    entropy, the measure would track catalog popularity rather than evidence."""
    low = Belief(asins=list("abcd"), scores=[3.0, 2.0, 1.0, 1.0])
    high = Belief(asins=list("abcd"), scores=[103.0, 102.0, 101.0, 101.0])
    assert low.entropy() == pytest.approx(high.entropy())


def test_the_unscored_tail_is_kept_rather_than_filtered():
    """Spec 5.4: demote, never remove. A dropped candidate is unrecoverable."""
    belief = Belief(asins=["a", "b"], scores=[2.0, 1.0], tail=["y", "z"])
    assert belief.ranking() == ["a", "b", "y", "z"]
    assert len(belief) == 4


def test_an_empty_belief_does_not_raise():
    empty = Belief(asins=[], scores=[])
    assert empty.ranking() == [] and empty.entropy() == 0.0 and empty.peak_share() == 0.0


# -- slot decay and override -------------------------------------------------


def test_decay_weights_recent_constraints_above_old_ones():
    slots = Slots(config=Config().with_overrides(slot_decay=0.5))
    slots.observe("I'm looking for Accessories Belts. Buckle closure", 1)
    slots.observe("For that, what matters is: leather.", 3)
    weights = slots.constraint_weights()
    assert weights[1] > weights[0], "the newer constraint must outweigh the older one"
    assert weights == pytest.approx([0.25, 1.0])


def test_decay_of_one_leaves_every_constraint_equal():
    slots = Slots(config=Config().with_overrides(slot_decay=1.0))
    slots.observe("I'm looking for Accessories Belts. Buckle closure", 1)
    slots.observe("For that, what matters is: leather.", 5)
    assert slots.constraint_weights() == [1.0, 1.0]


def test_an_override_demotes_the_old_constraint_and_keeps_it():
    """Measured in evaluation/override_audit.py: in 30 of 30 public override
    sessions the superseded preference is still a property of the target, so
    erasing it deletes true evidence."""
    slots = Slots(config=Config().with_overrides(slot_decay=1.0, override_demote=0.4))
    slots.observe("I'm looking for Shoes Slippers. Plush mule style", 1)
    slots.observe(OVERRIDE_MESSAGE, 2)
    assert slots.constraints() == ["Plush mule style", "Rubber sole"]
    assert slots.constraint_weights() == pytest.approx([0.4, 1.0])
    assert slots.pivot_turns == [2]


def test_override_demote_of_zero_is_literal_erasure():
    slots = Slots(config=Config().with_overrides(slot_decay=1.0, override_demote=0.0))
    slots.observe("I'm looking for Shoes Slippers. Plush mule style", 1)
    slots.observe(OVERRIDE_MESSAGE, 2)
    assert slots.constraint_weights() == pytest.approx([0.0, 1.0])


def test_a_pivot_is_detected_without_the_simulator_wording():
    slots = Slots()
    slots.observe("I'm looking for Shoes Slippers. Plush mule style", 1)
    slots.observe("Actually, I need a rubber sole instead.", 2)
    assert slots.pivot_turns == [2]


def test_an_unanswerable_attribute_is_retired_for_the_session():
    """Runtime reliability reweighting, spec 7.1. Asking twice wastes the turn."""
    slots = Slots()
    slots.observe("I'm looking for Accessories Belts, but I'm still exploring.", 1)
    first = slots.pick_attribute()
    slots.observe(f"I don't have an additional preference for {first}.", 2)
    assert first in slots.retired_attributes
    assert slots.pick_attribute() != first


def test_zero_weights_degrade_rather_than_divide_by_zero(fake_catalog_path):
    """agent.py must never raise, so neither may the ranker beneath it."""
    config = Config().with_overrides(slot_decay=0.0, override_demote=0.0)
    slots = Slots(config=config)
    slots.observe("I'm looking for Accessories Belts. Buckle closure", 1)
    slots.observe("For that, what matters is: leather.", 9)
    ranker = PriorRanker(load(fake_catalog_path), config)
    assert ranker.rank([("B000000001", 1.0), ("B000000002", 0.5)], slots, {})


# -- routing -----------------------------------------------------------------


def test_a_stated_requirement_routes_to_buying():
    assert route(BUYING_OPENER, 1, None, Config()).name == BUYING


def test_an_open_ended_opener_routes_to_browsing():
    assert route(BROWSING_OPENER, 0, None, Config()).name == BROWSING


def test_accumulated_constraints_convert_browsing_into_buying():
    """Spec 5.1: routing is per turn. A customer who opens vague and then states
    two constraints is Buying by turn three, and the pipeline has to follow."""
    config = Config()
    assert route("Those options are not quite right yet.", 0, None, config).name == BROWSING
    assert route("For that, what matters is: leather.", 2, None, config).name == BUYING


def test_routing_survives_without_the_simulator_openers():
    """The opener patterns are fitted to this harness. The general cues are not,
    and evaluation/intent_audit.py reports 0.995 turn one accuracy without them."""
    config = Config()
    assert route(BROWSING_OPENER, 0, None, config, openers=()).name == BROWSING
    assert route("I need a leather belt.", 1, None, config, openers=()).name == BUYING


def test_a_flat_belief_pushes_the_turn_back_towards_browsing():
    config = Config()
    peaked = buying_score("Those options are not quite right yet.", 1, 0.60, config)
    flat = buying_score("Those options are not quite right yet.", 1, 0.99, config)
    assert flat < peaked


def test_the_routing_score_stays_in_range():
    config = Config()
    assert 0.0 <= buying_score(BUYING_OPENER, 99, 0.0, config) <= 1.0
    assert 0.0 <= buying_score(BROWSING_OPENER, 0, 1.0, config) <= 1.0


# -- commit policy -----------------------------------------------------------


def test_the_cutoff_fires_on_a_flat_belief_and_narrows_the_shortlist():
    config = Config().with_overrides(flat_belief_entropy=0.5, overload_depth=25)
    decision = decide(Belief(asins=list("abcd"), scores=[1.0] * 4), 200, config)
    assert decision.overloaded and not decision.commit
    assert decision.depth == 25


def test_a_decided_belief_commits_at_the_default_depth():
    config = Config().with_overrides(flat_belief_entropy=0.9, commit_peak_share=0.2)
    decision = decide(Belief(asins=list("abcd"), scores=[9.0, 1.0, 0.0, 0.0]), 200, config)
    assert decision.commit and not decision.overloaded
    assert decision.depth == 200


def test_an_empty_belief_never_reports_an_overloaded_pool():
    """No candidates is a retrieval failure, not an over-general query. Treating
    it as overload would fire the cutoff on exactly the turns it cannot help."""
    assert not decide(Belief(asins=[], scores=[]), 200, Config()).overloaded
