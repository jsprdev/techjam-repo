"""Config, limited to the two ways it can corrupt a measurement.

A sweep that silently tunes nothing reports a wrong number for every variant,
and every decision made downstream of it is then wrong too. That is worse than
a slow agent, which is why these two survive and the trivia does not.
"""

from __future__ import annotations

import pytest

from src.config import Config


def test_unknown_field_raises_rather_than_silently_tuning_nothing():
    """A typo in a sweep definition must fail, not quietly sweep the default."""
    with pytest.raises(ValueError, match="unknown config fields"):
        Config().with_overrides(exact_phrase_bost=2.0)


def test_set_config_actually_changes_the_ranking(agent, profile):
    """Propagation that type-checks but reaches nothing would make every sweep
    row after the first a copy of the default."""
    agent.set_config(Config().with_overrides(weight_popularity=0.0))
    agent.reset("cfg", profile)
    low = [r["parent_asin"] for r in agent.respond("cfg", "leather", 1, 10)["recommendations"]]

    agent.set_config(Config().with_overrides(weight_popularity=50.0))
    agent.reset("cfg", profile)
    high = [r["parent_asin"] for r in agent.respond("cfg", "leather", 1, 10)["recommendations"]]

    assert low != high, "a 50x weight changed nothing, so it is not reaching the ranker"
    agent.set_config(Config())


def test_llm_is_off_by_default():
    """Official scoring may run with networking disabled."""
    assert Config().use_llm is False
