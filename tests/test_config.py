"""Config is the sweep harness's only lever, so typos must fail loudly."""

from __future__ import annotations

import pytest

from src.config import Config


def test_overrides_apply():
    assert Config().with_overrides(exact_phrase_boost=2.0).exact_phrase_boost == 2.0


def test_unknown_field_raises_rather_than_silently_tuning_nothing():
    with pytest.raises(ValueError, match="unknown config fields"):
        Config().with_overrides(exact_phrase_bost=2.0)


def test_config_is_immutable():
    with pytest.raises(Exception):
        Config().truncate_buying = 5


def test_to_dict_round_trips_every_field():
    config = Config().with_overrides(seed=7, use_llm=True)
    assert config.to_dict()["seed"] == 7
    assert config.to_dict()["use_llm"] is True


def test_llm_is_off_by_default():
    """The organiser may score us with networking disabled."""
    assert Config().use_llm is False


# -- config propagation, which the sweep harness depends on ------------------
# If a component keeps a stale config, a sweep silently reports the wrong
# number for every variant after the first. That is worse than a slow agent,
# because it corrupts every decision made downstream of it.


def test_set_config_reaches_every_component(agent):
    from src.config import Config

    tuned = Config().with_overrides(weight_popularity=9.0, rerank_depth=7)
    agent.set_config(tuned)
    assert agent.config is tuned
    assert agent.ranker.config is tuned
    assert agent.retriever.config is tuned


def test_set_config_actually_changes_the_ranking(agent, profile):
    """Guards against propagation that type-checks but does nothing."""
    from src.config import Config

    agent.set_config(Config().with_overrides(weight_popularity=0.0))
    agent.reset("cfg", profile)
    low = [r["parent_asin"] for r in agent.respond("cfg", "leather", 1, 10)["recommendations"]]

    agent.set_config(Config().with_overrides(weight_popularity=50.0))
    agent.reset("cfg", profile)
    high = [r["parent_asin"] for r in agent.respond("cfg", "leather", 1, 10)["recommendations"]]

    assert low != high, "a 50x popularity weight changed nothing, so it is not reaching the ranker"


def test_new_sessions_pick_up_the_new_config(agent, profile):
    from src.config import Config

    agent.set_config(Config().with_overrides(allow_other_fallback=False))
    agent.reset("fresh", profile)
    assert agent._sessions["fresh"].config.allow_other_fallback is False
