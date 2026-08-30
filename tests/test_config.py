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
