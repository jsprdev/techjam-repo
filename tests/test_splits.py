"""The held-out split must be disjoint, stratified and reproducible.

If this drifts, the generalisation number we report to judges is meaningless,
and we would not notice.
"""

from __future__ import annotations

from evaluation.splits import describe, load_sessions, split
from src.config import Config


def test_split_is_disjoint_and_total():
    sessions = load_sessions()
    train, holdout = split(sessions)
    train_ids = {s["sample_id"] for s in train}
    holdout_ids = {s["sample_id"] for s in holdout}
    assert not (train_ids & holdout_ids), "a session appears in both halves"
    assert len(train_ids | holdout_ids) == len(sessions), "a session was dropped"


def test_holdout_is_the_configured_size():
    sessions = load_sessions()
    _, holdout = split(sessions)
    assert len(holdout) == Config().holdout_size


def test_split_is_deterministic_across_calls():
    sessions = load_sessions()
    first = [s["sample_id"] for s in split(sessions)[1]]
    second = [s["sample_id"] for s in split(sessions)[1]]
    assert first == second


def test_every_scenario_survives_into_both_halves():
    """Boundary is only ten sessions. A naive random split can empty it."""
    sessions = load_sessions()
    train, holdout = split(sessions)
    assert set(describe(train)) == set(describe(holdout)) == set(describe(sessions))


def test_a_different_seed_gives_a_different_split():
    sessions = load_sessions()
    default = {s["sample_id"] for s in split(sessions)[1]}
    other = {s["sample_id"] for s in split(sessions, Config().with_overrides(seed=999))[1]}
    assert default != other
