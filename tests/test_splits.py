"""The held-out split, limited to what protects the generalisation number.

If the split leaks or drifts, the one number we can show a judge to prove we
did not overfit the public simulator becomes meaningless, and nothing would
tell us.
"""

from __future__ import annotations

from evaluation.splits import load_sessions, split
from src.config import Config


def test_split_is_disjoint_and_loses_nothing():
    sessions = load_sessions()
    train, holdout = split(sessions)
    train_ids = {s["sample_id"] for s in train}
    holdout_ids = {s["sample_id"] for s in holdout}
    assert not (train_ids & holdout_ids), "a session appears in both halves"
    assert len(train_ids | holdout_ids) == len(sessions), "a session was dropped"
    assert len(holdout) == Config().holdout_size


def test_split_is_deterministic_across_runs():
    """Every teammate and every rerun must see the same partition."""
    sessions = load_sessions()
    assert [s["sample_id"] for s in split(sessions)[1]] == [
        s["sample_id"] for s in split(sessions)[1]
    ]
