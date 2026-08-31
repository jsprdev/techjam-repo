"""Train and held-out splits over the 200 public sessions.

The held-out slice exists so we find out, before the organiser does, whether we
fitted the public simulator instead of the task. The private set has different
users and different target products, so a score that only holds on sessions we
tuned against is not a score.

Nobody tunes against the held-out slice. Not once, not to check. The moment it
informs a decision it stops measuring generalisation, and its value to a judge
is precisely that it never did.

The split is deterministic given `Config.seed`, so every teammate and every
rerun sees the same partition.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.config import Config

DEFAULT_DATASET = Path("techjam-conversational-search/data/public_set.jsonl")


def load_sessions(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"session file not found at {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def split(
    sessions: list[dict[str, Any]],
    config: Config | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (train, holdout), stratified by scenario type.

    Stratifying matters because the buckets are wildly uneven: 80 buying, 80
    browsing, 30 intent_override, 10 boundary. A naive random 40 could take half
    the boundary sessions and leave that bucket unmeasurable in both halves.
    """
    config = config or Config()
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        by_scenario.setdefault(str(session.get("scenario_type", "unknown")), []).append(session)

    holdout_ids: set[str] = set()
    total = len(sessions)
    for scenario, rows in sorted(by_scenario.items()):
        # Proportional share of the holdout budget, at least one per bucket so
        # no scenario is invisible in the held-out report.
        share = max(1, round(config.holdout_size * len(rows) / total))
        rng = random.Random(f"{config.seed}:{scenario}")
        chosen = rng.sample(rows, min(share, len(rows)))
        holdout_ids.update(str(row["sample_id"]) for row in chosen)

    train = [s for s in sessions if str(s["sample_id"]) not in holdout_ids]
    holdout = [s for s in sessions if str(s["sample_id"]) in holdout_ids]
    return train, holdout


def describe(sessions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for session in sessions:
        key = str(session.get("scenario_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
