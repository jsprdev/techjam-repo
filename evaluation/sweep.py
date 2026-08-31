"""Config sweep harness.

Three people tuning constants by hand, each rerunning the whole suite, is how a
five day project loses a day. This runs a set of config variants back to back
and prints them ranked, so a tuning decision is a table rather than an argument.

    python evaluation/sweep.py --grid exact_phrase_boost=1,2,4 --limit 60
    python evaluation/sweep.py --grid truncate_buying=100,200,400 slot_decay=0.8,1.0

Always sweeps the train split. Never the held-out slice: the moment a held-out
number informs a decision it stops measuring generalisation.

Cost note: a full 160 session run takes about four minutes, so a nine cell grid
is most of an hour. Use --limit while exploring, then confirm the winner on the
full split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"
# Same bootstrap as run_eval: running this file directly puts evaluation/ on the
# path rather than the repo root, and the evaluator imports starter.agent at load.
for _entry in (str(REPO_ROOT), str(KIT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from evaluator.local_evaluator import catalog_index, evaluate  # noqa: E402

from evaluation.run_eval import DEFAULT_CATALOG, latency_summary  # noqa: E402
from evaluation.splits import DEFAULT_DATASET, load_sessions, split  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.config import Config  # noqa: E402
from src.trace import SINK  # noqa: E402

# Config fields that change the retrieval index and therefore force an expensive
# rebuild. Field matrices are built independently of their runtime blending
# weights, so every current retrieval tuning parameter can reuse one index.
INDEX_FIELDS = frozenset()


def parse_grid(pairs: list[str]) -> dict[str, list[Any]]:
    """Turn ['a=1,2', 'b=x'] into {'a': [1, 2], 'b': ['x']}."""
    grid: dict[str, list[Any]] = {}
    for pair in pairs:
        field, _, raw = pair.partition("=")
        values: list[Any] = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            try:
                values.append(json.loads(chunk))
            except json.JSONDecodeError:
                values.append(chunk)
        grid[field.strip()] = values
    return grid


def variants(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    fields = sorted(grid)
    return [
        dict(zip(fields, combination))
        for combination in itertools.product(*(grid[f] for f in fields))
    ]


def run_one(
    overrides: dict[str, Any],
    sessions: list[dict[str, Any]],
    catalog_path: Path,
    index_cache: dict[tuple, Agent],
    catalog_bits: tuple,
) -> dict[str, Any]:
    config = Config().with_overrides(**overrides)
    key = tuple(sorted((f, getattr(config, f)) for f in INDEX_FIELDS))
    agent = index_cache.get(key)
    if agent is None:
        agent = Agent(catalog_path, config)
        index_cache[key] = agent
    else:
        # Reuse the built index but adopt the new tuning parameters. Agent owns
        # propagation so this cannot silently miss a component.
        agent.set_config(config)

    SINK.clear()
    SINK.enable()
    started = time.perf_counter()
    result = evaluate(agent, sessions, *catalog_bits)
    elapsed = time.perf_counter() - started
    SINK.disable()

    return {
        "overrides": overrides,
        "technical_score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "wall_seconds": round(elapsed, 1),
        "latency": latency_summary(SINK.rows()),
        "scenario_metrics": result.get("scenario_metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", nargs="*", default=[], metavar="FIELD=V1,V2")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N train sessions")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sweep.json"))
    args = parser.parse_args()

    grid = parse_grid(args.grid)
    cells = variants(grid)
    train, _ = split(load_sessions(args.dataset))
    if args.limit:
        # Say so out loud. A silently truncated sweep reads as full coverage.
        print(f"[sweep] LIMITED to the first {args.limit} of {len(train)} train sessions")
        train = train[: args.limit]

    print(f"[sweep] {len(cells)} variants over {len(train)} sessions")
    catalog_bits = catalog_index(args.catalog)
    index_cache: dict[tuple, Agent] = {}
    rows: list[dict[str, Any]] = []
    for position, overrides in enumerate(cells, 1):
        label = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "defaults"
        print(f"[sweep] {position}/{len(cells)}  {label}")
        row = run_one(overrides, train, args.catalog, index_cache, catalog_bits)
        rows.append(row)
        print(f"          score {row['technical_score']:.4f}  hit {row['hit_rate_at_10']:.3f}"
              f"  mrr {row['mrr']:.3f}  mttc {row['mttc']:.2f}  ({row['wall_seconds']}s)")

    rows.sort(key=lambda r: -r["technical_score"])
    print(f"\n{'score':>8}{'hit@10':>9}{'mrr':>8}{'mttc':>8}   config")
    print("-" * 68)
    for row in rows:
        label = ", ".join(f"{k}={v}" for k, v in row["overrides"].items()) or "defaults"
        print(f"{row['technical_score']:>8.4f}{row['hit_rate_at_10']:>9.3f}"
              f"{row['mrr']:>8.3f}{row['mttc']:>8.2f}   {label}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sessions": len(train), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n[sweep] -> {args.output}")


if __name__ == "__main__":
    main()
