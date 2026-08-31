"""Diagnose whether retrieval or reranking limits MRR.

Uses the official evaluator unchanged, then joins its per-session results with
trace-only candidate and score-component data. The target is known only in this
offline evaluation script, never by the Agent at response time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_eval import DEFAULT_CATALOG, run
from evaluation.splits import DEFAULT_DATASET, load_sessions, split
from src.config import Config
from src.trace import SINK


def position(items: list[dict[str, Any]], target: str) -> int | None:
    for index, item in enumerate(items, 1):
        if item["parent_asin"] == target:
            return index
    return None


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def diagnose(
    samples: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    trace_sessions = list(SINK.by_session().values())
    evaluated = result["sessions"]
    if len(samples) != len(evaluated) or len(samples) != len(trace_sessions):
        raise RuntimeError("evaluation results and traces are not aligned")

    rows: list[dict[str, Any]] = []
    for sample, session, turns in zip(samples, evaluated, trace_sessions):
        target = str(sample["ground_truth"]["parent_asin"])
        hit_turn = session["first_hit_turn"]
        if hit_turn is None:
            continue
        turn = next(item for item in turns if item["turn"] == hit_turn)
        extra = turn["extra"]
        retrieval = extra["retrieval_shortlist"]
        ranking = extra["ranking"]
        retrieval_rank = position(retrieval, target)
        final_rank = position(ranking, target)
        target_score = next((item for item in ranking if item["parent_asin"] == target), None)
        leader = ranking[0] if ranking else None
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "turn": hit_turn,
                "retrieval_rank": retrieval_rank,
                "final_rank": final_rank,
                "rank_delta": None if retrieval_rank is None or final_rank is None else retrieval_rank - final_rank,
                "leader_advantage": (
                    None
                    if target_score is None or leader is None
                    else {
                        name: round(float(leader[name]) - float(target_score[name]), 4)
                        for name in (
                            "retrieval",
                            "popularity",
                            "rating",
                            "phrase",
                            "total",
                        )
                    }
                ),
            }
        )

    deltas = [float(row["rank_delta"]) for row in rows if row["rank_delta"] is not None]
    displaced = [row for row in rows if row["final_rank"] and row["final_rank"] > 1]
    advantages = {
        name: mean([float(row["leader_advantage"][name]) for row in displaced if row["leader_advantage"]])
        for name in (
            "retrieval",
            "popularity",
            "rating",
            "phrase",
            "total",
        )
    }
    return {
        "converted_sessions": len(rows),
        "mean_retrieval_rank_on_hit": mean([float(row["retrieval_rank"]) for row in rows if row["retrieval_rank"]]),
        "mean_final_rank_on_hit": mean([float(row["final_rank"]) for row in rows if row["final_rank"]]),
        "promoted_by_reranker": sum(delta > 0 for delta in deltas),
        "demoted_by_reranker": sum(delta < 0 for delta in deltas),
        "unchanged_by_reranker": sum(delta == 0 for delta in deltas),
        "mean_rank_delta": mean(deltas),
        "mean_leader_advantage_when_target_not_first": advantages,
        "sessions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rank_diagnostics.json"))
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()

    overrides = {
        field.strip(): json.loads(raw)
        for pair in args.set
        for field, _, raw in [pair.partition("=")]
    }
    config = Config().with_overrides(**overrides) if overrides else Config()
    train, _ = split(load_sessions(args.dataset), config)
    result = run(
        args.catalog,
        args.dataset,
        "train",
        config,
        trace_path=None,
        detailed_traces=True,
    )
    report = diagnose(train, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sessions"}, indent=2))
    print(f"[diagnostics] -> {args.output}")


if __name__ == "__main__":
    main()
