"""One command evaluation against the official harness.

Wraps the organiser's `evaluate()` rather than reimplementing it. The submission
rules forbid modifying evaluator files, and a reimplementation would drift from
the thing that actually scores us, so we import theirs and hand it our Agent.

    python evaluation/run_eval.py                     # train split, the default
    python evaluation/run_eval.py --split holdout     # generalisation check
    python evaluation/run_eval.py --split all         # comparable to baseline
    python evaluation/run_eval.py --traces artifacts/traces.json

Prints a per-scenario table because the weakest bucket is the cheapest source of
points, and reports latency and memory because both are required disclosures.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"
# The evaluator imports `starter.agent` at module load, so the kit has to be on
# the path before we import it.
for entry in (str(REPO_ROOT), str(KIT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from evaluator.local_evaluator import catalog_index, evaluate  # noqa: E402

from evaluation.splits import DEFAULT_DATASET, describe, load_sessions, split  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.config import Config  # noqa: E402
from src.trace import SINK  # noqa: E402

DEFAULT_CATALOG = KIT / "data/catalog.jsonl"


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return usage / 1024.0 if sys.platform.startswith("linux") else usage / (1024.0 * 1024.0)


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    times = sorted(float(r["elapsed_ms"]) for r in rows)
    if not times:
        return {}
    def at(fraction: float) -> float:
        return round(times[min(len(times) - 1, int(fraction * len(times)))], 2)
    return {
        "turns": len(times),
        "mean_ms": round(sum(times) / len(times), 2),
        "p50_ms": at(0.50),
        "p95_ms": at(0.95),
        "max_ms": round(times[-1], 2),
    }


def render(result: dict[str, Any], label: str) -> str:
    lines = [
        f"\n{'=' * 66}",
        f"  {label}   n={result['sample_count']}",
        f"{'=' * 66}",
        f"  TechnicalScore   {result['recommended_technical_score']:.4f}"
        f"   (baseline 0.10671)",
        f"  HitRate@10       {result['hit_rate_at_10']:.4f}   weight 0.50",
        f"  MRR              {result['mrr']:.4f}   weight 0.30",
        f"  Efficiency       {result['efficiency']:.4f}   weight 0.20",
        f"  MTTC             {result['mttc']:.2f}",
        "",
        f"  {'scenario':<18}{'n':>5}{'hit@10':>10}{'mrr':>9}{'mttc':>8}",
        f"  {'-' * 48}",
    ]
    for name, metrics in result.get("scenario_metrics", {}).items():
        lines.append(
            f"  {name:<18}{metrics['sample_count']:>5}"
            f"{metrics['hit_rate_at_10']:>10.3f}{metrics['mrr']:>9.3f}"
            f"{metrics['mttc']:>8.2f}"
        )
    return "\n".join(lines)


def run(
    catalog_path: Path,
    dataset_path: Path,
    which: str,
    config: Config,
    trace_path: Path | None,
    detailed_traces: bool = False,
) -> dict[str, Any]:
    sessions = load_sessions(dataset_path)
    train, holdout = split(sessions, config)
    chosen = {"train": train, "holdout": holdout, "all": sessions}[which]
    print(f"[eval] split={which} sessions={len(chosen)} {describe(chosen)}")

    print(f"[eval] loading catalog and building index from {catalog_path} ...")
    build_started = time.perf_counter()
    agent = Agent(catalog_path, config)
    build_seconds = time.perf_counter() - build_started
    print(f"[eval] agent ready in {build_seconds:.1f}s")

    catalog_ids, categories, products = catalog_index(catalog_path)

    SINK.clear()
    SINK.enable(detailed=detailed_traces)
    started = time.perf_counter()
    result = evaluate(agent, chosen, catalog_ids, categories, products)
    wall_seconds = time.perf_counter() - started
    SINK.disable()

    result["run_meta"] = {
        "split": which,
        "sessions": len(chosen),
        "scenario_counts": describe(chosen),
        "index_build_seconds": round(build_seconds, 2),
        "eval_wall_seconds": round(wall_seconds, 2),
        "seconds_per_session": round(wall_seconds / max(1, len(chosen)), 3),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "latency": latency_summary(SINK.rows()),
        "config": config.to_dict(),
    }

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(SINK.by_session(), indent=2), encoding="utf-8")
        print(f"[eval] traces for {len(SINK.by_session())} sessions -> {trace_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument("--output", type=Path, default=Path("artifacts/results.json"))
    parser.add_argument("--traces", type=Path, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a Config field, repeatable, e.g. --set exact_phrase_boost=2.0",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    for pair in args.set:
        field, _, raw = pair.partition("=")
        overrides[field.strip()] = json.loads(raw)
    config = Config().with_overrides(**overrides) if overrides else Config()

    result = run(args.catalog, args.dataset, args.split, config, args.traces)
    print(render(result, f"split={args.split}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n[eval] full result -> {args.output}")
    meta = result["run_meta"]
    print(
        f"[eval] {meta['eval_wall_seconds']}s wall, "
        f"{meta['peak_memory_mb']} MB peak, "
        f"p95 turn {meta['latency'].get('p95_ms', 0)} ms"
    )


if __name__ == "__main__":
    main()
