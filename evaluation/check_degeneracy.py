"""Is the ranker actually reading the query, or just returning popular items?

A high `weight_popularity` moves every metric at once, which is the shape of a
real signal and also the shape of a metric-gaming artefact. The obvious failure
mode is a ranker that has learned to ignore the conversation and return the
catalog's most reviewed products regardless of what the customer said. That
would score well here and collapse on the private set.

This distinguishes the two. It runs several unrelated queries at a range of
popularity weights and reports how much their top ten answers overlap. A
degenerate ranker converges: the sets become identical and the overlap goes to
ten. A healthy one keeps them disjoint, because popularity only reorders a
shortlist that retrieval already filtered for relevance.

    python evaluation/check_degeneracy.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running this file directly puts evaluation/ on sys.path, not the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent import Agent  # noqa: E402
from src.config import Config  # noqa: E402

# Deliberately drawn from different catalog regions, phrased the way the
# simulated customer phrases things.
QUERIES = [
    "Watches Wrist Watches Stainless Steel Band Water Resistant",
    "Jewelry Necklaces Pendant Necklaces Triple Moon Pentagram",
    "Accessories Belts Buckle closure leather 100% Leather",
    "Shoes Sandals Slides Cork sole Adjustable Flat Thong",
    "Clothing Tops Tees Blouses 100% Polyester Pull On closure",
]

WEIGHTS = (0.0, 0.4, 1.0, 2.0, 5.0, 20.0)


class _DegenerateRanker:
    """Negative control: ignores the query entirely and returns the most
    reviewed products. If the check below cannot flag this, it cannot flag
    anything, and its verdict on the real ranker means nothing."""

    def __init__(self, catalog) -> None:
        ordered = sorted(
            catalog.products, key=lambda p: float(p.get("rating_number") or 0), reverse=True
        )
        self._popular = [str(p["parent_asin"]) for p in ordered[:50]]

    def rank(self, candidates, slots, profile) -> list[str]:
        return list(self._popular)


def _probe(agent, weight: float) -> tuple[int, int]:
    """Return (distinct top-10 sets, items shared by every query)."""
    config = Config().with_overrides(weight_popularity=weight)
    agent.set_config(config)
    tops = []
    for query in QUERIES:
        agent.reset("degeneracy-probe", {})
        result = agent.respond("degeneracy-probe", query, 1, 10)
        tops.append({item["parent_asin"] for item in result["recommendations"][:10]})
    return len({frozenset(t) for t in tops}), len(set.intersection(*tops))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "techjam-conversational-search/data/catalog.jsonl",
    )
    args = parser.parse_args()

    agent = Agent(args.catalog)

    # Negative control first. A check that cannot fail is not evidence, and this
    # one is quoted in config.py as the justification for weight_popularity.
    real_ranker = agent.ranker
    agent.ranker = _DegenerateRanker(agent.catalog)
    control_distinct, control_shared = _probe(agent, 1.0)
    agent.ranker = real_ranker
    control_caught = control_distinct < len(QUERIES) or control_shared > 0
    print(
        f"negative control (a ranker that ignores the query): "
        f"{control_distinct} distinct, {control_shared} shared -> "
        f"{'flagged, so the check has teeth' if control_caught else 'NOT FLAGGED'}"
    )
    if not control_caught:
        print("\nThe check failed to flag a deliberately degenerate ranker. It proves nothing.")
        raise SystemExit(1)

    print()
    print(f"{'weight':>8}{'distinct top-10s':>20}{'shared by all':>16}   verdict")
    print("-" * 62)
    degenerate = False
    for weight in WEIGHTS:
        distinct, shared = _probe(agent, weight)
        bad = distinct < len(QUERIES) or shared > 0
        degenerate = degenerate or bad
        print(f"{weight:>8}{distinct:>20}{shared:>16}   {'DEGENERATE' if bad else 'query dependent'}")

    print()
    if degenerate:
        print("At least one weight collapses the answer toward a query-independent list.")
        raise SystemExit(1)
    print(
        "No collapse at any weight, and the control confirms a collapse would be caught.\n"
        "Popularity reorders the retrieved shortlist, it never replaces relevance."
    )


if __name__ == "__main__":
    main()
