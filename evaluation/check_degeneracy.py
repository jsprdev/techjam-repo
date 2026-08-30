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
from pathlib import Path

from src.agent import Agent
from src.config import Config

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("techjam-conversational-search/data/catalog.jsonl"),
    )
    args = parser.parse_args()

    agent = Agent(args.catalog)
    print(f"{'weight':>8}{'distinct top-10s':>20}{'shared by all':>16}   verdict")
    print("-" * 62)
    degenerate = False
    for weight in WEIGHTS:
        config = Config().with_overrides(weight_popularity=weight)
        agent.config = config
        agent.ranker.config = config
        tops = []
        for query in QUERIES:
            agent.reset("degeneracy-probe", {})
            result = agent.respond("degeneracy-probe", query, 1, 10)
            tops.append({item["parent_asin"] for item in result["recommendations"][:10]})
        distinct = len({frozenset(t) for t in tops})
        shared = len(set.intersection(*tops))
        bad = distinct < len(QUERIES) or shared > 0
        degenerate = degenerate or bad
        print(f"{weight:>8}{distinct:>20}{shared:>16}   {'DEGENERATE' if bad else 'query dependent'}")

    print()
    if degenerate:
        print("At least one weight collapses the answer toward a query-independent list.")
        raise SystemExit(1)
    print(
        "No collapse at any weight. Popularity reorders the retrieved shortlist,\n"
        "it never replaces relevance, so the ranker is still reading the query."
    )


if __name__ == "__main__":
    main()
