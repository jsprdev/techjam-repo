"""Offline LLM pass: derive a semantic prior the runtime consumes deterministically.

WHY THIS EXISTS, AND WHY IT IS OFFLINE
--------------------------------------
Pillar I names the pipeline base as "Multi-Route Retrieval then LLM Semantic
Ranking". `docs/submission_rules.md` warns that official scoring may run with
network access disabled under CPU, memory and timeout limits. Those two
requirements conflict if the model call sits on the turn path.

This resolves the conflict by moving the model call off the turn path entirely.
The LLM runs once, here, offline. It writes `artifacts/semantic_prior.json`.
The agent reads that file at construction and never calls anything at run time,
so the graded run is a pure lookup and survives a network-disabled environment.

WHAT IT ASKS THE MODEL
----------------------
Not attributes the catalog already carries, and not anything a regex could
recover. Three judgments that need world knowledge:

  appeal      Would a typical shopper actually buy this, versus scroll past it?
              The evaluation target is a REAL purchase, so this is the closest
              query-independent proxy for the thing being scored. Raw
              rating_number already captures popularity; this is meant to
              capture desirability that the count alone misses, for instance a
              well-reviewed but niche novelty item.
  use_case    Absent from the catalog entirely: measured 0% coverage in
              phase0-findings.md. Nothing to normalise, it has to be derived.
  formality   Same, and it is what separates two otherwise identical dresses.

COST
----
The Batches API runs asynchronously at 50% of standard rates, which is the
right trade for a one-off offline pass with no latency requirement. Trimming
each product to its title, categories and first two features keeps the input
near 120 tokens; the structured reply is near 60 output tokens.

At 50,000 products that is roughly 6M input and 3M output tokens. On Claude
Haiku 4.5 ($1.00/$5.00 per MTok) batched, about $10.50. On Claude Sonnet 5
($2.00/$10.00) batched, about $21. Run --limit 500 first and check the
artefact actually moves the eval before paying for the full catalog.

USAGE
-----
    export ANTHROPIC_API_KEY=...          # the organiser provides no key
    python offline/build_semantic_prior.py --limit 500     # prototype
    python offline/build_semantic_prior.py                 # full catalog

Writes `artifacts/semantic_prior.json`. The artefact is committed; the key is
never committed and is read from the environment only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.catalog import load  # noqa: E402

DEFAULT_CATALOG = REPO_ROOT / "techjam-conversational-search/data/catalog.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/semantic_prior.json"

# Batched pricing is half the standard rate. Update from the model table if the
# published rates change; this only drives the printed estimate.
PRICE_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}

SYSTEM = """You judge clothing and accessory products for a shopping assistant.

For each product, return three fields:

appeal: 0.0 to 1.0. Would a typical shopper browsing this category actually buy
this item, or scroll past it? Judge mainstream desirability, not quality. A
plain well-made black belt scores high. A novelty slogan t-shirt or a highly
specific costume piece scores low even if it is well reviewed.

use_case: up to three short lowercase tags for the occasions or activities this
suits. Examples: everyday, work, formal, beach, gym, hiking, gift, party,
winter. Use [] if nothing specific applies.

formality: exactly one of casual, smart_casual, formal, athletic, or unknown.

Judge only from the text given. Do not guess beyond it."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["appeal", "use_case", "formality"],
    "properties": {
        "appeal": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "use_case": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "formality": {
            "type": "string",
            "enum": ["casual", "smart_casual", "formal", "athletic", "unknown"],
        },
    },
}


def product_prompt(product: dict[str, Any]) -> str:
    """Trim to what the judgment needs. Sending the full record triples cost."""
    features = [str(f) for f in (product.get("features") or [])][:2]
    categories = " > ".join(str(c) for c in (product.get("categories") or [])[-3:])
    return (
        f"Title: {product.get('title', '')}\n"
        f"Category: {categories}\n"
        f"Features: {'; '.join(features)}"
    )


def build_requests(products: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    return [
        Request(
            custom_id=str(product["parent_asin"]),
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=256,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": product_prompt(product)}],
            ),
        )
        for product in products
    ]


def estimate_cost(products: list[dict[str, Any]], model: str) -> None:
    chars = sum(len(product_prompt(p)) + len(SYSTEM) for p in products)
    input_tokens = chars / 4
    output_tokens = 60 * len(products)
    rate_in, rate_out = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["claude-haiku-4-5"])
    # Batches run at 50% of standard rates.
    cost = (input_tokens / 1e6 * rate_in + output_tokens / 1e6 * rate_out) * 0.5
    print(
        f"[prior] {len(products)} products, about {input_tokens/1e6:.1f}M input and "
        f"{output_tokens/1e6:.1f}M output tokens"
    )
    print(f"[prior] estimated batched cost on {model}: ${cost:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="enrich only the N most reviewed products. Targets skew popular, so "
        "this covers a useful slice cheaply: the top 500 hold 126 of the 200 "
        "public targets",
    )
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    catalog = load(args.catalog)
    products = list(catalog.products)
    if args.limit:
        products.sort(key=lambda p: float(p.get("rating_number") or 0), reverse=True)
        products = products[: args.limit]
        print(f"[prior] LIMITED to the {args.limit} most reviewed products")

    estimate_cost(products, args.model)
    if args.estimate_only:
        return

    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=build_requests(products, args.model))
    print(f"[prior] batch {batch.id} submitted, polling")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"[prior]   {batch.processing_status} ...")
        time.sleep(30)

    prior: dict[str, Any] = {}
    failures = 0
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type != "succeeded":
            failures += 1
            continue
        text = next(
            (b.text for b in entry.result.message.content if b.type == "text"), ""
        )
        try:
            prior[entry.custom_id] = json.loads(text)
        except json.JSONDecodeError:
            failures += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prior, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[prior] {len(prior)} enriched, {failures} failed -> {args.output}")


if __name__ == "__main__":
    main()
