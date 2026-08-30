"""Phase 0 diagnostics: what the frozen catalog actually contains.

Run this before building retrieval or the belief layer. Spec sections 9 and 10
both depend on the numbers it prints: the attribute coverage audit decides how
much of the ask_attribute enum is usable without an offline LLM pass, and the
field survival table resolves which of the documented Amazon fields the
organiser kept.

Usage:
    python evaluation/diagnostics.py --catalog data/catalog.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Iterator

# The ten legal ask_attribute values, mapped to the raw catalog fields that
# could supply them. use_case has no source field at all and must be derived.
ATTRIBUTE_SOURCES: dict[str, tuple[str, ...]] = {
    "material": ("Material", "Fabric Type", "Outer Material"),
    "color": ("Color", "Colour"),
    "size": ("Size",),
    "style": ("Style", "Style Name"),
    "brand": ("Brand", "Brand Name"),
}

# Chars per token is a rough 4:1 for English product text. Good enough to size
# an extraction budget to the nearest order of magnitude, which is all we need.
CHARS_PER_TOKEN = 4
# A per-product cap for the trimmed cost estimate. Products past this length
# are mostly boilerplate care instructions.
TRIM_CHARS = 1200


def load_catalog(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def detail(product: dict[str, Any], *names: str) -> Any:
    """First non-empty value among `names` in the free-form details dict."""
    details = product.get("details") or {}
    for name in names:
        value = details.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def product_text(product: dict[str, Any]) -> str:
    """The text an offline extraction pass would have to read per product."""
    parts = [
        str(product.get("title") or ""),
        " ".join(str(item) for item in (product.get("features") or [])),
        " ".join(str(item) for item in (product.get("description") or [])),
        json.dumps(product.get("details") or {}),
        " ".join(str(item) for item in (product.get("categories") or [])),
    ]
    return " ".join(parts)


def report(products: list[dict[str, Any]]) -> None:
    total = len(products)
    print(f"catalog rows: {total}")

    present: collections.Counter[str] = collections.Counter()
    populated: collections.Counter[str] = collections.Counter()
    for product in products:
        for key, value in product.items():
            present[key] += 1
            if value not in (None, "", [], {}):
                populated[key] += 1

    print("\nfield survival (spec 10.4)")
    for key, count in present.most_common():
        print(f"  {key:16s} present {count / total:6.1%}  populated {populated[key] / total:6.1%}")

    detail_keys: collections.Counter[str] = collections.Counter()
    for product in products:
        details = product.get("details")
        if isinstance(details, dict):
            detail_keys.update(details.keys())
    print(f"\ndetails dict: {len(detail_keys)} distinct keys, top 12 by coverage")
    for key, count in detail_keys.most_common(12):
        print(f"  {count / total:6.1%}  {key}")

    print("\nask_attribute coverage before any LLM extraction (spec 9)")
    checks: dict[str, Callable[[dict[str, Any]], bool]] = {
        "category": lambda p: bool(p.get("categories")),
        "brand": lambda p: bool(p.get("store")) or detail(p, *ATTRIBUTE_SOURCES["brand"]) is not None,
        "feature": lambda p: bool(p.get("features")),
        "budget": lambda p: p.get("price") not in (None, "", "None"),
        "material": lambda p: detail(p, *ATTRIBUTE_SOURCES["material"]) is not None,
        "color": lambda p: detail(p, *ATTRIBUTE_SOURCES["color"]) is not None,
        "style": lambda p: detail(p, *ATTRIBUTE_SOURCES["style"]) is not None,
        "size": lambda p: detail(p, *ATTRIBUTE_SOURCES["size"]) is not None,
        "use_case": lambda p: False,
    }
    for name, check in checks.items():
        coverage = sum(1 for p in products if check(p)) / total
        print(f"  {coverage:6.1%}  {name}")

    lengths = [len(product_text(p)) for p in products]
    full = sum(lengths) / CHARS_PER_TOKEN
    trimmed = sum(min(length, TRIM_CHARS) for length in lengths) / CHARS_PER_TOKEN
    print("\noffline extraction input size (spec 3.6, 6.1)")
    print(f"  mean {statistics.mean(lengths):.0f} chars, median {statistics.median(lengths):.0f}")
    print(f"  full pass    ~{full / 1e6:.1f}M input tokens")
    print(f"  trimmed to {TRIM_CHARS} chars/product  ~{trimmed / 1e6:.1f}M input tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl", type=Path)
    args = parser.parse_args()
    report(list(load_catalog(args.catalog)))


if __name__ == "__main__":
    main()
