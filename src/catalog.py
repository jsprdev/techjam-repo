"""Read-only access to the frozen product catalog.

The rules make the catalog strictly read only: no structural mutation and no
mock ASIN injection. Nothing in this module writes, and derived artefacts belong
in `artifacts/`, never back into `data/`.

Loading 50,000 products costs about a second and 200 MB, so the agent builds one
instance in `__init__` and every module shares it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Fields that survived into the frozen catalog. Verified in phase0-findings.md:
# main_category, bought_together, images and videos were all dropped, so do not
# reference them.
TEXT_FIELDS = ("title", "features", "description", "categories", "store")


@dataclass(frozen=True)
class Catalog:
    """Products in a stable order, plus an index from parent_asin to position."""

    products: list[dict[str, Any]]
    index: dict[str, int]

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.products)

    @property
    def asins(self) -> list[str]:
        return [str(p["parent_asin"]) for p in self.products]

    def get(self, parent_asin: str) -> dict[str, Any] | None:
        position = self.index.get(parent_asin)
        return None if position is None else self.products[position]

    def text(self, parent_asin: str) -> str:
        product = self.get(parent_asin)
        return "" if product is None else product_text(product)


def product_text(product: dict[str, Any]) -> str:
    """Concatenate the retrievable text of one product.

    Kept identical in spirit to the evaluator's own `searchable_text`, so that
    what we index is what the simulated customer quotes from.
    """
    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    details = product.get("details")
    if isinstance(details, dict):
        parts.extend(f"{key} {item}" for key, item in details.items())
    return " ".join(parts).strip()


def load(catalog_path: str | Path) -> Catalog:
    """Load the catalog from JSONL. Raises if the file is missing or short."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(
            f"catalog not found at {path}. Download catalog.jsonl.gz from the "
            "participant kit release and decompress it into data/."
        )
    products = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    index = {str(p["parent_asin"]): i for i, p in enumerate(products)}
    return Catalog(products=products, index=index)
