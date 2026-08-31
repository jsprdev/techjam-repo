"""Shared fixtures.

Most tests run against a tiny synthetic catalog rather than the real 50,000
products, because building the real index takes twenty seconds and a test suite
nobody waits for is a test suite nobody runs. The handful of tests that genuinely
need the real thing are marked `slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CATALOG = REPO_ROOT / "techjam-conversational-search/data/catalog.jsonl"

# Enough products, with enough shared vocabulary, that TF-IDF has something to
# discriminate on. min_df=2 in the retriever means a term must appear twice.
FAKE_PRODUCTS = [
    {
        "parent_asin": "B000000001",
        "title": "Leather Belt with Buckle Closure",
        "features": ["100% Leather", "Buckle closure", "Two Row Stitch"],
        "description": ["A rustic handmade full grain leather belt."],
        "categories": ["Accessories", "Belts"],
        "store": "HideAndDrink",
        "details": {"Material": "Leather", "Color": "brown"},
        "average_rating": 4.5,
        "rating_number": 1200,
        "price": 45.0,
    },
    {
        "parent_asin": "B000000002",
        "title": "Stainless Steel Wrist Watch",
        "features": ["Water Resistant", "Stainless Steel Band", "3 Year Battery"],
        "description": ["A classic wrist watch with a steel band."],
        "categories": ["Watches", "Wrist Watches"],
        "store": "Casio",
        "details": {"Material": "Stainless Steel", "Color": "silver"},
        "average_rating": 4.7,
        "rating_number": 8000,
        "price": 30.0,
    },
    {
        "parent_asin": "B000000003",
        "title": "Cotton Short Sleeve T-Shirt",
        "features": ["60% Cotton, 40% Polyester", "Pull On closure"],
        "description": ["A soft cotton t-shirt for everyday wear."],
        "categories": ["Men", "Shirts", "T-Shirts"],
        "store": "BasicWear",
        "details": {"Material": "Cotton", "Color": "black"},
        "average_rating": 4.1,
        "rating_number": 340,
        "price": 15.0,
    },
    {
        "parent_asin": "B000000004",
        "title": "Cork Footbed Slide Sandals",
        "features": ["Cork sole", "Adjustable Flat Thong"],
        "description": ["Summer slides with a cork footbed."],
        "categories": ["Shoes", "Sandals", "Slides"],
        "store": "HEVA",
        "details": {"Material": "Cork", "Color": "brown"},
        "average_rating": 3.9,
        "rating_number": 90,
        "price": 25.0,
    },
    {
        "parent_asin": "B000000005",
        "title": "Silver Pendant Necklace with Moon Charm",
        "features": ["Triple Moon Pentagram Symbol", "alloy"],
        "description": ["A pendant necklace with a moon charm."],
        "categories": ["Jewelry", "Necklaces", "Pendant Necklaces"],
        "store": "PurpleWhale",
        "details": {"Material": "alloy", "Color": "silver"},
        "average_rating": 4.0,
        "rating_number": 55,
        "price": 12.0,
    },
    {
        "parent_asin": "B000000006",
        "title": "Leather Wallet with Coin Pocket",
        "features": ["100% Leather", "Coin pocket"],
        "description": ["A compact leather wallet."],
        "categories": ["Accessories", "Wallets"],
        "store": "HideAndDrink",
        "details": {"Material": "Leather", "Color": "black"},
        "average_rating": 4.4,
        "rating_number": 700,
        "price": 35.0,
    },
]


@pytest.fixture(scope="session")
def fake_catalog_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("catalog") / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for product in FAKE_PRODUCTS:
            handle.write(json.dumps(product) + "\n")
    return path


@pytest.fixture(scope="session")
def agent(fake_catalog_path):
    from src.agent import Agent

    return Agent(fake_catalog_path)


@pytest.fixture
def profile() -> dict:
    return {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 5.0,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort"],
        "summary": "Prior purchases emphasize fit and comfort.",
    }


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real 50,000 product catalog")
