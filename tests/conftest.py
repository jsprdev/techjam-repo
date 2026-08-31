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

# The retriever uses min_df=2, which is DOCUMENT frequency: a term must appear
# in at least two products or it is dropped from the vocabulary entirely. So a
# fixture of six unrelated items would silently lose almost every distinguishing
# term and retrieval would score on stopwords. Each item below therefore has a
# near neighbour sharing its vocabulary, which also makes ranking tests
# meaningful, since the retriever has to separate two similar products rather
# than six obviously different ones.
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

# Near neighbours. Each shares vocabulary with one item above so min_df=2 keeps
# the distinguishing terms, and each is less popular so popularity-prior tests
# have a known expected winner.
FAKE_PRODUCTS += [
    {
        "parent_asin": "B000000007",
        "title": "Leather Belt Classic Buckle",
        "features": ["100% Leather", "Buckle closure"],
        "description": ["A plain leather belt."],
        "categories": ["Accessories", "Belts"],
        "store": "PlainGoods",
        "details": {"Material": "Leather", "Color": "black"},
        "average_rating": 3.8,
        "rating_number": 40,
        "price": 20.0,
    },
    {
        "parent_asin": "B000000008",
        "title": "Stainless Steel Wrist Watch Classic",
        "features": ["Water Resistant", "Stainless Steel Band"],
        "description": ["A steel band wrist watch."],
        "categories": ["Watches", "Wrist Watches"],
        "store": "Timely",
        "details": {"Material": "Stainless Steel", "Color": "black"},
        "average_rating": 4.0,
        "rating_number": 60,
        "price": 40.0,
    },
    {
        "parent_asin": "B000000010",
        "title": "Silver Moon Pentagram Pendant Necklace Charm",
        "features": ["Triple Moon Pentagram Symbol", "alloy"],
        "description": ["A pentagram pendant necklace."],
        "categories": ["Jewelry", "Necklaces", "Pendant Necklaces"],
        "store": "MoonCraft",
        "details": {"Material": "alloy", "Color": "silver"},
        "average_rating": 3.7,
        "rating_number": 20,
        "price": 10.0,
    },
    {
        "parent_asin": "B000000009",
        "title": "Cork Footbed Sandals Summer",
        "features": ["Cork sole", "Adjustable Flat Thong"],
        "description": ["Cork footbed summer sandals."],
        "categories": ["Shoes", "Sandals", "Slides"],
        "store": "SunStep",
        "details": {"Material": "Cork", "Color": "tan"},
        "average_rating": 4.2,
        "rating_number": 30,
        "price": 22.0,
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
    """The production agent, blanket exception handler and all."""
    from src.agent import Agent

    return Agent(fake_catalog_path)


@pytest.fixture(scope="session")
def strict_agent(fake_catalog_path):
    """An agent that re-raises instead of degrading to the popularity list.

    Use this for anything asserting the pipeline WORKS. With the production
    agent, a completely dead retriever still returns a contract-legal popular
    list, so `violations(result) == []` and `result["recommendations"]` both
    hold and the assertion tests nothing.
    """
    from src.agent import Agent
    from src.config import Config

    return Agent(fake_catalog_path, Config().with_overrides(strict_errors=True))


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
