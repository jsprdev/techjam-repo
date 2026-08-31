"""The offline artefact must never be able to break the graded run.

It is optional by construction: a missing, malformed or partial artefact has to
degrade to no signal rather than to an exception or a distorted ranking. That is
what lets the LLM stage exist without putting the submission at risk.
"""

from __future__ import annotations

from src import semantic


def test_a_missing_artefact_degrades_to_no_signal():
    assert len(semantic.load("/nonexistent/path.json")) == 0


def test_malformed_json_degrades_rather_than_raising(tmp_path):
    bad = tmp_path / "prior.json"
    bad.write_text("{not json", encoding="utf-8")
    assert len(semantic.load(bad)) == 0


def test_entries_outside_the_valid_range_are_dropped(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        '{"A": {"appeal": 0.5}, "B": {"appeal": 9.9}, "C": {"appeal": "high"}, "D": 3}',
        encoding="utf-8",
    )
    prior = semantic.load(path)
    assert set(prior.appeal) == {"A"}, "an out-of-range or non-numeric appeal was kept"


def test_an_unenriched_product_takes_the_supplied_default():
    """A partial artefact must not penalise what it does not cover, or coverage
    becomes a ranking signal in its own right."""
    prior = semantic.SemanticPrior(appeal={"A": 0.9}, use_case={}, formality={})
    assert prior.appeal_of("A", 0.5) == 0.9
    assert prior.appeal_of("MISSING", 0.5) == 0.5


def test_the_shipped_artefact_is_actually_shipped():
    """The artefact must be IN THE REPOSITORY, not just on someone's disk.

    artifacts/ is gitignored for sweep and eval output, which silently excluded
    this file too. A clean clone then had no LLM stage at all: the loader
    degraded to no signal exactly as designed, so nothing failed loudly and the
    score quietly dropped back to its without-the-stage value. Caught by cloning
    the repository fresh and running the README steps.
    """
    from pathlib import Path

    from src import semantic

    path = Path(semantic.DEFAULT_PATH)
    assert path.exists(), f"{path} is missing from the checkout"
    assert len(semantic.load(path)) > 0, "the shipped semantic prior loaded as empty"


def test_the_agent_runs_with_the_artefact_removed(fake_catalog_path, monkeypatch):
    monkeypatch.setattr(semantic, "DEFAULT_PATH", "/nonexistent/path.json")
    from src.agent import Agent
    from src.response import violations

    built = Agent(fake_catalog_path)
    built.reset("s", {})
    result = built.respond("s", "leather belt", 1, 10)
    assert violations(result) == []
    assert result["recommendations"]
