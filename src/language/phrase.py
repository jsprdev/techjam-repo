"""Question wording, grounded in the candidate set. ROLE 3.

Spec 6.6. The maths picks the attribute, this writes the sentence. "Are you
thinking more casual or dressy?" rather than "Please specify style." It is small,
and it is the surface a judge actually sees in the demo video.

**This module holds no belief and decides nothing.** It receives an attribute
that `src/policy/question.py` already chose, a commit decision
`src/policy/commit.py` already made, and a shortlist that ranking already
ordered, and it returns a string. That is the layer boundary from spec section 4
and it is the reason this file cannot import from `src/state/` or `src/policy/`.

No LLM call. The organiser may score with networking disabled, so the wording
that ships has to be deterministic; grounding it in what the shortlist actually
contains gets most of what a model would give and costs nothing. `rerank.py`
holds the optional model path, and it never reaches this file.
"""

from __future__ import annotations

import re

# Values worth offering back to the customer, per attribute. Drawn from the
# catalog's own vocabulary rather than invented, so an option we name is an
# option the catalog can actually satisfy.
_VOCABULARY: dict[str, tuple[str, ...]] = {
    "material": (
        "cotton", "polyester", "nylon", "leather", "wool",
        "spandex", "silk", "rayon", "denim", "linen",
    ),
    "color": (
        "black", "white", "blue", "red", "pink", "green",
        "brown", "gray", "grey", "purple", "yellow", "orange",
    ),
    "style": ("casual", "formal", "classic", "modern", "slim", "relaxed", "vintage"),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work", "travel"),
}

# How each attribute is named in a sentence. The bare enum value reads like a
# form field, which is exactly the tone spec 6.6 asks us to avoid.
_READABLE: dict[str, str] = {
    "brand": "a particular brand",
    "budget": "a price range",
    "category": "a specific type",
    "color": "colour",
    "feature": "a particular feature",
    "material": "the material",
    "other": "anything else that matters",
    "size": "sizing",
    "style": "the style",
    "use_case": "where you will wear it",
}


def _options(attribute: str, texts: list[str], limit: int = 2) -> list[str]:
    """The attribute's values that actually occur in the current shortlist.

    Grounding matters more than it looks: offering "cotton or leather" when the
    shortlist holds neither teaches the customer nothing and invites an answer
    the catalog cannot satisfy.
    """
    vocabulary = _VOCABULARY.get(attribute)
    if not vocabulary or not texts:
        return []
    blob = " ".join(texts).lower()
    found = [
        value
        for value in vocabulary
        if re.search(rf"\b{re.escape(value)}\b", blob)
    ]
    # "gray" and "grey" are the same answer spelled twice.
    if "gray" in found and "grey" in found:
        found.remove("grey")
    return found[:limit]


def question(attribute: str | None, shortlist_texts: list[str], overloaded: bool) -> str:
    """Write this turn's message.

    `overloaded` is the over-generality cutoff from `src/policy/commit.py`. When
    it is set the sentence stops presenting a recommendation and starts asking
    the customer to narrow, which is the structured proactive clarification
    Pillar II names. The recommendations themselves are still returned either
    way; only the framing moves.
    """
    if attribute is None:
        return "Here are the closest matches I found."

    readable = _READABLE.get(attribute, attribute.replace("_", " "))
    options = _options(attribute, shortlist_texts)

    if options:
        choices = " or ".join(options)
        if overloaded:
            return (
                f"I am still seeing a very wide range here. To narrow it down, "
                f"were you thinking {choices}?"
            )
        return f"Here are some options. Were you thinking {choices}, or something else?"

    if overloaded:
        return (
            f"That could match a lot of things right now. Do you have a "
            f"preference on {readable}?"
        )
    return f"Here are some options. Any preference on {readable}?"
