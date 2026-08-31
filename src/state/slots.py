"""Dialogue state: what the customer has actually told us. ROLE 2.

The customer's utterances arrive as whole sentences wrapped in conversational
framing, and a large fraction of them carry no information at all. Feeding them
to retrieval verbatim, as the first version did, poisons the query: by turn six
a typical session had accumulated "I don't have an additional preference for
brand", "... for budget" and "... for color", contributing the tokens brand,
budget, colour, preference and additional to a search over a clothing catalog.

So this module's real job is separating signal from framing before anything
downstream sees it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import Config

# The ask order, measured rather than assumed.
#
# The customer only answers an attribute if one of the target's own constraint
# phrases falls into that bucket. Measured across all 200 public sessions, the
# buckets those phrases actually land in are:
#
#     feature    50.5% of constraints, answerable in 96.0% of sessions
#     material   37.8%                                76.5%
#     color       7.5%                                25.5%
#     style       2.4%                                 9.0%
#     size        1.4%                                 4.5%
#     use_case    0.5%                                 2.0%
#
# Reproduce with the bucket audit in evaluation/ask_yield.py.
ATTRIBUTES_BY_YIELD = (
    "feature",
    "material",
    "color",
    "style",
    "size",
    "use_case",
)

# Asking any of these is a guaranteed miss. Nothing the customer says is ever
# classified into them, so the reply is always "I don't have an additional
# preference" and the information the turn could have bought is lost. The first
# version of this module asked category and brand on turns 2 and 3, spending
# the two most valuable early asks on questions with no possible answer.
UNANSWERABLE = ("category", "brand", "budget")

# Matches ANY undisclosed constraint regardless of bucket, so it is the single
# most productive ask available. Kept as a fallback rather than the default
# because "tell me anything else" is a worse thing to say to a shopper than a
# specific question, and the specific questions above already reach 96%.
WILDCARD = "other"

# Utterances that carry no information. Matching these is worth more than any
# other single change in this module: they are frequent, and every one of them
# adds noise tokens to the query that actively compete with real constraints.
NO_INFORMATION = (
    re.compile(r"^i don't have an additional preference for\b", re.I),
    re.compile(r"^i don't have a preference for\b", re.I),
    re.compile(r"^those options are not quite right yet\b", re.I),
)

# Conversational framing wrapped around the payload. Everything before the
# colon is scaffolding; the constraint itself follows it.
FRAMING = (
    re.compile(r"^for that,\s*what matters is:\s*", re.I),
    re.compile(r"^a key requirement is:\s*", re.I),
    re.compile(r"^actually,\s*ignore my earlier preference\.\s*what i need is:\s*", re.I),
    re.compile(r"^i'?m looking for\s+", re.I),
)

# The opening turn names the category and may append a constraint after a full
# stop. "but I'm still exploring" is filler on the Browsing track.
# The category must be GREEDY. A non-greedy [^.,]+? with an optional suffix
# matches a single character, silently turning "Accessories Belts" into "A" and
# destroying the strongest signal in the session.
OPENING = re.compile(
    r"^i'?m looking for\s+(?P<category>[^.,]+)"
    r"(?:,\s*but i'?m still exploring)?"
    r"[.,]?\s*(?P<rest>.*)$",
    re.I,
)


@dataclass
class Slots:
    """Accumulated dialogue state for one session."""

    config: Config = field(default_factory=Config)
    turn: int = 0
    profile: dict = field(default_factory=dict)
    category: str = ""
    _phrases: list[str] = field(default_factory=list)
    _asked: list[str] = field(default_factory=list)
    _exhausted: set[str] = field(default_factory=set)
    _informative_turns: int = 0

    # -- ingest --------------------------------------------------------------

    def observe(self, user_message: str, turn: int) -> None:
        self.turn = turn
        text = (user_message or "").strip()
        if not text:
            return

        if turn == 1:
            match = OPENING.match(text)
            if match:
                self.category = match.group("category").strip()
                text = match.group("rest").strip()
                if not text:
                    return

        if any(pattern.match(text) for pattern in NO_INFORMATION):
            # The customer just told us this bucket is empty for their target.
            # Record it so the policy never spends another turn on it.
            if self._asked:
                self._exhausted.add(self._asked[-1])
            if self.config.drop_no_information:
                return

        for phrase in self._payloads(text):
            if not phrase:
                continue
            # Deliberately NOT deduplicated. A constraint the customer states
            # twice is one they care about, and repeating it in the query
            # raises its term frequency. Treating repetition as redundancy
            # measurably loses score. Config.dedupe_phrases exists to re-test
            # this rather than take it on faith.
            if self.config.dedupe_phrases and phrase in self._phrases:
                continue
            self._phrases.append(phrase)
            self._informative_turns += 1

    def _payloads(self, text: str) -> list[str]:
        """Strip framing and split a disclosure into its individual constraints.

        The simulator joins multiple constraints with "; ", and each one is
        lifted verbatim from the target product's own record, so splitting them
        apart gives ranking whole phrases it can match literally.
        """
        for pattern in FRAMING:
            stripped = pattern.sub("", text)
            if stripped != text:
                text = stripped
                break
        return [part.strip(" .;") for part in text.split(";") if part.strip(" .;")]

    # -- read ----------------------------------------------------------------

    def constraints(self) -> list[str]:
        """The disclosed constraint phrases, framing removed, most recent last."""
        return list(self._phrases)

    def to_query(self) -> str:
        parts: list[str] = []
        if self.category:
            # The opening category is the single most reliable signal we get:
            # it is the tail of the target's own category path.
            parts.append(self.category)
        parts.extend(self._phrases)
        if self.config.use_profile_tags:
            parts.extend(str(tag) for tag in (self.profile.get("preference_tags") or []))
        return " ".join(parts)

    @property
    def informative_turns(self) -> int:
        """How many turns actually told us something. Drives the ask policy."""
        return self._informative_turns

    # -- ask policy ----------------------------------------------------------

    def pick_attribute(self) -> str | None:
        """Choose the next attribute to ask about.

        Ordered by measured yield, skipping anything already asked. An attribute
        that came back empty is never retried: the customer told us that bucket
        is empty for this target, and asking again spends a turn to be told so
        twice.
        """
        for attribute in ATTRIBUTES_BY_YIELD:
            if attribute not in self._asked and attribute not in self._exhausted:
                self._asked.append(attribute)
                return attribute
        if self.config.allow_other_fallback and WILDCARD not in self._exhausted:
            self._asked.append(WILDCARD)
            return WILDCARD
        return None

    @property
    def asked(self) -> list[str]:
        return list(self._asked)
