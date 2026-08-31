"""Distilled session context and the per turn adaptation record. ROLE 2.

Spec 7.1, Personalised Context Distillation. The agent never replays raw dialogue
into retrieval. It carries the supplied profile plus the accumulated slots and
compresses them into one query string per turn, and this object is where that
compression is held and, just as importantly, where it is counted.

The counting is the point. Three of Pillar III's four runtime behaviours were
already running before this file existed, built for score and never named:
attributes retiring when the customer cannot answer them, the pipeline shape
being re-selected every turn, and the dialogue being distilled rather than
replayed. A behaviour nobody measures is a behaviour nobody can show a judge, so
every turn appends to `history` and `evaluation/self_evolution.py` reads it back.

This object holds no scoring logic. It records what the policy decided and what
the belief looked like; it never decides anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.state.slots import Slots


@dataclass(frozen=True)
class TurnRecord:
    """What the pipeline chose on one turn, and what it saw."""

    turn: int
    track: str
    track_confidence: float
    width: int
    depth: int
    entropy: float
    peak_share: float
    overloaded: bool
    committed: bool
    asked: str | None
    constraints: int
    retired: int
    raw_chars: int
    distilled_chars: int


@dataclass
class SessionContext:
    """Everything one session carries between turns."""

    session_id: str
    slots: Slots
    history: list[TurnRecord] = field(default_factory=list)
    raw_chars: int = 0

    @property
    def last_entropy(self) -> float | None:
        """Belief flatness as of the previous turn, or None on turn one.

        The router reads this rather than the current turn's entropy, because the
        track has to be chosen before retrieval runs and the belief does not
        exist until after ranking. Using the previous turn is the honest version
        of "the router reads the belief state's current entropy": it is the most
        recent belief that actually exists when the decision is made.
        """
        return self.history[-1].entropy if self.history else None

    @property
    def last_track(self) -> str:
        return self.history[-1].track if self.history else ""

    def record(self, turn: TurnRecord) -> None:
        self.history.append(turn)

    def compression(self) -> float:
        """Distilled query characters per character of raw dialogue.

        Below 1.0 means the session state is smaller than the transcript it came
        from, which is the whole claim behind distillation. It grows towards 1.0
        as the customer discloses more, because a disclosed constraint is kept
        verbatim on purpose: the simulated customer quotes the target product's
        own record, so paraphrasing it would destroy the strongest signal we get.
        """
        if not self.history or self.raw_chars <= 0:
            return 0.0
        return self.history[-1].distilled_chars / self.raw_chars
