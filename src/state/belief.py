"""Belief over catalog items: how likely each candidate is the target. ROLE 2.

Spec 5.3. The belief and the slot state are two different objects and both are
kept. Slots are the explicit accumulated constraints, the thing that gets shown,
logged and reasoned about. The belief is the item level distribution those slots
feed into, and it is what the commit policy actually reads.

Two rules from the spec bind here specifically:

1. Soft reweighting, never hard filtering (5.4). Evidence demotes a non matching
   item, it never removes it. `tail` exists for exactly that: candidates past the
   rerank depth keep their place at the bottom of the ranking rather than being
   dropped, so a constraint that was wrong about the metadata stays recoverable.
2. The language layer never holds this object and never decides when to commit
   (section 4). Only `src/policy/` reads `entropy`. Language models are badly
   calibrated about their own confidence and calibration is the whole point of
   this layer.

This module does not decide what the evidence is. `src/rank/` blends retrieval
score, priors and phrase evidence and hands the result here; this file only
knows how to normalise, order and measure the distribution that comes out. That
split is what keeps the file small and the scoring in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Entropy is measured over the top this many candidates.
#
# Not the whole pool. Past a few dozen items the distribution is made of
# candidates no policy would ever act on, and their near identical scores drag
# every session's entropy to 1.0 whether or not the top of the list is decisive.
# 50 is five times the ten items the customer actually sees, which is wide
# enough that the measure is not just a restatement of the answer.
ENTROPY_SUPPORT = 50


@dataclass(frozen=True)
class Belief:
    """A scored, ordered view over the current candidate pool.

    `asins` and `scores` are aligned and already sorted best first. `tail` holds
    the candidates retrieval returned but ranking never scored, kept so that
    `ranking()` is a permutation of the pool rather than a subset of it.
    """

    asins: list[str]
    scores: list[float]
    tail: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.asins) + len(self.tail)

    def ranking(self) -> list[str]:
        """Every candidate, best first, scored items ahead of the unscored tail."""
        return [*self.asins, *self.tail]

    def top(self, k: int) -> list[str]:
        return self.ranking()[:k]

    def mass(self, support: int = ENTROPY_SUPPORT) -> list[float]:
        """The top `support` scores as a probability distribution.

        The floor is subtracted before normalising. Every candidate carries the
        same popularity and rating offset, so without that subtraction the shared
        baseline dominates the total and every distribution looks uniform no
        matter how decisive the evidence at the top is. Subtracting the floor
        measures the spread that the evidence actually created.
        """
        top = self.scores[:support]
        if not top:
            return []
        floor = min(top)
        # The epsilon keeps a distribution where every score is identical from
        # dividing by zero. It resolves to the uniform distribution, entropy 1.0,
        # which is the right reading of "no evidence separates these".
        spread = [score - floor + 1e-12 for score in top]
        total = sum(spread)
        return [value / total for value in spread]

    def entropy(self, support: int = ENTROPY_SUPPORT) -> float:
        """Normalised Shannon entropy in [0, 1]. 1.0 is flat, 0.0 is decided.

        Normalising by log(n) makes the number comparable across turns even
        though the pool width changes turn by turn, which is the whole reason
        `Config.flat_belief_entropy` can be a single threshold rather than a
        curve.
        """
        distribution = self.mass(support)
        if len(distribution) < 2:
            return 0.0
        total = -sum(p * math.log(p) for p in distribution if p > 0.0)
        return total / math.log(len(distribution))

    def peak_share(self, support: int = ENTROPY_SUPPORT) -> float:
        """Fraction of the mass held by the single best candidate.

        Entropy answers "is anything separated". This answers "is the winner
        separated", which is the question the commit policy actually asks, and
        the two disagree when a handful of strong candidates sit above a long
        flat tail.
        """
        distribution = self.mass(support)
        return distribution[0] if distribution else 0.0
