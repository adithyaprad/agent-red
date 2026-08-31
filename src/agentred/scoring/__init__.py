"""Turning a set of conversations into what an operator is told.

One thing lives here that lives nowhere else: a property of a *set* of conversations. Every
check in `judge/` reads one transcript, which is right for a broken rule and blind to an agent
that answers the same question two ways.
"""

from agentred.scoring.consistency import Attempt, Comparison, Divergence, compare

__all__ = ["Attempt", "Comparison", "Divergence", "compare"]
