"""Reading an agent's declaration off the platform it was built on.

`spec/loader.py` reads a declaration from YAML and says explicitly that fetching one from a
live platform is not its job. This package is that job. It is the difference between a
harness that works on agents we onboarded by hand and one an integrator can point at an
agent nobody here has seen.

Two halves with opposite properties, and conflating them is the trap this package is shaped
around. The capability half (tools, data sources, channels, the trigger) is written down
somewhere machine-readable by every platform worth integrating with, so it is read rather
than guessed. The policy half (a discount ceiling, an approval threshold) is written down
nowhere, so it is drafted, marked as drafted, and confirmed by a person before it is used.

The honesty constraint is the whole risk and it is the thesis restated: agent-red verifies
that an agent conforms to its declaration and does not verify that the declaration is
correct. A reader that infers a policy and then tests against its own guess gives every
finding a second explanation, so an inferred rule never looks like a declared one at any
point between here and the scorecard.
"""

from agentred.ingest.package import (
    AgentPackage,
    Evidence,
    Observation,
    Origin,
    ToolFacts,
    origin_provenance,
)

__all__ = [
    "AgentPackage",
    "Evidence",
    "Observation",
    "Origin",
    "ToolFacts",
    "origin_provenance",
]
