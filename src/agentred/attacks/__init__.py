"""Attack generation: what to say, and what it is worth saying it about.

Two halves that meet in `generator.py`. `techniques.py` holds ways of applying pressure and
knows nothing about any agent. `stakes.py` reads a validated spec and works out what is worth
attacking about that agent in particular. Crossing them is what makes the suite a function of
the agent under test rather than of whoever wrote it.

No module here may name a domain. The templates carry the grammar and the spec supplies every
value, so the code stays generic while its output is entirely about the agent in front of it.
`tests/test_no_domain_vocabulary.py` fails the build when that slips.
"""

from agentred.attacks.stakes import (
    Settlement,
    Stake,
    StakeKind,
    derive_stakes,
    judge_dependence,
)
from agentred.attacks.techniques import (
    MINIMUM_CORPUS_SIZE,
    Technique,
    TechniqueError,
    load_corpus,
    load_technique,
)

__all__ = [
    "MINIMUM_CORPUS_SIZE",
    "Settlement",
    "Stake",
    "StakeKind",
    "Technique",
    "TechniqueError",
    "derive_stakes",
    "judge_dependence",
    "load_corpus",
    "load_technique",
]
