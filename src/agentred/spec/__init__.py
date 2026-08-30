"""The integration contract: what an agent is, and what it is authorised to do.

Every other package reads this one. It depends on nothing else in the tree, and in
particular `targets/` implements it rather than owning it, so `judge/` never has to import
from `targets/` to know what a violation is.
"""

from agentred.spec.loader import SpecError, load_spec, load_spec_dir
from agentred.spec.models import (
    AgentConfig,
    AgentPolicy,
    AgentSpec,
    Bound,
    Consequence,
    DataScope,
    DataSource,
    EnumeratedBound,
    NumericBound,
    Precondition,
    Provenance,
    RelationalBound,
    ResultReference,
    Subject,
    ToolDeclaration,
    VersionTuple,
)

__all__ = [
    "AgentConfig",
    "AgentPolicy",
    "AgentSpec",
    "Bound",
    "Consequence",
    "DataScope",
    "DataSource",
    "EnumeratedBound",
    "NumericBound",
    "Precondition",
    "Provenance",
    "RelationalBound",
    "ResultReference",
    "SpecError",
    "Subject",
    "ToolDeclaration",
    "VersionTuple",
    "load_spec",
    "load_spec_dir",
]
