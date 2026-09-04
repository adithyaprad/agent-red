"""Checking a recovered declaration against one a person wrote.

The reader's claim is that a declaration does not have to be authored, and the only way to
know whether that is true is to author one, recover the same agent independently, and put
the two side by side. This is the same move the world generator is checked with: an agent
whose generated and hand-written halves disagree is how the generated one is known to be
wrong, and the disagreement is worth more than the agreement because it is the part that
would otherwise have shipped unnoticed.

**What a difference means is not obvious, and the shape here is built around that.** A field
the reader did not recover is not a defect if no reader was pointed at the source that
carries it, and it is a defect if one was. So a difference is reported as one of three
things rather than as a count: recovered and identical, recovered and disagreeing, or not
covered by any reader that ran. The middle one is the only one that is a bug, and collapsing
the three into a percentage would hide it among the other two.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentred.spec.models import AgentConfig, AgentPolicy, ToolDeclaration


class Verdict(StrEnum):
    """What happened to one field of a hand-written declaration when it was recovered.

    Attributes:
        MATCHED: The reader produced the same value. The claim holds for this field.
        DIVERGED: The reader produced a different value. This is the only verdict that is a
            defect in the reader, and the only one worth acting on.
        UNCOVERED: No reader that ran reads the source this field comes from. Expected, and
            reported rather than scored, because a reader pointed at one connector is not
            failing to find an agent's data sources; it is not looking.
        ADDED: The reader produced a field the hand-written declaration does not have. Not
            automatically wrong, and always worth reading: the platform may carry something
            the author forgot.
    """

    MATCHED = "matched"
    DIVERGED = "diverged"
    UNCOVERED = "uncovered"
    ADDED = "added"


@dataclass(frozen=True, slots=True)
class FieldDiff:
    """One field of one declaration, compared.

    Attributes:
        subject: What was compared, for example `tool issue_refund` or `tool get_order
            parameters`.
        verdict: See `Verdict`.
        authored: What the hand-written declaration said, rendered short.
        recovered: What the reader produced, rendered short.
    """

    subject: str
    verdict: Verdict
    authored: str = ""
    recovered: str = ""


@dataclass(frozen=True, slots=True)
class Recovery:
    """The whole comparison, and the one number that is honest to quote from it.

    Attributes:
        agent_id: The agent both declarations describe.
        fields: Every field compared, in a stable order.
    """

    agent_id: str
    fields: tuple[FieldDiff, ...]

    def of(self, verdict: Verdict) -> tuple[FieldDiff, ...]:
        """Every field with one verdict.

        Args:
            verdict: The verdict to select.

        Returns:
            The matching fields, in the order they were compared.
        """
        return tuple(field for field in self.fields if field.verdict is verdict)

    @property
    def faithful(self) -> bool:
        """Whether the reader disagreed with the author about anything it read.

        Uncovered fields do not count against it: a reader that was not pointed at a source
        has made no claim about it. This is the property a test asserts, rather than a
        recovery percentage, because a percentage moves when a reader is pointed at more
        sources and says nothing about whether either reading was right.
        """
        return not self.of(Verdict.DIVERGED)

    def render(self) -> str:
        """The comparison as lines a person reads.

        Returns:
            One line per field that is not a match, headed by the counts. Matches are
            counted and not listed: a reader that recovered forty fields correctly produces
            forty lines nobody reads, and the two that diverged are then in the middle of
            them.
        """
        counts = {verdict: len(self.of(verdict)) for verdict in Verdict}
        head = (
            f"{self.agent_id}: {counts[Verdict.MATCHED]} matched, "
            f"{counts[Verdict.DIVERGED]} diverged, "
            f"{counts[Verdict.UNCOVERED]} not covered by any reader that ran, "
            f"{counts[Verdict.ADDED]} found that the author did not declare"
        )
        lines = [
            f"  {field.verdict}: {field.subject}"
            + (
                f" (authored {field.authored}, recovered {field.recovered})"
                if field.authored or field.recovered
                else ""
            )
            for field in self.fields
            if field.verdict is not Verdict.MATCHED
        ]
        return "\n".join([head, *lines])


def _short(value: object) -> str:
    """A value rendered for a diff line, truncated so one long schema cannot fill the report."""
    text = str(value)
    return text if len(text) <= 60 else f"{text[:57]}..."


def _compare_tool(authored: ToolDeclaration, recovered: ToolDeclaration) -> list[FieldDiff]:
    """Compare one tool the author declared with the same tool as recovered.

    Args:
        authored: The hand-written declaration of the tool.
        recovered: What the reader produced for it.

    Returns:
        One `FieldDiff` per attribute compared. `consequence` is compared like any other
        field: it is the one no connector carries, so a run where it matches is a run where
        somebody answered the question correctly, and that is worth seeing separately from
        the schema having been read off the wire.
    """
    checks = (
        ("parameters", authored.parameters, recovered.parameters),
        ("description", authored.description, recovered.description),
        ("consequence", authored.consequence, recovered.consequence),
    )
    return [
        FieldDiff(
            subject=f"tool {authored.name} {field}",
            verdict=Verdict.MATCHED if left == right else Verdict.DIVERGED,
            authored="" if left == right else _short(left),
            recovered="" if left == right else _short(right),
        )
        for field, left, right in checks
    ]


def _obligation(rule: Any) -> tuple[str, ...]:
    """What a rule requires, stripped of what it is called.

    Args:
        rule: Any policy statement.

    Returns:
        A key naming the obligation: for a bound, its kind, tool and argument; for a
        precondition, the pair of calls it orders; for the rest, the tool and the rule type.

    A comparison keyed on names would be worthless here, and worthless in the direction that
    flatters nobody. A reader generates a name from what it read, an author writes one from
    what they meant, and `issue_refund_amount_limit` and `refund_ceiling` are the same
    requirement under two labels. Keyed on names, every recovered rule reports as a miss and
    an addition at once, and the report says the reader recovered nothing when it recovered
    the rule exactly.
    """
    kind = type(rule).__name__
    tool = str(getattr(rule, "tool", ""))
    if hasattr(rule, "requires"):
        return ("precondition", tool, str(rule.requires))
    argument = str(getattr(rule, "argument", "") or getattr(rule, "value_from", ""))
    return (kind, tool, argument)


def _obligations(policy: AgentPolicy | None) -> dict[tuple[str, ...], str]:
    """Every rule in a policy, keyed by what it requires rather than by its name.

    Args:
        policy: The policy to read, or `None`.

    Returns:
        A mapping from obligation key to the rule's own name, so a report can still say
        which rule matched in the words each side used for it.
    """
    if policy is None:
        return {}
    buckets = (
        policy.bounds,
        policy.preconditions,
        policy.idempotency,
        policy.outbound,
        policy.citations,
    )
    return {_obligation(rule): rule.name for rules in buckets for rule in rules}


def _compare_policies(
    authored: AgentPolicy | None, recovered: AgentPolicy | None
) -> list[FieldDiff]:
    """Compare the rules two policies hold, by what each rule is named for.

    Args:
        authored: The policy a person wrote, or `None`.
        recovered: The policy the readers produced, or `None`.

    Returns:
        One `FieldDiff` per rule on either side. A rule the author declared and no reader
        produced is `UNCOVERED` rather than `DIVERGED`, because a reader pointed at a form
        with no field for a rule has not disagreed about it. A rule the readers produced and
        the author did not declare is `ADDED`, and those are the ones worth reading: a
        workflow enforces orderings nobody wrote down, and an ordering the deployment
        enforces is a rule whether or not it reached a document.
    """
    left = _obligations(authored)
    right = _obligations(recovered)
    fields = [
        FieldDiff(
            subject=f"rule {name}",
            verdict=Verdict.MATCHED if key in right else Verdict.UNCOVERED,
            recovered=right.get(key, ""),
        )
        for key, name in left.items()
    ]
    fields.extend(
        FieldDiff(subject=f"rule {name}", verdict=Verdict.ADDED)
        for key, name in right.items()
        if key not in left
    )
    return fields


def compare(
    authored: AgentConfig,
    recovered: AgentConfig,
    *,
    uncovered: tuple[str, ...] = (),
    authored_policy: AgentPolicy | None = None,
    recovered_policy: AgentPolicy | None = None,
) -> Recovery:
    """Put a hand-written declaration and a recovered one side by side.

    Args:
        authored: The declaration a person wrote, treated as the thing to be recovered
            rather than as the truth. Where the two disagree the author may be the one who
            is wrong, which is why the verdict is `DIVERGED` rather than `WRONG`.
        recovered: What the readers produced.
        uncovered: Parts of the declaration no reader that ran covers, from
            `Emission.unreadable`. Named here so they are reported as unlooked-for rather
            than as missing.
        authored_policy: The policy a person wrote, when there is one to compare against.
        recovered_policy: The policy the readers produced.

    Returns:
        The comparison.
    """
    fields: list[FieldDiff] = []
    authored_tools = {tool.name: tool for tool in authored.tools}
    recovered_tools = {tool.name: tool for tool in recovered.tools}

    for name, tool in authored_tools.items():
        if name not in recovered_tools:
            fields.append(FieldDiff(subject=f"tool {name}", verdict=Verdict.UNCOVERED))
            continue
        fields.extend(_compare_tool(tool, recovered_tools[name]))
    for name in recovered_tools:
        if name not in authored_tools:
            fields.append(FieldDiff(subject=f"tool {name}", verdict=Verdict.ADDED))

    fields.extend(_compare_policies(authored_policy, recovered_policy))
    fields.extend(FieldDiff(subject=line, verdict=Verdict.UNCOVERED) for line in uncovered)
    return Recovery(agent_id=authored.agent_id, fields=tuple(fields))
