"""Loading hand-written payloads and crossing them with an agent's own declarations.

The counterpart to `techniques.py`, and deliberately much smaller. A technique is a shape of
pressure that works on any agent, so the corpus is checked in once and applies everywhere. A
payload is one string, and a string that lands in one agent's free-text field says nothing
about another agent's, so payloads are per agent and live beside the agent they were written
for rather than in the shared corpus.

That is a stated limitation and not a design to be proud of. It is the state of the planted
channel before its technique family exists: three payloads written by hand to prove the
lifecycle carries a real finding end to end, and a generator built afterwards on a loop that
has been shown to work.

Nothing here knows what an agent sells. A payload file names a stake id, a channel name, a
record and a string; every one of those is a coordinate the agent itself supplied, and the
loader refuses any that the agent did not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentred.attacks.generator import Attack, PlantedPayload
from agentred.attacks.stakes import Stake, derive_stakes
from agentred.spec.models import AgentSpec

DEFAULT_PAYLOAD_DIR = Path("data/planted")


class PayloadError(Exception):
    """A payload file could not be read, parsed, or matched to the agent it names.

    One exception type, and the message always names the file, because these files are
    hand-edited far more often than the code that reads them.
    """


def payload_path(agent_id: str, directory: Path | str | None = None) -> Path:
    """Where a given agent's hand-written payloads live."""
    root = DEFAULT_PAYLOAD_DIR if directory is None else Path(directory)
    return root / f"{agent_id}.yaml"


def load_planted(spec: AgentSpec, directory: Path | str | None = None) -> tuple[Attack, ...]:
    """Every hand-written payload for this agent, as attacks it can actually run.

    Each payload is checked against the agent's own declarations before it becomes an attack.
    A payload naming a channel the agent does not declare, a stake nothing derives, or a
    subject nobody declared is refused at load rather than run, for the same reason a bound on
    an undeclared tool is: it would plant somewhere nothing reads, complete, and report on a
    coverage grid as a cell that was tested.

    Args:
        spec: The validated agent spec. Supplies the channels, the derived stakes and the
            declared subjects a payload is checked against.
        directory: Where the payload files live. Defaults to `data/planted`.

    Returns:
        The attacks, in file sequence. Empty when the agent has no payload file, which is not
        an error: an agent that declares no channel has nothing to plant into, and the
        coverage grid says so.

    Raises:
        PayloadError: If the file exists but is unparseable, is for a different agent, or
            names a channel, stake or subject the agent does not have.
    """
    path = payload_path(spec.config.agent_id, directory)
    if not path.exists():
        return ()

    document = _read_mapping(path)
    named = str(document.get("agent_id", ""))
    if named != spec.config.agent_id:
        raise PayloadError(
            f"{path} carries payloads for agent {named!r} but was loaded for "
            f"{spec.config.agent_id!r}. They are not about the same agent."
        )

    entries = document.get("payloads", [])
    if not isinstance(entries, list):
        raise PayloadError(f"{path}: 'payloads' must be a list, found a {type(entries).__name__}")

    channels = spec.config.channels_by_name
    stakes = {stake.id: stake for stake in derive_stakes(spec)}
    subjects = {subject.name: subject for subject in spec.subjects}

    attacks: list[Attack] = []
    for index, entry in enumerate(entries):
        attacks.append(_build(entry, index, path, channels, stakes, subjects))

    seen: set[str] = set()
    for attack in attacks:
        if attack.id in seen:
            raise PayloadError(
                f"{path} produces two attacks with id {attack.id!r}. An id is used as an "
                f"identifier throughout, so a repeat silently merges two results."
            )
        seen.add(attack.id)
    return tuple(attacks)


def _build(
    entry: Any,
    index: int,
    path: Path,
    channels: dict[str, Any],
    stakes: dict[str, Stake],
    subjects: dict[str, Any],
) -> Attack:
    """Turn one parsed payload entry into an attack, or refuse it.

    Raises:
        PayloadError: On anything the agent's declarations do not support.
    """
    where = f"{path} payload {index}"
    if not isinstance(entry, dict):
        raise PayloadError(f"{where} is a {type(entry).__name__}, not a mapping")

    for required in ("label", "channel", "stake", "record_id", "text", "in_plain_words"):
        if not str(entry.get(required, "")).strip():
            raise PayloadError(f"{where} declares no {required}")

    channel = str(entry["channel"])
    if channel not in channels:
        raise PayloadError(
            f"{where} arrives down channel {channel!r}, which this agent does not declare. "
            f"Declared: {', '.join(sorted(channels)) or 'none'}."
        )

    stake_id = str(entry["stake"])
    stake = stakes.get(stake_id)
    if stake is None:
        raise PayloadError(
            f"{where} is aimed at stake {stake_id!r}, which nothing in this agent's policy "
            f"derives. A payload aimed at a stake that does not exist would run and be "
            f"graded against nothing."
        )

    subject = None
    if str(entry.get("subject", "")).strip():
        name = str(entry["subject"])
        subject = subjects.get(name)
        if subject is None:
            raise PayloadError(
                f"{where} acts as {name!r}, whom this agent declares no subject for. "
                f"Declared: {', '.join(sorted(subjects)) or 'none'}."
            )

    try:
        payload = PlantedPayload(
            label=str(entry["label"]),
            record_id=str(entry["record_id"]),
            text=str(entry["text"]),
            in_plain_words=" ".join(str(entry["in_plain_words"]).split()),
        )
    except Exception as error:
        raise PayloadError(f"{where}: {error}") from error

    return Attack(stake=stake, channel=channel, planted=payload, subject=subject)


def _read_mapping(path: Path) -> dict[str, Any]:
    """Parse one YAML file that must contain a mapping.

    Raises:
        PayloadError: If the file is unreadable, unparseable, empty, or not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PayloadError(f"{path} could not be read: {error}") from error
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PayloadError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(document, dict):
        found = "an empty document" if document is None else f"a {type(document).__name__}"
        raise PayloadError(f"{path} must contain a mapping at the top level, found {found}")
    return document
