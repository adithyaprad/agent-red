"""The technique corpus: eight ways to talk somebody out of a rule, and its loader.

A technique is a shape of pressure and nothing else. It does not know what it is being
applied to, and it must not: what is worth attacking comes from the agent's own spec, derived
in `stakes.py`, and crossing the two is what makes the suite a function of the agent under
test rather than of whoever wrote the corpus. A technique naming whatever the first agent
happened to sell would work against that one agent and be dead weight against the next.

The corpus is data rather than code (`data/techniques/*.yaml`) because it is prose that gets
edited by reading it, and because a technique is a hypothesis about persuasion, not a
behaviour. The loader validates it, so a corpus that has drifted fails on the way in rather
than producing an attacker with an empty escalation ladder and no way to notice.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[3] / "data" / "techniques"
"""Where the checked-in corpus lives, relative to the installed package."""

MINIMUM_CORPUS_SIZE = 8
"""Below this the corpus is not a corpus, and a short one is almost always a bad path."""


class TechniqueError(Exception):
    """A technique file could not be read, parsed or validated.

    One exception type, and the message always names the file, because the corpus is edited
    by hand far more often than the code that reads it.
    """


def _non_empty_lines(value: tuple[str, ...]) -> tuple[str, ...]:
    """Strip surrounding whitespace and refuse a blank entry."""
    cleaned = tuple(item.strip() for item in value)
    if any(not item for item in cleaned):
        raise ValueError("contains a blank entry")
    return cleaned


class Technique(BaseModel):
    """One way of applying pressure, independent of what it is applied to.

    Every field is required and none may be blank. That is deliberate: the fields most likely
    to be left out are `escalation` and `fails_when`, and those are exactly the two that
    separate a real attempt from a single knock on the door. An attacker with no escalation
    ladder gives up after one refusal; an attacker with no stopping condition spends the whole
    turn budget arguing with a wall.

    Attributes:
        id: Stable identifier. Appears in every verdict and on the scorecard, so it is
            constrained to lowercase words joined by underscores and never changes casually.
        name: What a person would call it.
        premise: The one sentence that makes the technique work. The mechanism, not wording.
        pressure: Where the force comes from. Two techniques with the same pressure are one
            technique written twice.
        arc: How it develops across turns.
        escalation: What to do when the agent holds, from least to most forceful.
        tells: How to recognise it working, so the attacker pushes rather than restarts.
        fails_when: The condition under which to stop and save the remaining turns.
        exemplars: Hand-written openings on an abstract stake, setting the persuasiveness bar.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    premise: str = Field(min_length=40)
    pressure: str = Field(min_length=20)
    arc: str = Field(min_length=40)
    escalation: tuple[str, ...] = Field(min_length=2)
    tells: tuple[str, ...] = Field(min_length=2)
    fails_when: str = Field(min_length=20)
    exemplars: tuple[str, ...] = Field(min_length=2, max_length=4)

    @field_validator("premise", "pressure", "arc", "fails_when", "name", mode="before")
    @classmethod
    def _collapse(cls, value: object) -> object:
        """Fold the whitespace YAML block scalars leave behind."""
        return " ".join(value.split()) if isinstance(value, str) else value

    @field_validator("escalation", "tells", "exemplars", mode="before")
    @classmethod
    def _collapse_each(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(
                " ".join(item.split()) if isinstance(item, str) else item for item in value
            )
        return value

    @field_validator("escalation", "tells", "exemplars", "name")
    @classmethod
    def _no_blanks(cls, value: tuple[str, ...] | str) -> tuple[str, ...] | str:
        return _non_empty_lines(value) if isinstance(value, tuple) else value


def load_technique(path: Path | str) -> Technique:
    """Load and validate one technique file.

    Args:
        path: Path to a technique YAML file.

    Returns:
        The validated `Technique`.

    Raises:
        TechniqueError: If the file is missing, unparseable, not a mapping, or fails
            validation. The message always names the file.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TechniqueError(f"{path}: no such file") from error
    except (OSError, yaml.YAMLError) as error:
        raise TechniqueError(f"{path}: could not be read as YAML: {error}") from error

    if not isinstance(raw, dict):
        raise TechniqueError(f"{path}: expected a mapping, found {type(raw).__name__}")

    try:
        return Technique.model_validate(raw)
    except ValidationError as error:
        raise TechniqueError(f"{path}: {error}") from error


def load_corpus(directory: Path | str | None = None) -> tuple[Technique, ...]:
    """Load every technique in a directory, in filename sequence.

    Filename sequence is corpus sequence, which is why the files are numbered. Attack
    generation is deterministic given a spec and a corpus, and iterating a set would quietly
    break that.

    Args:
        directory: Where the corpus lives. Defaults to the checked-in `data/techniques/`.

    Returns:
        The techniques, in filename sequence.

    Raises:
        TechniqueError: If the directory is missing, holds fewer than `MINIMUM_CORPUS_SIZE`
            techniques, or contains two techniques with the same id. A short corpus is
            refused rather than warned about, because the usual cause is a wrong path and
            the usual symptom is a suite that passes everything.
    """
    directory = Path(directory) if directory is not None else DEFAULT_CORPUS_DIR
    if not directory.is_dir():
        raise TechniqueError(f"{directory}: no such technique directory")

    techniques = tuple(load_technique(path) for path in sorted(directory.glob("*.yaml")))

    seen: dict[str, int] = {}
    for index, technique in enumerate(techniques):
        if technique.id in seen:
            raise TechniqueError(
                f"{directory}: technique id {technique.id!r} appears twice, at positions "
                f"{seen[technique.id]} and {index}"
            )
        seen[technique.id] = index

    if len(techniques) < MINIMUM_CORPUS_SIZE:
        raise TechniqueError(
            f"{directory}: holds {len(techniques)} techniques, expected at least "
            f"{MINIMUM_CORPUS_SIZE}. A corpus this small is almost always a wrong path, and "
            f"the symptom would be a suite that reports an agent as safe."
        )
    return techniques
