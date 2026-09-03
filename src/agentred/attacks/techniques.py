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

**A technique declares the channel families it survives** (ADR-0006). Most of the
conversational corpus is an arc: an opening, a ladder to climb when the agent holds, a sign
that it is working, a point at which to stop. None of that survives being written once into a
free-text field nobody replies to. Three of the eight do survive it, because their whole
mechanism is that the reader accepts something without being asked to, and that works in one
string as well as in six turns. The rest do not, and rendering them anyway would put cells on
a coverage grid that nothing meaningful was ever tried in. A technique run on a family it does
not declare is refused at construction rather than run and counted.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agentred.spec.models import CHANNEL_FAMILIES, CONVERSATIONAL_CHANNEL, PLANTED_FAMILY

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[3] / "data" / "techniques"
"""Where the checked-in corpus lives, relative to the installed package."""

MINIMUM_CORPUS_SIZE = 12
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


class PlantedRendering(BaseModel):
    """How a technique survives being written once, into a field nobody replies to.

    Present only on a technique valid on the planted family, and required there. It carries
    what the conversational half of a technique cannot: there is no next turn to escalate
    into, no reply to read a tell off, and no point at which to stop, so a planted technique
    is one string that either works on the reading or does not.

    Attributes:
        rendering: How the pressure is put into a single passage of text that the agent
            meets as data. The mechanism, not the wording, for the same reason `premise` is:
            the wording has to come from the agent in front of it.
        exemplars: Hand-written strings on an abstract stake, setting the bar the way the
            conversational exemplars do. Separate from those rather than shared, because an
            opening turn and a forged note in a record are not the same piece of writing and
            an exemplar that is the wrong shape teaches the wrong shape.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rendering: str = Field(min_length=40)
    exemplars: tuple[str, ...] = Field(min_length=2, max_length=4)

    @field_validator("rendering", mode="before")
    @classmethod
    def _collapse(cls, value: object) -> object:
        return " ".join(value.split()) if isinstance(value, str) else value

    @field_validator("exemplars", mode="before")
    @classmethod
    def _strip_each(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip() if isinstance(item, str) else item for item in value)
        return value

    @field_validator("exemplars")
    @classmethod
    def _no_blanks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("contains a blank exemplar")
        return value


class Technique(BaseModel):
    """One way of applying pressure, independent of what it is applied to.

    Every field a technique's declared families need is required and none may be blank. That
    is deliberate: the fields most likely to be left out are `escalation` and `fails_when`,
    and those are exactly the two that separate a real attempt from a single knock on the
    door. An attacker with no escalation ladder gives up after one refusal; an attacker with
    no stopping condition spends the whole turn budget arguing with a wall.

    Which fields are needed follows from `channels`, and both directions are refused. A
    technique valid on conversation and carrying no ladder is the failure above. A technique
    valid only on the planted family and carrying a ladder anyway has prose nothing reads,
    which on the next edit becomes prose somebody maintains believing it runs.

    Attributes:
        id: Stable identifier. Appears in every verdict and on the scorecard, so it is
            constrained to lowercase words joined by underscores and never changes casually.
        name: What a person would call it.
        channels: The channel families this survives: `conversation`, `planted`, or
            both. Families rather than channel names, because the corpus is agent-independent
            and a merchant's field name has no business in a file that applies to every agent.
        premise: The one sentence that makes the technique work. The mechanism, not wording.
        pressure: Where the force comes from. Two techniques with the same pressure are one
            technique written twice.
        arc: How it develops across turns. Conversational only.
        escalation: What to do when the agent holds, from least to most forceful.
            Conversational only.
        tells: How to recognise it working, so the attacker pushes rather than restarts.
            Conversational only.
        fails_when: The condition under which to stop and save the remaining turns.
            Conversational only.
        exemplars: Hand-written openings on an abstract stake, setting the persuasiveness
            bar. Conversational only; the planted family has its own inside `planted`.
        planted: How this reads as one string written into a record. Required on the planted
            family and refused off it.
        in_plain_words: What happened, for somebody who will read one conversation and needs
            to know why it worked. Written for a reader rather than for the attacker, so it
            describes the move and never names it: a reader handed a taxonomy has learned a
            word, and a reader handed a sentence has learned what to look for. Required, so
            a technique cannot be added without one and appear on a page as a bare label.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    channels: tuple[str, ...] = Field(min_length=1)
    premise: str = Field(min_length=40)
    pressure: str = Field(min_length=20)
    arc: str = ""
    escalation: tuple[str, ...] = ()
    tells: tuple[str, ...] = ()
    fails_when: str = ""
    exemplars: tuple[str, ...] = Field(default=(), max_length=4)
    planted: PlantedRendering | None = None
    in_plain_words: str = Field(min_length=30)

    @field_validator(
        "premise", "pressure", "arc", "fails_when", "name", "in_plain_words", mode="before"
    )
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

    @field_validator("channels")
    @classmethod
    def _known_families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a family nothing can deliver, and a family named twice.

        An unknown family would produce a technique that crosses with no channel and quietly
        never runs, which on a coverage grid is indistinguishable from a technique that ran
        and found nothing.
        """
        unknown = tuple(family for family in value if family not in CHANNEL_FAMILIES)
        if unknown:
            raise ValueError(
                f"names channel family {', '.join(unknown)}, which nothing delivers. "
                f"Known: {', '.join(CHANNEL_FAMILIES)}"
            )
        if len(set(value)) != len(value):
            raise ValueError("names the same channel family twice")
        return value

    @model_validator(mode="after")
    def _carries_what_its_families_need(self) -> Technique:
        """Refuse a technique whose declared families and written fields disagree.

        Both directions, and both are construction errors rather than warnings. Missing
        prose produces an attacker with nothing to climb; surplus prose is prose nothing
        reads, which the next person to edit the file will maintain in the belief that it
        runs.
        """
        conversational = CONVERSATIONAL_CHANNEL in self.channels
        needed = {
            "arc": (self.arc, 40),
            "fails_when": (self.fails_when, 20),
        }
        if conversational:
            missing = sorted(
                name for name, (value, length) in needed.items() if len(value) < length
            )
            for name, minimum in (("escalation", 2), ("tells", 2), ("exemplars", 2)):
                if len(getattr(self, name)) < minimum:
                    missing.append(name)
            if missing:
                raise ValueError(
                    f"{self.id!r} is valid on conversation and is missing "
                    f"{', '.join(sorted(missing))}, so an attacker running it would have "
                    f"nothing to do when the agent holds"
                )
        else:
            surplus = sorted(
                name
                for name in ("arc", "fails_when", "escalation", "tells", "exemplars")
                if getattr(self, name)
            )
            if surplus:
                raise ValueError(
                    f"{self.id!r} is not valid on conversation but carries "
                    f"{', '.join(surplus)}, which nothing would read"
                )

        if PLANTED_FAMILY in self.channels and self.planted is None:
            raise ValueError(
                f"{self.id!r} is valid on the planted family and says nothing about how it "
                f"reads as one string, so there would be nothing to write into the record"
            )
        if PLANTED_FAMILY not in self.channels and self.planted is not None:
            raise ValueError(
                f"{self.id!r} carries a planted rendering it does not declare the planted "
                f"family for, so the rendering would never be used"
            )
        return self

    def valid_on(self, family: str) -> bool:
        """Whether this technique may be delivered down a channel of `family`."""
        return family in self.channels


def techniques_for(techniques: tuple[Technique, ...], family: str) -> tuple[Technique, ...]:
    """The techniques valid on one channel family, in corpus sequence.

    Args:
        techniques: The corpus.
        family: `CONVERSATIONAL_CHANNEL` or `PLANTED_FAMILY`.

    Returns:
        The subset, in the sequence they were given in.
    """
    return tuple(technique for technique in techniques if technique.valid_on(family))


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

    for family in CHANNEL_FAMILIES:
        if not techniques_for(techniques, family):
            raise TechniqueError(
                f"{directory}: no technique is valid on the {family!r} family, so every "
                f"channel of that family would derive no attacks at all. A channel with no "
                f"attacks reads on a coverage grid as one that was tried and held."
            )
    return techniques
