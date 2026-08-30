"""Reading a spec off disk, and failing loudly when it does not hold together.

The models in `models.py` do the validating. This module's only job is to turn YAML into
them and to attach the file and the field path to whatever went wrong, because the person
who has to fix a bad spec is a merchant integrator reading a stack trace, not the author of
the validator.

It reads YAML only. Nothing here fetches over the network: retrieving a config from a live
platform is surface 1 of the integration contract and belongs to whatever implements it,
not to the contract itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentred.spec.models import AgentConfig, AgentPolicy, AgentSpec, Subject

CONFIG_FILENAME = "config.yaml"
POLICY_FILENAME = "policy.yaml"
SUBJECTS_FILENAME = "subjects.yaml"
"""Identities the harness may act as, kept out of the policy deliberately.

They are test fixtures, not rules. Folding them into `policy.yaml` would mean the policy
version changes every time an identity is added, and the policy version is one quarter of the
tuple a scorecard is valid for. Adding somebody to impersonate would silently invalidate every
scorecard ever produced for that agent, for a change that alters no rule at all.
"""


class SpecError(Exception):
    """A spec could not be read, parsed or validated.

    Raised instead of letting `yaml.YAMLError` or `pydantic.ValidationError` escape, so
    that callers have one exception type to handle and the message always names the file.
    """


def load_spec(
    config_path: Path | str,
    policy_path: Path | str,
    subjects_path: Path | str | None = None,
) -> AgentSpec:
    """Load a config, a policy and any subjects, and check them against each other.

    Args:
        config_path: Path to the config YAML.
        policy_path: Path to the policy YAML.
        subjects_path: Path to the subjects YAML. Absent, or missing on disk, is allowed and
            yields no subjects, which the spec then refuses if its policy needs them.

    Returns:
        A validated `AgentSpec`.

    Raises:
        SpecError: If a file is missing, is not a YAML mapping, fails its own validation, or
            the policy does not describe the config (a bound on an undeclared tool, a
            precondition on an undeclared tool, a scope on an unreachable source, a session
            scoped by identifiers no declared subject supplies).
    """
    config_path, policy_path = Path(config_path), Path(policy_path)
    config = _build(AgentConfig, _read_mapping(config_path), config_path)
    policy = _build(AgentPolicy, _read_mapping(policy_path), policy_path)
    subjects = _read_subjects(Path(subjects_path)) if subjects_path is not None else ()
    try:
        return AgentSpec(config=config, policy=policy, subjects=subjects)
    except ValidationError as error:
        raise SpecError(
            f"{policy_path} does not describe {config_path}:\n{_format(error)}"
        ) from error


def _read_subjects(path: Path) -> tuple[Subject, ...]:
    """Load the identities the harness may act as, if the file is there.

    Args:
        path: Path to the subjects YAML.

    Returns:
        The validated subjects, or none if the file does not exist. A missing file is not an
        error here: whether this agent needs subjects at all is a question only the policy can
        answer, and `AgentSpec` answers it.

    Raises:
        SpecError: If the file exists but is unparseable, is not a mapping, or holds a subject
            that fails validation.
    """
    if not path.exists():
        return ()
    document = _read_mapping(path)
    entries = document.get("subjects", [])
    if not isinstance(entries, list):
        raise SpecError(f"{path}: 'subjects' must be a list, found a {type(entries).__name__}")
    return tuple(_build(Subject, entry, path) for entry in entries)


def load_spec_dir(directory: Path | str) -> AgentSpec:
    """Load the spec from a directory holding `config.yaml` and `policy.yaml`.

    This is the layout `targets/` uses and the layout the CLI expects, so that a spec is
    one path a user can point at rather than three they can mismatch. `subjects.yaml` is
    optional on disk; whether this agent may go without one is decided by its policy.

    Args:
        directory: Directory containing both files.

    Returns:
        A validated `AgentSpec`.

    Raises:
        SpecError: If the directory or either file is missing, or validation fails.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SpecError(f"{directory} is not a directory")
    return load_spec(
        directory / CONFIG_FILENAME,
        directory / POLICY_FILENAME,
        directory / SUBJECTS_FILENAME,
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    """Parse one YAML file that must contain a mapping.

    Raises:
        SpecError: If the file is missing, unparseable, empty, or not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise SpecError(f"{path} does not exist") from error
    except OSError as error:
        raise SpecError(f"{path} could not be read: {error}") from error

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SpecError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict):
        found = "an empty document" if document is None else f"a {type(document).__name__}"
        raise SpecError(f"{path} must contain a mapping at the top level, found {found}")
    return document


def _build[ModelT: AgentConfig | AgentPolicy](
    model: type[ModelT], document: dict[str, Any], path: Path
) -> ModelT:
    """Validate one parsed document into its model, naming the file on failure.

    Raises:
        SpecError: On any validation failure.
    """
    try:
        return model.model_validate(document)
    except ValidationError as error:
        raise SpecError(f"{path} is not a valid {model.__name__}:\n{_format(error)}") from error


def _format(error: ValidationError) -> str:
    """Render a pydantic error as one indented line per problem.

    Pydantic's own rendering carries a URL and a type code on every line, which buries the
    field path. What an integrator needs is the path and the message.
    """
    lines = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "(root)"
        lines.append(f"  {location}: {detail['msg']}")
    return "\n".join(lines)
