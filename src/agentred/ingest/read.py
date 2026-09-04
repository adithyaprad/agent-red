"""Reading one agent, from the manifest that says where the rest of it is.

The four readers each know one source and none of them knows where its source is. That is
what a manifest is: the file an agent is installed from already names the connectors it
requires, the data it may reach and the entry point that starts it, because an installer
needs all three. Pointing agent-red at that same file is what makes the integration one
gesture rather than four, and it is why the reader takes an installed agent rather than a
list of URLs.

**Adapters contribute, they do not compete.** Each returns a package and the packages are
combined in a fixed order: the connector first because the tool schemas are what later
readers reason against, then the instance configuration, then the workflow. A reader that
finds nothing contributes nothing and is not an error, because an agent with no workflow is
an agent built as a model loop rather than an agent read wrongly.

**What this refuses to do is fill in.** Every source is optional and every absence is
recorded. A manifest naming a connector and nothing else yields a real declaration of a tool
surface, an empty policy, and a list of what nobody looked at. That is a smaller true
statement, and the alternative, quietly emitting an agent with no rules, is a false one that
scores perfectly.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from agentred.ingest.adapters import instance as instance_adapter
from agentred.ingest.adapters import mcp as mcp_adapter
from agentred.ingest.adapters import workflow as workflow_adapter
from agentred.ingest.package import AgentPackage


class ManifestError(Exception):
    """An installed agent could not be read from its manifest."""


@dataclass(frozen=True, slots=True)
class Manifest:
    """Where the parts of one agent's declaration live.

    Attributes:
        agent_id: Stable identifier for the agent.
        version: Version of the config to emit. Part of the validity tuple.
        model: The model the agent runs on. Also part of the validity tuple.
        connectors: Connector URLs serving the agent's tools.
        instance: Path to the stored instance configuration, when there is one.
        instructions: Path to the agent's prose, when there is one.
        workflow: An import path, `module:attribute`, naming a zero-argument callable that
            returns the agent's workflow. A path rather than an object because a manifest is
            a file, and a factory rather than an instance because a workflow is normally
            built with its dependencies rather than sitting at module scope.
        root: The directory the manifest was read from, which relative paths resolve against.
    """

    agent_id: str
    version: str
    model: str
    connectors: tuple[str, ...] = ()
    instance: Path | None = None
    instructions: Path | None = None
    workflow: str = ""
    root: Path = Path()


def load_manifest(path: Path | str) -> Manifest:
    """Read a manifest off disk.

    Args:
        path: Path to the manifest YAML.

    Returns:
        The manifest, with every path resolved against the manifest's own directory so that
        an installed agent is relocatable.

    Raises:
        ManifestError: If the file is missing, is not a YAML mapping, or omits the three
            fields that have no discoverable value: the agent's identifier, the version to
            emit, and the model it runs on.
    """
    location = Path(path)
    try:
        raw = yaml.safe_load(location.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"no manifest at {location}") from error
    except yaml.YAMLError as error:
        raise ManifestError(f"{location} is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ManifestError(f"{location} is not a YAML mapping")

    root = location.parent
    missing = [key for key in ("agent_id", "version", "model") if not raw.get(key)]
    if missing:
        raise ManifestError(
            f"{location} omits {', '.join(missing)}. None of the three can be read off a "
            f"running agent, and all three are part of the tuple a scorecard is valid for."
        )

    def resolved(key: str) -> Path | None:
        value = raw.get(key)
        return (root / str(value)) if value else None

    return Manifest(
        agent_id=str(raw["agent_id"]),
        version=str(raw["version"]),
        model=str(raw["model"]),
        connectors=tuple(str(url) for url in (raw.get("connectors") or [])),
        instance=resolved("instance"),
        instructions=resolved("instructions"),
        workflow=str(raw.get("workflow") or ""),
        root=root,
    )


def _load_workflow(reference: str) -> Any:
    """Build the workflow a manifest names.

    Args:
        reference: `module:attribute`, naming a zero-argument callable.

    Returns:
        Whatever the callable returned.

    Raises:
        ManifestError: If the reference is malformed, does not import, or names something
            that cannot be called with no arguments.
    """
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ManifestError(f"workflow {reference!r} is not of the form module:attribute")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise ManifestError(f"cannot import workflow {reference!r}: {error}") from error
    try:
        return factory()
    except TypeError as error:
        raise ManifestError(
            f"workflow {reference!r} is not a callable taking no arguments: {error}"
        ) from error


def _combine(base: AgentPackage, addition: AgentPackage) -> AgentPackage:
    """Fold one reader's findings into another's.

    Args:
        base: What has been read so far.
        addition: What the next reader found.

    Returns:
        A package holding both. Later readers add and never overwrite: two readers that both
        found a fact agree by construction, since each reads a different source, and a
        reader that could silently replace an earlier one's finding would make the order the
        adapters happen to run in load-bearing.
    """
    return replace(
        base.with_rules(*addition.rules),
        tools=base.tools or addition.tools,
        engine=base.engine or addition.engine,
        instructions=base.instructions or addition.instructions,
        data_sources=base.data_sources or addition.data_sources,
        data_scope=base.data_scope or addition.data_scope,
        sources=(*base.sources, *addition.sources),
        notes=(*base.notes, *addition.notes),
    )


async def read_agent(manifest: Manifest, *, http_client: Any = None) -> AgentPackage:
    """Read every source a manifest names, in the order later readers depend on.

    Args:
        manifest: Where the parts of the declaration live.
        http_client: An `httpx2.AsyncClient` for the connectors, when one should be reused.

    Returns:
        One package holding everything the named sources carried, with a hole for every
        question none of them answered.

    Raises:
        ManifestError: If a named source could not be read. A source named and unreadable is
            refused rather than skipped: skipping it would produce a thinner declaration that
            looks exactly like the one a manifest naming fewer sources produces.
    """
    package = AgentPackage(agent_id=manifest.agent_id)
    if manifest.connectors:
        try:
            found = await mcp_adapter.read_agent(
                manifest.agent_id, manifest.connectors, http_client=http_client
            )
        except mcp_adapter.McpReadError as error:
            raise ManifestError(str(error)) from error
        package = _combine(package, found)
    if manifest.instance is not None:
        try:
            package = _combine(package, instance_adapter.read_instance(manifest.instance))
        except instance_adapter.InstanceReadError as error:
            raise ManifestError(str(error)) from error
    if manifest.instructions is not None:
        try:
            prose = manifest.instructions.read_text(encoding="utf-8")
        except OSError as error:
            raise ManifestError(f"cannot read instructions: {error}") from error
        package = replace(package, instructions=prose)
    if manifest.workflow:
        package = workflow_adapter.read_workflow(_load_workflow(manifest.workflow), package)
    return package
