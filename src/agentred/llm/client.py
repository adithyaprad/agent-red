"""One model client, three routes, one environment variable.

agent-red calls a model in two places that matter: to compose attacks, and to judge the
interpretive residue the detectors cannot decide. Both go through `ModelClient`, so that
tests can substitute a recorded transcript and so that moving between Amazon Bedrock,
Claude Platform on AWS and the first-party API is a deployment concern rather than a code
change.

Route selection is `AGENTRED_LLM_ROUTE`. Unset, it is resolved from whichever credentials
are present, preferring the AWS routes and falling back to a first-party key, so that an
operator holding nothing but an Anthropic key can run the whole suite.

The three clients differ in exactly two ways, and both are handled here: how they are
constructed, and whether model ids carry a provider prefix. Everything downstream of
`complete()` is identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "claude-opus-5"
"""The model agent-red itself reasons with. Targets declare their own, in their config."""

DEFAULT_MAX_TOKENS = 8_000
DEFAULT_EFFORT = "high"
ROUTE_ENV_VAR = "AGENTRED_LLM_ROUTE"


class Route(StrEnum):
    """Where model requests are sent.

    Attributes:
        BEDROCK: Amazon Bedrock, partner-operated. Model ids carry an `anthropic.` prefix.
        AWS: Claude Platform on AWS, Anthropic-operated. Bare model ids, SigV4 auth, needs
            a region and a workspace id.
        FIRST_PARTY: The Anthropic API directly. Bare model ids, `ANTHROPIC_API_KEY`.
    """

    BEDROCK = "bedrock"
    AWS = "aws"
    FIRST_PARTY = "first_party"

    def model_id(self, model: str) -> str:
        """Translate a first-party model id into this route's form.

        Args:
            model: A first-party id such as `claude-opus-5`.

        Returns:
            The id this route expects. Only Bedrock rewrites it, and an id that already
            carries the prefix is returned unchanged so a caller can pass either.
        """
        if self is Route.BEDROCK and not model.startswith("anthropic."):
            return f"anthropic.{model}"
        return model


class LLMConfigurationError(RuntimeError):
    """No usable route could be resolved, or the named route lacks its credentials.

    Raised at client construction, never mid-run, so that a misconfigured environment
    fails before a suite starts rather than four hundred conversations into one.
    """


@dataclass(frozen=True)
class Usage:
    """Token counts for one call, carried so that a run can report what it cost.

    Attributes:
        input_tokens: Uncached input tokens.
        output_tokens: Generated tokens.
        cache_read_tokens: Input tokens served from the prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    """What one call returned.

    Attributes:
        text: The concatenated text blocks. Thinking blocks are not included.
        stop_reason: The API stop reason, or `None` if the route did not report one.
        model: The model id that actually served the request, as reported by the API.
        usage: Token counts.
    """

    text: str
    stop_reason: str | None
    model: str
    usage: Usage


@runtime_checkable
class ModelClient(Protocol):
    """The only model surface the rest of the tree may use.

    Kept narrow deliberately: attack generation and judging both need text in and text
    out, optionally constrained to a JSON schema. Anything wider would have to be faked
    faithfully in `tests/fakes/`, and a fake that drifts from the real client turns green
    tests into evidence of nothing.
    """

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        output_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send one request and return the response.

        Args:
            system: The system prompt.
            messages: Messages in Anthropic wire format.
            max_tokens: Output ceiling.
            effort: Thinking effort, `low` through `max`.
            output_schema: A JSON schema the response must conform to. Used by the judge,
                where a free-text verdict would have to be parsed and could be malformed.

        Returns:
            The response text, stop reason and usage.
        """
        ...


def resolve_route(env: dict[str, str] | None = None) -> Route:
    """Decide which route to use from the environment.

    `AGENTRED_LLM_ROUTE` wins if set. Otherwise the route is inferred from credentials,
    preferring Claude Platform on AWS, then Bedrock, then the first-party API. The
    first-party fallback is deliberate: an operator with one Anthropic key must be able to
    run everything without an AWS account.

    Args:
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        The resolved route.

    Raises:
        LLMConfigurationError: If `AGENTRED_LLM_ROUTE` names an unknown route, or no route
            has usable credentials.
    """
    env = dict(os.environ) if env is None else env

    if (named := env.get(ROUTE_ENV_VAR, "").strip()) != "":
        try:
            return Route(named)
        except ValueError:
            options = ", ".join(route.value for route in Route)
            raise LLMConfigurationError(
                f"{ROUTE_ENV_VAR}={named!r} is not a route. Expected one of: {options}."
            ) from None

    if env.get("ANTHROPIC_AWS_WORKSPACE_ID") and env.get("AWS_REGION"):
        return Route.AWS
    if env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION"):
        return Route.BEDROCK
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"):
        return Route.FIRST_PARTY

    raise LLMConfigurationError(
        "No model route is configured. Set ANTHROPIC_API_KEY for the first-party API, or "
        "AWS_REGION for Amazon Bedrock, or AWS_REGION plus ANTHROPIC_AWS_WORKSPACE_ID for "
        f"Claude Platform on AWS. {ROUTE_ENV_VAR} overrides this detection."
    )


def build_sdk_client(route: Route, env: dict[str, str] | None = None) -> Any:
    """Construct the Anthropic SDK client for one route.

    Args:
        route: The route to build for.
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        An `AnthropicBedrockMantle`, `AnthropicAWS` or `Anthropic` instance. All three
        expose the same `messages.create` surface.

    Raises:
        LLMConfigurationError: If the route's required configuration is missing. Both AWS
            routes need a region, and Claude Platform on AWS also needs a workspace id;
            neither has a default, so this is checked here rather than left to a stack
            trace from inside the SDK.
    """
    import anthropic

    env = dict(os.environ) if env is None else env

    match route:
        case Route.BEDROCK:
            region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
            if not region:
                raise LLMConfigurationError("The bedrock route needs AWS_REGION.")
            return anthropic.AnthropicBedrockMantle(aws_region=region)
        case Route.AWS:
            missing = [
                name for name in ("AWS_REGION", "ANTHROPIC_AWS_WORKSPACE_ID") if not env.get(name)
            ]
            if missing:
                raise LLMConfigurationError(
                    f"The aws route needs {' and '.join(missing)}, with no default."
                )
            return anthropic.AnthropicAWS()
        case Route.FIRST_PARTY:
            return anthropic.Anthropic()


class AnthropicModelClient:
    """A `ModelClient` backed by the Anthropic SDK on whichever route is configured.

    Adaptive thinking is on for every call and effort is the knob, which is the current
    shape of the API rather than a per-call budget. Requests are not streamed: every call
    agent-red makes is a single bounded turn well inside the default timeout.

    Attributes:
        route: The route in use.
        model: The first-party model id. Translated per route at call time.
    """

    def __init__(
        self,
        *,
        route: Route | None = None,
        model: str = DEFAULT_MODEL,
        env: dict[str, str] | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        """Build a client, resolving the route from the environment unless one is given.

        Args:
            route: Force a route. Defaults to `resolve_route(env)`.
            model: First-party model id.
            env: Environment to read. Defaults to `os.environ`.
            sdk_client: An already-constructed SDK client, for tests that want the real
                request-shaping code with a stub transport underneath.

        Raises:
            LLMConfigurationError: If no route resolves, or the route is misconfigured.
        """
        self.route = route if route is not None else resolve_route(env)
        self.model = model
        self._client = sdk_client if sdk_client is not None else build_sdk_client(self.route, env)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        output_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send one request. See `ModelClient.complete`."""
        output_config: dict[str, Any] = {"effort": effort}
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}

        response = self._client.messages.create(
            model=self.route.model_id(self.model),
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config=output_config,
        )
        return _to_model_response(response)


def _to_model_response(response: Any) -> ModelResponse:
    """Flatten an SDK message into the narrow response the tree consumes.

    Text blocks are concatenated and thinking blocks dropped: a judge that reasons in its
    thinking block and answers in its text block must be scored on the answer.
    """
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    usage = getattr(response, "usage", None)
    return ModelResponse(
        text=text,
        stop_reason=getattr(response, "stop_reason", None),
        model=getattr(response, "model", ""),
        usage=Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        ),
    )
