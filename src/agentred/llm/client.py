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

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "claude-opus-5"
"""The model agent-red itself reasons with. Targets declare their own, in their config."""

DEFAULT_MAX_TOKENS = 8_000
DEFAULT_EFFORT = "high"
ROUTE_ENV_VAR = "AGENTRED_LLM_ROUTE"

MAX_ATTEMPTS = 5
"""How many times one request is sent before the error is allowed through.

Bounded rather than generous. A suite running four conversations at once against on-demand
capacity will be throttled, and the correct response to that is to wait and try again; the
correct response to being throttled five times running is to stop, because a run that
quietly takes six hours instead of one is a run nobody watched.
"""

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
"""HTTP statuses worth sending the same request again for.

Throttling and overload, plus the transient server-side failures. Deliberately excludes 400
and 404: a malformed request or a model id the route does not serve will fail identically
five times, and retrying it turns a clear error into a slow one.
"""

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0

SCHEMA_TOOL = "emit_response"
"""The tool a route without native structured output is forced to call instead.

Named neutrally on purpose. It appears in the request on every schema-constrained call, and
a name describing what the caller wanted would leak that caller's vocabulary into a module
that is deliberately ignorant of it.
"""

BEDROCK_PREFIX = "global.anthropic."
"""The cross-region inference profile prefix Bedrock serves on demand.

Named rather than inlined because it is the whole content of `Route.model_id`, and because
the wrong value here produces a 400 that reads like a missing model rather than like a
throughput question.
"""


class Route(StrEnum):
    """Where model requests are sent.

    Attributes:
        BEDROCK: Amazon Bedrock, partner-operated. Model ids are rewritten to the
            cross-region inference profile form, `global.anthropic.<model>`.
        AWS: Claude Platform on AWS, Anthropic-operated. Bare model ids, SigV4 auth, needs
            a region and a workspace id.
        FIRST_PARTY: The Anthropic API directly. Bare model ids, `ANTHROPIC_API_KEY`.
    """

    BEDROCK = "bedrock"
    AWS = "aws"
    FIRST_PARTY = "first_party"

    @property
    def supports_output_format(self) -> bool:
        """Whether this route accepts a JSON schema on `output_config.format`.

        Bedrock does not: it rejects the field outright. It does support forcing a single
        tool call, which constrains the answer to a schema by a different mechanism and is
        what `AnthropicModelClient` falls back to. See D7.

        Returns:
            True where the schema can be sent directly.
        """
        return self is not Route.BEDROCK

    def model_id(self, model: str) -> str:
        """Translate a first-party model id into this route's form.

        Bedrock is the only route that rewrites, and it rewrites to an inference profile
        rather than to a foundation model id. The distinction is not cosmetic: Bedrock lists
        `anthropic.claude-opus-5` as an available model and then refuses to serve it,
        because a bare foundation model id requires a provisioned throughput commitment.
        The id that on-demand invocation accepts is the cross-region profile,
        `global.anthropic.claude-opus-5`. See D6.

        Args:
            model: A first-party id such as `claude-opus-5`.

        Returns:
            The id this route expects. An id that already carries the prefix is returned
            unchanged, so a caller can pass either.
        """
        if self is Route.BEDROCK and not model.startswith(BEDROCK_PREFIX):
            return f"{BEDROCK_PREFIX}{model.removeprefix('anthropic.')}"
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
        retries: How many times this request was sent again after a retryable failure.
            Carried so a run can report that it was throttled rather than leaving a
            two-hour suite and a twenty-minute suite looking the same from the outside.
    """

    text: str
    stop_reason: str | None
    model: str
    usage: Usage
    retries: int = 0


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


def is_retryable(error: BaseException) -> bool:
    """Whether sending the same request again could plausibly succeed.

    Decided from the status code rather than from the SDK's exception classes, so that this
    module keeps its lazy import of `anthropic` and so that a new exception type in the SDK
    does not silently become unretryable. A connection error carries no status and is
    retried on its class name, which is the one case where the code is not enough.

    Args:
        error: What the SDK raised.

    Returns:
        True for throttling, overload and transient server failures. False for anything the
        route will reject identically next time, which is most 4xx.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in RETRY_STATUSES
    return type(error).__name__ in {"APIConnectionError", "APITimeoutError"}


def backoff_seconds(attempt: int, *, jitter: Callable[[], float] = random.random) -> float:
    """How long to wait before attempt `attempt + 1`, counting from zero.

    Exponential with full jitter and a ceiling. The jitter is the point rather than a
    detail: four conversations throttled by the same burst would otherwise retry in lockstep
    and reproduce the burst that throttled them.

    Args:
        attempt: Zero-based index of the attempt that just failed.
        jitter: Source of randomness in `[0, 1)`. Injected for tests.

    Returns:
        Seconds to sleep, never above `BACKOFF_CAP_SECONDS`.
    """
    ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2**attempt))
    return ceiling * jitter()


def build_sdk_client(route: Route, env: dict[str, str] | None = None) -> Any:
    """Construct the Anthropic SDK client for one route.

    Args:
        route: The route to build for.
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        An `AnthropicBedrock`, `AnthropicAWS` or `Anthropic` instance. All three expose
        the same `messages.create` surface.

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
            return anthropic.AnthropicBedrock(aws_region=region)
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


def build_async_sdk_client(route: Route, env: dict[str, str] | None = None) -> Any:
    """The async twin of `build_sdk_client`, for a target that runs on a workflow engine.

    A workflow-built target reaches the model through the Anthropic SDK directly rather than
    through the Claude Agent SDK, and its steps are awaited, so it needs the async client for
    the same route the rest of the harness resolved. Keeping it here rather than in
    `targets/` means a route is still described in exactly one place, and it is why a
    workflow target can be served on Claude Platform on AWS, which `agent_sdk_env` has to
    refuse because the Agent SDK cannot reach it.

    Args:
        route: The route to build for.
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        An `AsyncAnthropicBedrock`, `AsyncAnthropicAWS` or `AsyncAnthropic` instance.

    Raises:
        LLMConfigurationError: If the route's required configuration is missing, on the same
            terms as the synchronous builder.
    """
    import anthropic

    env = dict(os.environ) if env is None else env

    match route:
        case Route.BEDROCK:
            region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
            if not region:
                raise LLMConfigurationError("The bedrock route needs AWS_REGION.")
            return anthropic.AsyncAnthropicBedrock(aws_region=region)
        case Route.AWS:
            missing = [
                name for name in ("AWS_REGION", "ANTHROPIC_AWS_WORKSPACE_ID") if not env.get(name)
            ]
            if missing:
                raise LLMConfigurationError(
                    f"The aws route needs {' and '.join(missing)}, with no default."
                )
            return anthropic.AsyncAnthropicAWS()
        case Route.FIRST_PARTY:
            return anthropic.AsyncAnthropic()


def agent_sdk_env(route: Route, env: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a Claude Agent SDK target needs to reach the model on this route.

    The targets are agents rather than single calls, so they authenticate through the Agent
    SDK rather than through `AnthropicModelClient`. Returning the variables here keeps one
    answer to "where do model requests go" instead of two, and means serving a target on
    Bedrock is a consequence of the resolved route rather than something an operator has to
    remember to export.

    Args:
        route: The resolved route.
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        Variables to overlay on the target process's environment. Empty for the first-party
        route, where the SDK reads the same key this process already holds.

    Raises:
        LLMConfigurationError: If the route is one the Agent SDK cannot use. Claude Platform
            on AWS is such a route: `AnthropicModelClient` can reach it but the Agent SDK
            cannot, so a target served under it would fail on its first turn rather than at
            startup. Refused here instead, where the message can say what to do.
    """
    env = dict(os.environ) if env is None else env
    match route:
        case Route.BEDROCK:
            region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
            if not region:
                raise LLMConfigurationError(
                    "Serving a target on the bedrock route needs AWS_REGION."
                )
            return {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": region}
        case Route.AWS:
            raise LLMConfigurationError(
                "The Claude Agent SDK cannot use the aws route (Claude Platform on AWS), so a "
                "target cannot be served under it. Set AGENTRED_LLM_ROUTE=bedrock to serve "
                "targets on AWS, or use the first-party route."
            )
        case Route.FIRST_PARTY:
            return {}


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
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build a client, resolving the route from the environment unless one is given.

        Args:
            route: Force a route. Defaults to `resolve_route(env)`.
            model: First-party model id.
            env: Environment to read. Defaults to `os.environ`.
            sdk_client: An already-constructed SDK client, for tests that want the real
                request-shaping code with a stub transport underneath.
            max_attempts: Sends per request, including the first. One disables retrying.
            sleep: How to wait between attempts. Injected so a test can assert the backoff
                without spending it.

        Raises:
            LLMConfigurationError: If no route resolves, or the route is misconfigured.
        """
        self.route = route if route is not None else resolve_route(env)
        self.model = model
        self._client = sdk_client if sdk_client is not None else build_sdk_client(self.route, env)
        self._max_attempts = max(1, max_attempts)
        self._sleep = sleep

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        output_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send one request, retrying a throttled or overloaded route. See `ModelClient`.

        Retries are bounded and the last error is re-raised rather than swallowed. A
        substituted response would be indistinguishable downstream from a real one, and an
        attack turn that was never actually composed would be graded as though it had been.

        Raises:
            Exception: Whatever the SDK raised, once the attempts are spent or the failure
                is not one that retrying can fix.
        """
        request: dict[str, Any] = {"output_config": {"effort": effort}}
        if output_schema is not None:
            if self.route.supports_output_format:
                request["output_config"]["format"] = {
                    "type": "json_schema",
                    "schema": output_schema,
                }
            else:
                request["tools"] = [
                    {
                        "name": SCHEMA_TOOL,
                        "description": "Return the answer in the required shape.",
                        "input_schema": output_schema,
                    }
                ]
                request["tool_choice"] = {"type": "tool", "name": SCHEMA_TOOL}

        for attempt in range(self._max_attempts):
            try:
                response = self._client.messages.create(
                    model=self.route.model_id(self.model),
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    thinking={"type": "adaptive"},
                    **request,
                )
            except Exception as error:
                if attempt + 1 >= self._max_attempts or not is_retryable(error):
                    raise
                self._sleep(backoff_seconds(attempt))
                continue
            return _to_model_response(response, retries=attempt)
        raise AssertionError("unreachable: the loop either returns or raises")


def _to_model_response(response: Any, *, retries: int = 0) -> ModelResponse:
    """Flatten an SDK message into the narrow response the tree consumes.

    Text blocks are concatenated and thinking blocks dropped: a judge that reasons in its
    thinking block and answers in its text block must be scored on the answer.

    A route that answered a schema-constrained call by being forced into a tool call carries
    its answer in that call's arguments instead, and it is serialised back to JSON text here.
    That keeps the difference inside this module: every caller receives a string it can parse
    the same way regardless of route, and the narrow `ModelClient` surface stays narrow.
    """
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    if not text:
        forced = [
            block
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", "") == SCHEMA_TOOL
        ]
        if forced:
            text = json.dumps(getattr(forced[0], "input", {}), default=str)
    usage = getattr(response, "usage", None)
    return ModelResponse(
        text=text,
        stop_reason=getattr(response, "stop_reason", None),
        model=getattr(response, "model", ""),
        retries=retries,
        usage=Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        ),
    )
