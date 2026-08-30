"""Route resolution and request shaping. Offline: the SDK is stubbed."""

import json
from typing import ClassVar

import pytest

from agentred.llm.client import (
    SCHEMA_TOOL,
    AnthropicModelClient,
    LLMConfigurationError,
    ModelClient,
    Route,
    agent_sdk_env,
    backoff_seconds,
    build_sdk_client,
    is_retryable,
    resolve_route,
)
from tests.fakes.model import StubAPIError, StubSDKClient

BEDROCK_ENV = {"AWS_REGION": "us-east-1"}
AWS_ENV = {"AWS_REGION": "us-east-1", "ANTHROPIC_AWS_WORKSPACE_ID": "ws_1"}
FIRST_PARTY_ENV = {"ANTHROPIC_API_KEY": "not-a-real-key"}


class TestRouteResolution:
    @pytest.mark.parametrize("route", list(Route))
    def test_the_env_var_wins(self, route):
        assert resolve_route({"AGENTRED_LLM_ROUTE": route.value, **FIRST_PARTY_ENV}) is route

    def test_an_unknown_route_names_the_alternatives(self):
        with pytest.raises(LLMConfigurationError, match="bedrock, aws, first_party"):
            resolve_route({"AGENTRED_LLM_ROUTE": "openai"})

    def test_a_blank_route_falls_through_to_detection(self):
        assert resolve_route({"AGENTRED_LLM_ROUTE": "  ", **FIRST_PARTY_ENV}) is Route.FIRST_PARTY

    def test_workspace_id_selects_claude_platform_on_aws(self):
        assert resolve_route(AWS_ENV) is Route.AWS

    def test_a_region_alone_selects_bedrock(self):
        assert resolve_route(BEDROCK_ENV) is Route.BEDROCK

    def test_a_key_alone_selects_first_party(self):
        assert resolve_route(FIRST_PARTY_ENV) is Route.FIRST_PARTY

    def test_aws_outranks_a_first_party_key(self):
        assert resolve_route({**AWS_ENV, **FIRST_PARTY_ENV}) is Route.AWS

    def test_no_credentials_names_all_three_routes(self):
        with pytest.raises(LLMConfigurationError) as caught:
            resolve_route({})
        message = str(caught.value)
        assert "ANTHROPIC_API_KEY" in message
        assert "AWS_REGION" in message
        assert "ANTHROPIC_AWS_WORKSPACE_ID" in message


class TestModelIds:
    def test_bedrock_rewrites_to_an_inference_profile(self):
        """Not `anthropic.claude-opus-5`, which Bedrock lists and then refuses to serve.

        Pinned as a literal rather than built from the constant, so that changing the
        constant fails here instead of silently changing every request. The failure this
        guards against is a 400 saying the model does not exist, which reads like a wrong
        model name rather than like the throughput question it actually is.
        """
        assert Route.BEDROCK.model_id("claude-opus-5") == "global.anthropic.claude-opus-5"

    def test_bedrock_does_not_double_prefix(self):
        assert (
            Route.BEDROCK.model_id("global.anthropic.claude-opus-5")
            == "global.anthropic.claude-opus-5"
        )

    def test_bedrock_replaces_a_foundation_model_prefix(self):
        """A caller holding the id Bedrock lists gets the id Bedrock serves."""
        assert Route.BEDROCK.model_id("anthropic.claude-opus-5") == "global.anthropic.claude-opus-5"

    @pytest.mark.parametrize("route", [Route.AWS, Route.FIRST_PARTY])
    def test_the_other_routes_pass_the_id_through(self, route):
        assert route.model_id("claude-opus-5") == "claude-opus-5"


class TestSDKConstruction:
    def test_bedrock_without_a_region_fails_before_any_request(self):
        with pytest.raises(LLMConfigurationError, match="needs AWS_REGION"):
            build_sdk_client(Route.BEDROCK, {})

    def test_the_aws_route_names_every_missing_variable(self):
        with pytest.raises(LLMConfigurationError) as caught:
            build_sdk_client(Route.AWS, {})
        assert "AWS_REGION and ANTHROPIC_AWS_WORKSPACE_ID" in str(caught.value)

    def test_the_aws_route_names_only_what_is_missing(self):
        with pytest.raises(LLMConfigurationError, match="needs ANTHROPIC_AWS_WORKSPACE_ID"):
            build_sdk_client(Route.AWS, BEDROCK_ENV)


class TestRequestShaping:
    def client(self, route=Route.FIRST_PARTY, **kwargs):
        sdk = StubSDKClient()
        return AnthropicModelClient(route=route, sdk_client=sdk, **kwargs), sdk

    def test_satisfies_the_protocol(self):
        client, _ = self.client()
        assert isinstance(client, ModelClient)

    def test_sends_adaptive_thinking_and_an_effort(self):
        client, sdk = self.client()
        client.complete(system="s", messages=[{"role": "user", "content": "hi"}], effort="low")
        request = sdk.requests[0]
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"]["effort"] == "low"
        assert "format" not in request["output_config"]

    def test_an_output_schema_becomes_a_json_schema_format(self):
        client, sdk = self.client()
        schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
        client.complete(system="s", messages=[], output_schema=schema)
        assert sdk.requests[0]["output_config"]["format"] == {
            "type": "json_schema",
            "schema": schema,
        }

    def test_translates_the_model_id_for_the_route(self):
        client, sdk = self.client(route=Route.BEDROCK, model="claude-sonnet-5")
        client.complete(system="s", messages=[])
        assert sdk.requests[0]["model"] == "global.anthropic.claude-sonnet-5"

    def test_concatenates_text_blocks_and_drops_thinking(self):
        client, _ = self.client()
        response = client.complete(system="s", messages=[])
        assert response.text == "hello world"

    def test_carries_usage_through(self):
        client, _ = self.client()
        usage = client.complete(system="s", messages=[]).usage
        assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (11, 7, 3)


class TestAgentSDKEnvironment:
    """What a Claude Agent SDK target needs to reach the model, per route.

    The targets authenticate through the Agent SDK rather than through `AnthropicModelClient`,
    so this is the second half of the same routing question and the half that is easy to leave
    to an operator's shell.
    """

    def test_bedrock_switches_the_sdk_and_carries_the_region(self):
        env = agent_sdk_env(Route.BEDROCK, {"AWS_REGION": "ap-south-1"})
        assert env == {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "ap-south-1"}

    def test_bedrock_accepts_the_default_region_variable(self):
        env = agent_sdk_env(Route.BEDROCK, {"AWS_DEFAULT_REGION": "ap-south-1"})
        assert env["AWS_REGION"] == "ap-south-1"

    def test_bedrock_without_a_region_is_refused(self):
        with pytest.raises(LLMConfigurationError, match="needs AWS_REGION"):
            agent_sdk_env(Route.BEDROCK, {})

    def test_the_aws_route_is_refused_rather_than_left_to_fail_mid_conversation(self):
        """The Agent SDK cannot reach Claude Platform on AWS.

        A target served under it would answer its first turn with an error, which the runner
        would record against whichever attack was running. Refused at construction instead.
        """
        with pytest.raises(LLMConfigurationError, match="cannot use the aws route"):
            agent_sdk_env(
                Route.AWS, {"AWS_REGION": "ap-south-1", "ANTHROPIC_AWS_WORKSPACE_ID": "w"}
            )

    def test_the_first_party_route_needs_nothing_added(self):
        assert agent_sdk_env(Route.FIRST_PARTY, {"ANTHROPIC_API_KEY": "k"}) == {}


class TestRetries:
    """A throttled route is waited out; a rejected request is not.

    The distinction matters more than the retrying does. Four conversations running at once
    against on-demand capacity will be throttled, and without this a 429 kills whichever
    attack was unlucky. But retrying a 400 five times turns a clear error into a slow one,
    and retrying anything at all without jitter reproduces the burst that caused the problem.
    """

    def client(self, failures, **kwargs):
        sdk = StubSDKClient(failures=failures)
        waits: list[float] = []
        client = AnthropicModelClient(
            route=Route.FIRST_PARTY,
            env=FIRST_PARTY_ENV,
            sdk_client=sdk,
            sleep=waits.append,
            **kwargs,
        )
        return client, sdk, waits

    def test_a_throttled_request_is_sent_again_and_succeeds(self):
        client, sdk, waits = self.client([StubAPIError(429), StubAPIError(529)])
        response = client.complete(system="s", messages=[])
        assert response.text == "hello world"
        assert len(sdk.requests) == 3
        assert len(waits) == 2

    def test_the_response_reports_how_many_retries_it_cost(self):
        """A suite that was throttled and one that was not must not look the same."""
        client, _, _ = self.client([StubAPIError(429)])
        assert client.complete(system="s", messages=[]).retries == 1

    def test_a_clean_response_reports_no_retries(self):
        client, _, _ = self.client([])
        assert client.complete(system="s", messages=[]).retries == 0

    def test_a_rejected_request_is_not_retried(self):
        """A 400 fails identically next time. Retrying it only delays the message."""
        client, sdk, waits = self.client([StubAPIError(400)])
        with pytest.raises(StubAPIError):
            client.complete(system="s", messages=[])
        assert len(sdk.requests) == 1
        assert waits == []

    def test_the_error_is_raised_once_the_attempts_are_spent(self):
        """Never a substituted response: an attack turn that was not composed must not be
        gradeable as though it had been."""
        client, sdk, _ = self.client([StubAPIError(429)] * 5, max_attempts=3)
        with pytest.raises(StubAPIError):
            client.complete(system="s", messages=[])
        assert len(sdk.requests) == 3

    def test_max_attempts_of_one_disables_retrying(self):
        client, sdk, _ = self.client([StubAPIError(429)], max_attempts=1)
        with pytest.raises(StubAPIError):
            client.complete(system="s", messages=[])
        assert len(sdk.requests) == 1


class TestBackoff:
    def test_it_grows_and_is_capped(self):
        full = {"jitter": lambda: 1.0}
        assert backoff_seconds(0, **full) == 1.0
        assert backoff_seconds(1, **full) == 2.0
        assert backoff_seconds(2, **full) == 4.0
        assert backoff_seconds(20, **full) == 30.0

    def test_jitter_spreads_the_retry(self):
        """Without it, everything throttled by one burst retries in lockstep."""
        assert backoff_seconds(3, jitter=lambda: 0.0) == 0.0
        assert backoff_seconds(3, jitter=lambda: 0.5) == 4.0

    def test_a_connection_failure_is_retryable_without_a_status(self):
        assert is_retryable(type("APIConnectionError", (Exception,), {})())

    def test_an_arbitrary_error_is_not_retryable(self):
        assert not is_retryable(ValueError("nothing to do with the network"))


class TestStructuredOutputPerRoute:
    """A schema reaches the model by whichever mechanism the route actually supports.

    Bedrock rejects `output_config.format` outright and supports forcing a single tool call
    instead. The difference is confined to this module: every caller gets back a JSON string
    it parses the same way, whichever route served it.
    """

    SCHEMA: ClassVar[dict] = {"type": "object", "properties": {"turn": {"type": "string"}}}

    def client(self, route, env, sdk=None):
        sdk = sdk or StubSDKClient()
        return AnthropicModelClient(route=route, env=env, sdk_client=sdk), sdk

    def test_a_supporting_route_sends_the_schema_directly(self):
        client, sdk = self.client(Route.FIRST_PARTY, FIRST_PARTY_ENV)
        client.complete(system="s", messages=[], output_schema=self.SCHEMA)
        assert sdk.requests[0]["output_config"]["format"]["schema"] == self.SCHEMA
        assert "tools" not in sdk.requests[0]

    def test_bedrock_forces_a_tool_call_instead(self):
        """`output_config.format` is a 400 there, and a 400 is not retryable."""
        client, sdk = self.client(Route.BEDROCK, BEDROCK_ENV)
        client.complete(system="s", messages=[], output_schema=self.SCHEMA)
        request = sdk.requests[0]
        assert "format" not in request["output_config"]
        assert request["tools"][0]["input_schema"] == self.SCHEMA
        assert request["tool_choice"] == {"type": "tool", "name": SCHEMA_TOOL}

    def test_no_schema_means_no_tool_on_either_route(self):
        for route, env in ((Route.FIRST_PARTY, FIRST_PARTY_ENV), (Route.BEDROCK, BEDROCK_ENV)):
            client, sdk = self.client(route, env)
            client.complete(system="s", messages=[])
            assert "tools" not in sdk.requests[0]
            assert "format" not in sdk.requests[0]["output_config"]

    def test_a_forced_tool_answer_comes_back_as_json_text(self):
        """Callers parse a string. They must not have to know which route served them."""
        sdk = StubSDKClient(tool_input={"stop": False, "turn": "hello"})
        client, _ = self.client(Route.BEDROCK, BEDROCK_ENV, sdk)
        response = client.complete(system="s", messages=[], output_schema=self.SCHEMA)
        assert json.loads(response.text) == {"stop": False, "turn": "hello"}

    def test_the_capability_is_named_on_the_route(self):
        assert not Route.BEDROCK.supports_output_format
        assert Route.FIRST_PARTY.supports_output_format
        assert Route.AWS.supports_output_format
