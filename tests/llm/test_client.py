"""Route resolution and request shaping. Offline: the SDK is stubbed."""

import pytest

from agentred.llm.client import (
    AnthropicModelClient,
    LLMConfigurationError,
    ModelClient,
    Route,
    build_sdk_client,
    resolve_route,
)
from tests.fakes.model import StubSDKClient

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
    def test_bedrock_prefixes(self):
        assert Route.BEDROCK.model_id("claude-opus-5") == "anthropic.claude-opus-5"

    def test_bedrock_does_not_double_prefix(self):
        assert Route.BEDROCK.model_id("anthropic.claude-opus-5") == "anthropic.claude-opus-5"

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
        assert sdk.requests[0]["model"] == "anthropic.claude-sonnet-5"

    def test_concatenates_text_blocks_and_drops_thinking(self):
        client, _ = self.client()
        response = client.complete(system="s", messages=[])
        assert response.text == "hello world"

    def test_carries_usage_through(self):
        client, _ = self.client()
        usage = client.complete(system="s", messages=[]).usage
        assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (11, 7, 3)
