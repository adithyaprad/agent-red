"""Model access.

One environment switch across Bedrock, Claude Platform on AWS and the first-party API,
behind an interface narrow enough to fake faithfully in tests.
"""

from agentred.llm.client import (
    DEFAULT_MODEL,
    AnthropicModelClient,
    LLMConfigurationError,
    ModelClient,
    ModelResponse,
    Route,
    Usage,
    resolve_route,
)

__all__ = [
    "DEFAULT_MODEL",
    "AnthropicModelClient",
    "LLMConfigurationError",
    "ModelClient",
    "ModelResponse",
    "Route",
    "Usage",
    "resolve_route",
]
