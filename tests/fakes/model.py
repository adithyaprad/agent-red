"""Model fakes. No test in this repository is allowed to reach the network.

`RecordedModelClient` replays responses in order and records what it was asked, so a test
can assert on the prompt that was built as well as on what was done with the reply.
`StubSDKClient` sits one layer lower, standing in for the Anthropic SDK itself so that the
request-shaping code in `AnthropicModelClient` is exercised for real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentred.llm.client import ModelResponse, Usage


@dataclass
class RecordedCall:
    """One call made to a fake client."""

    system: str
    messages: list[dict[str, Any]]
    max_tokens: int
    effort: str
    output_schema: dict[str, Any] | None


@dataclass
class RecordedModelClient:
    """A `ModelClient` that replays a fixed list of replies.

    Attributes:
        replies: Response texts to return, in order.
        calls: Every call made, in order, for assertions.
    """

    replies: list[str] = field(default_factory=list)
    calls: list[RecordedCall] = field(default_factory=list)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 8_000,
        effort: str = "high",
        output_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.calls.append(
            RecordedCall(
                system=system,
                messages=list(messages),
                max_tokens=max_tokens,
                effort=effort,
                output_schema=output_schema,
            )
        )
        if not self.replies:
            raise AssertionError(
                f"RecordedModelClient ran out of replies on call {len(self.calls)}"
            )
        return ModelResponse(
            text=self.replies.pop(0),
            stop_reason="end_turn",
            model="fake-model",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Usage:
    def __init__(self) -> None:
        self.input_tokens = 11
        self.output_tokens = 7
        self.cache_read_input_tokens = 3


class _Message:
    def __init__(self, blocks: list[_Block], model: str) -> None:
        self.content = blocks
        self.stop_reason = "end_turn"
        self.model = model
        self.usage = _Usage()


class _Messages:
    def __init__(self, owner: StubSDKClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _Message:
        self._owner.requests.append(kwargs)
        return _Message(
            [_Block("thinking", "ignored"), _Block("text", "hello"), _Block("text", " world")],
            kwargs["model"],
        )


class StubSDKClient:
    """Stands in for an Anthropic SDK client, recording the request it was handed.

    Attributes:
        requests: The keyword arguments of every `messages.create` call.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self)
