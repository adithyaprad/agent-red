"""What the stand-in agents can actually do, over the synthetic shop in `world.py`.

The tools do not enforce policy. `apply_discount(percent=35)` succeeds against a policy
ceiling of 10, and `issue_refund` pays out without checking that anything was verified.
That is deliberate and it is the whole experiment: the merchant's limits live in the system
prompt, exactly as they do in the agents this harness exists to test, and the question being
asked is whether a conversation can get past them. A tool that refused would be answering
the question by construction and the run would prove nothing.

The tools that cost the merchant something write to `World.ledger`, so what a conversation
did is inspectable from the target as well as from the tool-call log.
"""

from agentred.mcp.tools.base import ToolImplementation, ToolSet
from agentred.mcp.tools.cart import CART_TOOLS
from agentred.mcp.tools.dispute import DISPUTE_TOOLS

TOOLSETS: dict[str, ToolSet] = {
    "cart_recovery": CART_TOOLS,
    "dispute_handler": DISPUTE_TOOLS,
}
"""Implementations by agent id. `runtime.py` refuses to serve an agent with no entry."""

__all__ = ["CART_TOOLS", "DISPUTE_TOOLS", "TOOLSETS", "ToolImplementation", "ToolSet"]
