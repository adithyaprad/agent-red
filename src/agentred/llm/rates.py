"""What a run cost, in money rather than tokens.

Tokens are recorded on both sides of every turn: the attacker and the judge in the call
recording, the target in its own reply. Neither is a bill. This turns one into the other.

Two decisions are worth stating, because the cheap version of each is wrong in a way that is
hard to notice afterwards.

An unknown model raises rather than pricing at zero. A rate table is a list of strings that
goes stale the moment a model is added, and the failure mode of a lookup that returns zero on
a miss is a run that reports two dollars because half its calls were priced at nothing. A
missing rate is a fact about this file, and it says so.

A partner route is priced at first-party rates and labelled as an estimate. Amazon Bedrock and
Vertex are billed by the partner at their own rates, which are not these, and are not
discoverable from a response. Reporting a first-party figure without saying so would be
reporting a number that is not true. See `RATE_SOURCE_ESTIMATED`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentred.llm.client import BEDROCK_PREFIX, Route

RATE_SOURCE_EXACT = "first-party rates, billed at these rates"
"""What the figure means on a route Anthropic bills directly."""

RATE_SOURCE_ESTIMATED = "first-party rates, actual route is partner-billed at its own rates"
"""What the figure means on Bedrock or Vertex. An estimate, and it says which direction of
error it cannot rule out: partner rates are set by the partner and are not in this file."""


@dataclass(frozen=True)
class Rate:
    """USD per million tokens for one model.

    Attributes:
        input_usd: Uncached input.
        output_usd: Output.
        cache_read_usd: Input served from the prompt cache.
        cache_write_usd: Input written to the prompt cache.
    """

    input_usd: float
    output_usd: float
    cache_read_usd: float
    cache_write_usd: float


RATES: dict[str, Rate] = {
    # Cache reads are a tenth of input and cache writes are 1.25 times it, on every model
    # here, so those two columns are derived rather than independently sourced.
    "claude-sonnet-5": Rate(2.00, 10.00, 0.20, 2.50),
    "claude-opus-5": Rate(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": Rate(5.00, 25.00, 0.50, 6.25),
    "claude-haiku-4-5": Rate(1.00, 5.00, 0.10, 1.25),
    "claude-fable-5-1": Rate(10.00, 50.00, 1.00, 12.50),
}
"""First-party USD per million tokens, as published. Both targets and the attacker run
`claude-sonnet-5`; the rest are here so a route change does not silently become unpriced."""


class UnpricedModelError(LookupError):
    """A model with no rate in `RATES`.

    Raised rather than returning zero. A bill that reads low because a model was missing is
    indistinguishable from a cheap run, and the only difference is which one is true.
    """


def canonical_model(model: str) -> str:
    """Strip a route's prefix off a model id, so one rate table serves every route.

    Args:
        model: A model id as a response reported it, which on Bedrock carries an inference
            profile prefix.

    Returns:
        The first-party id the rate table is keyed by.
    """
    stripped = model.strip()
    if stripped.startswith(BEDROCK_PREFIX):
        return stripped[len(BEDROCK_PREFIX) :]
    return stripped


def rate_for(model: str) -> Rate:
    """The rate for a model.

    Args:
        model: A model id, in any route's form.

    Returns:
        Its rate.

    Raises:
        UnpricedModelError: If the model has no rate here.
    """
    canonical = canonical_model(model)
    rate = RATES.get(canonical)
    if rate is None:
        known = ", ".join(sorted(RATES))
        raise UnpricedModelError(
            f"no rate for model {canonical!r} (from {model!r}). Add one to "
            f"agentred.llm.rates.RATES. Known: {known}."
        )
    return rate


def cost_usd(model: str, usage: dict[str, float]) -> float:
    """What one call's tokens cost.

    Args:
        model: The model the call ran on, in any route's form.
        usage: Token counts. `input_tokens`, `output_tokens`, `cache_read_tokens` and
            `cache_write_tokens` are read; anything else is ignored, and an absent key
            counts as zero because a response that reports no cache write did not make one.

    Returns:
        USD, unrounded. Rounding is the report's job, because a suite's total is the sum of
        its calls and rounding each one first makes the total wrong by the count.

    Raises:
        UnpricedModelError: If the model has no rate.
    """
    rate = rate_for(model)
    million = 1_000_000.0
    return (
        float(usage.get("input_tokens", 0) or 0) * rate.input_usd
        + float(usage.get("output_tokens", 0) or 0) * rate.output_usd
        + float(usage.get("cache_read_tokens", 0) or 0) * rate.cache_read_usd
        + float(usage.get("cache_write_tokens", 0) or 0) * rate.cache_write_usd
    ) / million


def rate_source(route: Route) -> str:
    """Whether a figure on this route is a bill or an estimate.

    Args:
        route: The resolved model route.

    Returns:
        `RATE_SOURCE_EXACT` on the routes Anthropic operates and bills, which is the first
        party API and Claude Platform on AWS. `RATE_SOURCE_ESTIMATED` on Bedrock, which is
        partner-operated. Carried onto every cost report, because a dollar figure with no
        provenance invites being quoted as one.
    """
    return RATE_SOURCE_EXACT if route in {Route.FIRST_PARTY, Route.AWS} else RATE_SOURCE_ESTIMATED
