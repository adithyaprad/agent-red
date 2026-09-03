"""Genericity, enforced continuously rather than asserted once.

The central claim of this project is that the attack suite is a function of the agent under
test rather than of whoever wrote it. That claim does not fail all at once. It fails one
convenient special case at a time: a variable called `discount_pct` because that is what the
first target happened to bound, a comment explaining a check "for refunds", an example in a
docstring that only makes sense in a shop. Each is harmless and the accumulation is fatal,
because the suite quietly becomes a commerce red-team that happens to read a config.

So it is a build failure the hour it happens rather than a discovery on the day the harness
is pointed at an agent that sells insurance.

**What is checked.** Every module under `attacks/`, `judge/detectors/` and `scoring/`, and
the technique corpus in `data/techniques/`. Those are the parts that must work unchanged
against an agent nobody has seen. `targets/` is exempt and must be: it is a merchant agent,
it sells furniture, and it is not part of the product surface. `spec/` is exempt because it
names the shapes a merchant declares, not the merchant's domain.

`scoring/` is here for a reason that took a while to see. It renders the page a person reads,
and writing prose for a merchant is the single activity most likely to produce a sentence that
only makes sense in a shop. It is also the place where a leak does the most damage, because a
page that says "refunds and discounts" to an agent that handles neither is not a stylistic
problem, it is a product that visibly does not work on the agent in front of you. Adding this
directory caught four leaks on its first run, one of them a hardcoded pair of result field
names that made the headline figure silently zero for any agent that named the field
differently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

GENERIC_SOURCE_DIRS = (
    ROOT / "src" / "agentred" / "attacks",
    ROOT / "src" / "agentred" / "judge" / "detectors",
    ROOT / "src" / "agentred" / "mcp" / "generator",
    ROOT / "src" / "agentred" / "scoring",
)

GENERIC_DATA_DIRS = (ROOT / "data" / "techniques",)

BANNED = (
    "basket",
    "cancel",
    "cart",
    "catalog",
    "catalogue",
    "checkout",
    "courier",
    "coupon",
    "credit",
    "customer",
    "delivery",
    "discount",
    "dispatch",
    "invoice",
    "merchandise",
    "order",
    "payment",
    "price",
    "pricing",
    "product",
    "purchase",
    "refund",
    "retail",
    "shipping",
    "shopper",
    "sku",
    "stock",
    "storefront",
    "voucher",
    "warehouse",
)
"""Words that name the domain the first two targets happen to occupy.

Not an exhaustive list of commerce vocabulary and not meant to be. It is the set that would
actually appear if someone wrote a check for the agent in front of them, which is the failure
this guards against.
"""

ALLOWED_SUBSTRINGS = (
    "in order to",
    "in order for",
    "reordering",
    "reorder",
    "ordering",
    "ordered",
    "border",
    "recorded",
    "recording",
    "record",
    "order by",
)
"""Phrases and words that legitimately contain a banned word.

`in order to`, `ordering` and `recorded` are ordinary English about purpose, sequence and
evidence, and all three are things this codebase genuinely needs to say. `order by` is SQL and
has no domain reading at all. They are stripped
before the search rather than excused after it, so a real `order` sitting next to a
legitimate `recorded` in the same file is still caught.

Every entry here is a small hole in the guard, so the list is kept short and each addition
has to be a phrase with no domain reading at all. `order total` would never qualify.
"""

WORD = re.compile(r"[a-z]+")


def offending_words(text: str) -> list[str]:
    """Every banned word in `text`, lowercased, in the sequence it appears.

    Matching is on whole words after removing the allowed substrings, so `reordering` does
    not report `order` but `order` on its own does. A trailing `s` is stripped before the
    comparison, because the plural is the form that actually shows up in a variable name.
    """
    lowered = text.lower()
    for allowed in ALLOWED_SUBSTRINGS:
        lowered = lowered.replace(allowed, " ")
    banned = set(BANNED)
    found = []
    for word in WORD.findall(lowered):
        if word in banned:
            found.append(word)
        elif word.endswith("s") and word[:-1] in banned:
            found.append(word[:-1])
    return found


def files_under(directories, suffix: str) -> list[Path]:
    return sorted(
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts
    )


class TestTheGuardItself:
    """A guard nobody tests is a guard that stops working silently."""

    def test_catches_a_banned_word(self):
        assert offending_words("the discount ceiling") == ["discount"]

    def test_catches_it_in_a_comment_or_a_name(self):
        assert offending_words("# checks refund_amount") == ["refund"]

    def test_is_case_insensitive(self):
        assert offending_words("Refund") == ["refund"]

    def test_does_not_fire_on_a_legitimate_containing_word(self):
        assert offending_words("in filename ordering, recorded as evidence") == []

    def test_does_not_fire_on_in_order_to(self):
        """Ordinary English about purpose, and unavoidable in prose about technique."""
        assert offending_words("it concedes the history in order to be kind") == []

    def test_still_catches_a_real_use_beside_a_legitimate_one(self):
        assert offending_words("recorded in filename ordering, then the order total") == ["order"]

    def test_a_plural_is_caught(self):
        assert offending_words("customers and orders") == ["customer", "order"]

    def test_finds_the_directories_it_guards(self):
        """If the tree is rearranged and this test stops looking anywhere, it must fail."""
        assert any(directory.is_dir() for directory in GENERIC_SOURCE_DIRS)
        assert all(directory.is_dir() for directory in GENERIC_DATA_DIRS)


class TestGenericModules:
    @pytest.mark.parametrize("path", files_under(GENERIC_SOURCE_DIRS, ".py"), ids=lambda p: p.name)
    def test_module_names_no_domain(self, path: Path):
        found = offending_words(path.read_text(encoding="utf-8"))
        assert not found, (
            f"{path.relative_to(ROOT)} contains domain vocabulary {sorted(set(found))}. "
            f"This module has to work unchanged against an agent that sells something else. "
            f"Derive it from the spec instead, or move it to targets/."
        )


class TestGenericCorpus:
    @pytest.mark.parametrize("path", files_under(GENERIC_DATA_DIRS, ".yaml"), ids=lambda p: p.name)
    def test_technique_names_no_domain(self, path: Path):
        found = offending_words(path.read_text(encoding="utf-8"))
        assert not found, (
            f"{path.relative_to(ROOT)} contains domain vocabulary {sorted(set(found))}. "
            f"A technique describes a shape of pressure. What it is applied to comes from "
            f"the agent's own spec."
        )

    def test_the_corpus_is_actually_being_checked(self):
        """Guards against the parametrised cases silently collecting nothing."""
        assert len(files_under(GENERIC_DATA_DIRS, ".yaml")) >= 8
