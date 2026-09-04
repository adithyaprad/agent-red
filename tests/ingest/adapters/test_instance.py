"""Reading the limits and the data access a no-code builder stored.

Two things here are worth a test rather than a read-through. A limit entry that sets no limit
is a form somebody half filled in, and reading it as an unbounded argument would turn an
abandoned intention into a declared permission. And a granted source that says what it is keyed
by has answered more than one that only says its name: a channel plants into a record, so a
source that cannot name its records is a source nothing can be planted into.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentred.ingest.adapters.instance import InstanceReadError, read_instance


def written(tmp_path: Path, body: dict[str, object]) -> Path:
    """One instance configuration on disk."""
    path = tmp_path / "instance.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_a_stored_limit_becomes_a_bound_in_the_operators_own_words(tmp_path: Path) -> None:
    package = read_instance(
        written(
            tmp_path,
            {
                "instance_id": "an_agent",
                "limits": [
                    {
                        "action": "refund_charge",
                        "field": "amount",
                        "max": 5000,
                        "min": 0,
                        "label": "One month's box back.",
                    }
                ],
            },
        )
    )

    (recovered,) = package.rules
    assert recovered.rule.tool == "refund_charge"
    assert recovered.rule.argument == "amount"
    assert recovered.rule.maximum == 5000
    assert recovered.rule.description == "One month's box back."


def test_a_limit_that_sets_no_limit_is_refused(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        {"instance_id": "an_agent", "limits": [{"action": "refund_charge", "field": "amount"}]},
    )

    with pytest.raises(InstanceReadError, match="neither a maximum nor a minimum"):
        read_instance(path)


def test_a_source_listed_by_name_alone_is_read_as_a_name(tmp_path: Path) -> None:
    package = read_instance(
        written(
            tmp_path,
            {
                "instance_id": "an_agent",
                "data_access": {"sources": ["subscriptions"], "identifiers": ["subscription_id"]},
            },
        )
    )

    (source,) = package.data_sources
    assert source.name == "subscriptions"
    assert source.identifier_kinds == ()


def test_a_source_that_says_what_it_is_keyed_by_carries_it(tmp_path: Path) -> None:
    package = read_instance(
        written(
            tmp_path,
            {
                "instance_id": "an_agent",
                "data_access": {
                    "sources": [
                        {
                            "name": "subscriptions",
                            "label": "Live and cancelled subscriptions.",
                            "identifiers": ["subscription_id", "customer_id"],
                        }
                    ],
                    "identifiers": ["subscription_id"],
                },
            },
        )
    )

    (source,) = package.data_sources
    assert source.identifier_kinds == ("subscription_id", "customer_id")
    assert source.description == "Live and cancelled subscriptions."


def test_a_granted_source_naming_nothing_is_refused(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        {"instance_id": "an_agent", "data_access": {"sources": [{"identifiers": ["a_kind"]}]}},
    )

    with pytest.raises(InstanceReadError, match="names nothing"):
        read_instance(path)


def test_granting_access_without_saying_what_one_session_covers_stays_a_question(
    tmp_path: Path,
) -> None:
    package = read_instance(
        written(tmp_path, {"instance_id": "an_agent", "data_access": {"sources": ["orders"]}})
    )

    assert package.data_scope is not None
    assert not package.data_scope.resolved


def test_the_shapes_no_form_field_holds_are_named_on_every_read(tmp_path: Path) -> None:
    package = read_instance(written(tmp_path, {"instance_id": "an_agent"}))

    assert len(package.notes) == 3
    assert any("added up" in note for note in package.notes)
