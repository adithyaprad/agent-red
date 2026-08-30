"""Loading specs from disk. Offline: every fixture is written into tmp_path."""

import pytest

from agentred.spec import SpecError, load_spec, load_spec_dir

CONFIG = """
agent_id: cart-recovery
version: "1"
model: claude-sonnet-5
instructions: Help the customer complete their purchase.
tools:
  - name: apply_discount
    description: Apply a percentage discount to the open cart.
    consequence: money
    parameters:
      type: object
      properties:
        pct: {type: number}
  - name: lookup_cart
    consequence: inert
    parameters:
      type: object
      properties:
        cart_id: {type: string}
data_sources:
  - name: carts
    description: Open carts.
    identifier_kinds: [cart_id, customer_id]
"""

POLICY = """
agent_id: cart-recovery
version: "1"
bounds:
  - kind: numeric
    name: discount_ceiling
    tool: apply_discount
    argument: pct
    maximum: 10
preconditions:
  - name: discount_needs_a_cart
    tool: apply_discount
    requires: lookup_cart
data_scope:
  sources: [carts]
  subject_identifier_kinds: [customer_id]
"""


SUBJECTS = """
subjects:
  - name: someone
    identifiers:
      customer_id: C-1
    facts:
      - They have been waiting a week.
"""


@pytest.fixture
def spec_dir(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG)
    (tmp_path / "policy.yaml").write_text(POLICY)
    (tmp_path / "subjects.yaml").write_text(SUBJECTS)
    return tmp_path


def test_loads_a_valid_spec_directory(spec_dir):
    spec = load_spec_dir(spec_dir)
    assert spec.config.agent_id == "cart-recovery"
    assert [t.name for t in spec.config.consequential_tools] == ["apply_discount"]
    assert spec.policy.bounds[0].maximum == 10
    assert spec.policy.data_scope.sources == ("carts",)
    assert spec.version_tuple.model_version == "claude-sonnet-5"
    assert spec.ungated_consequential_tools() == ()


def test_loads_explicit_paths(spec_dir):
    spec = load_spec(spec_dir / "config.yaml", spec_dir / "policy.yaml", spec_dir / "subjects.yaml")
    assert spec.policy.is_fully_declared


def test_missing_directory(tmp_path):
    with pytest.raises(SpecError, match="is not a directory"):
        load_spec_dir(tmp_path / "nowhere")


def test_missing_file(spec_dir):
    (spec_dir / "policy.yaml").unlink()
    with pytest.raises(SpecError, match=r"policy\.yaml does not exist"):
        load_spec_dir(spec_dir)


def test_unparseable_yaml(spec_dir):
    (spec_dir / "policy.yaml").write_text("bounds: [unclosed\n")
    with pytest.raises(SpecError, match="is not valid YAML"):
        load_spec_dir(spec_dir)


@pytest.mark.parametrize("body", ["", "- a\n- b\n"])
def test_top_level_must_be_a_mapping(spec_dir, body):
    (spec_dir / "config.yaml").write_text(body)
    with pytest.raises(SpecError, match="must contain a mapping"):
        load_spec_dir(spec_dir)


def test_error_names_the_file_and_the_field(spec_dir):
    (spec_dir / "config.yaml").write_text(
        CONFIG.replace("consequence: money", "consequence: vibes")
    )
    with pytest.raises(SpecError) as caught:
        load_spec_dir(spec_dir)
    message = str(caught.value)
    assert "config.yaml is not a valid AgentConfig" in message
    assert "tools.0.consequence" in message


def test_cross_validation_error_names_both_files(spec_dir):
    (spec_dir / "policy.yaml").write_text(POLICY.replace("tool: apply_discount", "tool: refund"))
    with pytest.raises(SpecError) as caught:
        load_spec_dir(spec_dir)
    message = str(caught.value)
    assert "policy.yaml does not describe" in message
    assert "config.yaml" in message


def test_unknown_field_is_rejected_rather_than_ignored(spec_dir):
    (spec_dir / "policy.yaml").write_text(POLICY + "\nrate_limit: 5\n")
    with pytest.raises(SpecError, match="rate_limit"):
        load_spec_dir(spec_dir)


class TestSubjects:
    """An agent scoped to a subject needs somebody real to be, or nothing gets tested."""

    def test_subjects_are_loaded(self, spec_dir):
        spec = load_spec_dir(spec_dir)
        assert [s.name for s in spec.subjects] == ["someone"]
        assert spec.subjects[0].identifiers["customer_id"] == "C-1"
        assert spec.subjects[0].facts == ("They have been waiting a week.",)

    def test_a_scoped_agent_with_no_subjects_is_refused(self, spec_dir):
        """Otherwise the harness opens with an identifier that resolves to nothing.

        The agent then declines to act on a record it cannot find, every rule reports as
        never in play, and the run reads as a clean sheet for an agent nobody questioned.
        """
        (spec_dir / "subjects.yaml").unlink()
        with pytest.raises(SpecError, match="no subjects are declared"):
            load_spec_dir(spec_dir)

    def test_a_subject_missing_a_declared_kind_is_refused(self, spec_dir):
        (spec_dir / "subjects.yaml").write_text(
            "subjects:\n  - name: someone\n    identifiers:\n      email: a@b.test\n"
        )
        with pytest.raises(SpecError, match="declares no customer_id"):
            load_spec_dir(spec_dir)

    def test_an_unscoped_agent_needs_none(self, spec_dir):
        (spec_dir / "subjects.yaml").unlink()
        (spec_dir / "policy.yaml").write_text(
            POLICY.replace("  subject_identifier_kinds: [customer_id]\n", "")
        )
        assert load_spec_dir(spec_dir).subjects == ()

    def test_a_blank_identifier_value_is_refused(self, spec_dir):
        (spec_dir / "subjects.yaml").write_text(
            "subjects:\n  - name: someone\n    identifiers:\n      customer_id: '  '\n"
        )
        with pytest.raises(SpecError, match="no value"):
            load_spec_dir(spec_dir)

    def test_subjects_must_be_a_list(self, spec_dir):
        (spec_dir / "subjects.yaml").write_text("subjects: {name: someone}\n")
        with pytest.raises(SpecError, match="must be a list"):
            load_spec_dir(spec_dir)
