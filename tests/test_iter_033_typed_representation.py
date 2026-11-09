from __future__ import annotations

import pytest

from scripts.diagnostics.iter_033.typed_scoring import answer_set, atom, score_sets


def test_entity_label_is_not_a_literal_value() -> None:
    assert atom("Entity", "m.0042") != atom("Value", "42")
    result = score_sets({atom("Entity", "m.0042")}, {atom("Value", "42")})
    assert (result.hit, result.precision, result.recall, result.f1) == (0, 0.0, 0.0, 0.0)


def test_exact_typed_scalar_matches_and_numeric_mismatch_fails() -> None:
    assert score_sets({atom("Value", "42")}, {atom("Value", "42")}).hit == 1
    assert score_sets({atom("Value", "13")}, {atom("Value", "6")}).hit == 0


@pytest.mark.parametrize("left,right", [("042", "42"), ("+4", "4"), ("4.0", "4"), ("-1", "1")])
def test_scalar_lexical_values_are_not_silently_normalized(left: str, right: str) -> None:
    assert score_sets({atom("Value", left)}, {atom("Value", right)}).hit == 0


def test_mixed_multi_answer_f1_uses_typed_set_overlap() -> None:
    score = score_sets(
        {atom("Value", "2"), atom("Entity", "m.person")},
        {atom("Value", "2"), atom("Value", "3")},
    )
    assert score.hit == 1
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.f1 == 0.5


def test_duplicates_do_not_change_set_score() -> None:
    one = score_sets([atom("Value", "2")], [atom("Value", "2"), atom("Value", "2")])
    duplicate = score_sets([atom("Value", "2"), atom("Value", "2")], [atom("Value", "2")])
    assert one == duplicate
    assert one.f1 == 1.0


def test_empty_answer_sets_score_zero_under_this_audit_contract() -> None:
    assert score_sets([], []).f1 == 0.0
    assert score_sets([atom("Value", "1")], []).f1 == 0.0


def test_unknown_answer_kind_and_empty_atoms_are_rejected() -> None:
    with pytest.raises(ValueError):
        atom("Unknown", "42")
    with pytest.raises(ValueError):
        atom("Value", "")
    with pytest.raises(ValueError):
        answer_set([("Entity", "")])
