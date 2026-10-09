"""Exact declared-domain source predicates and unsupported-expression refusals."""

import pytest

from merlin.targetgen.rtl.source_predicates import SourcePredicateError, source_predicate


def test_selected_operand_comparisons_boolean_precedence_and_multiline_ranges():
    source = """object Public {
 val selected = (cmd.code >= START && cmd.code <= END) ||
   (cmd.code === EXTRA && cmd.code =/= END)
}
"""
    actual = source_predicate(
        source,
        binding="selected",
        operand="cmd.code",
        declarations={1: "START", 4: "MID", 7: "END", 11: "EXTRA", 19: "OTHER"},
    )
    assert actual["selected_source_values"] == [1, 4, 7, 11]
    assert actual["symbols"] == ["END", "EXTRA", "START"]
    assert actual["source_lines"] == [2, 3]
    assert "complete_hardware_domain" not in actual


def test_boolean_precedence_comes_from_actual_source():
    source = "val selected = code === FIRST || code === SECOND && code === THIRD"
    actual = source_predicate(
        source, binding="selected", operand="code", declarations={2: "FIRST", 6: "SECOND", 9: "THIRD"}
    )
    assert actual["selected_source_values"] == [2]


@pytest.mark.parametrize(
    "expression",
    [
        "other === FIRST",
        "code === MISSING",
        "code === 7.U",
        "code === FIRST; launch()",
        "code === FIRST // a comment",
        "code == FIRST",
        "code >= FIRST &&",
        "helper(code)",
        "(code === FIRST",
        "code === FIRST || true.B",
    ],
)
def test_unknown_or_effectful_source_expressions_refuse(expression):
    with pytest.raises(SourcePredicateError):
        source_predicate("val selected = " + expression, binding="selected", operand="code", declarations={2: "FIRST"})


def test_ambiguous_binding_or_symbol_domain_refuses():
    with pytest.raises(SourcePredicateError, match="exactly once"):
        source_predicate(
            "val selected = code === FIRST\nval selected = code === FIRST",
            binding="selected",
            operand="code",
            declarations={2: "FIRST"},
        )
    with pytest.raises(SourcePredicateError, match="ambiguous"):
        source_predicate(
            "val selected = code === FIRST", binding="selected", operand="code", declarations={2: "FIRST", 3: "FIRST"}
        )


@pytest.mark.parametrize(
    "source",
    [
        "/*\nval selected = code === FIRST\n*/",
        'val documentation = """\nval selected = code === FIRST\n"""',
        "/* outer /* nested */\nval selected = code === FIRST\n*/",
    ],
)
def test_commented_or_quoted_text_cannot_impersonate_a_selected_source_binding(source):
    with pytest.raises(SourcePredicateError, match="exactly once"):
        source_predicate(source, binding="selected", operand="code", declarations={2: "FIRST"})


def test_original_active_binding_and_line_spans_survive_unrelated_nested_comments_and_strings():
    source = (
        'val documentation = """\nval selected = code === FIRST\n"""\n'
        "/* outer /* nested */\nval selected = code === FIRST\n*/\n"
        'val quote = "\\\" /* quoted, not a comment */"\n'
        "val selected = code === SECOND\n"
    )
    actual = source_predicate(source, binding="selected", operand="code", declarations={2: "FIRST", 4: "SECOND"})
    assert actual["selected_source_values"] == [4]
    assert actual["source_lines"] == [8, 8]


@pytest.mark.parametrize("source", ['val text = """unclosed', "/* unclosed", 'val text = "unclosed'])
def test_unclosed_lexical_text_refuses_source_membership(source):
    with pytest.raises(SourcePredicateError, match="unclosed"):
        source_predicate(source, binding="selected", operand="code", declarations={2: "FIRST"})
