"""The agent-facing verification spec is DERIVED from the answer-free capsule declarations and carries no
golden. Hermetic: a synthetic corpus with a golden.yaml sitting beside the capsule proves the generator
never lets an output value into the spec. Plus a gated check that a real target's ops/dtypes/policy derive.
"""

from __future__ import annotations

import types

import pytest
import yaml

from merlin.targetgen import verification_spec as VS


def _synth_te(root):
    return types.SimpleNamespace(
        target="synth", capsule_corpus=root / "isa", corpus_siblings=lambda: [], isa_headers=["docs/isa.md"]
    )


def test_spec_is_answer_free_even_with_a_golden_beside_the_capsule(tmp_path):
    cap = tmp_path / "isa" / "T0_matmul"
    cap.mkdir(parents=True)
    (cap / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "T0_matmul",
                "label": "public",
                "inputs": [
                    {"name": "A", "role": "input", "dtype": "i8"},
                    {"name": "W", "role": "weight", "dtype": "i8"},
                ],
                "operation": {"op": "matmul", "attributes": {"output_dtype": "i32", "epilogue": ["relu"]}},
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
                "expected": {"instruction_classes": ["MVIN", "MATMUL", "MVOUT"]},
            }
        )
    )
    # the ANSWER KEY sits right beside it — the generator must never read or echo it.
    (cap / "golden.yaml").write_text(yaml.safe_dump({"golden_source": "x", "outputs": {"Y": [[424242, 424242]]}}))

    te = _synth_te(tmp_path)
    spec = VS.build_spec(te)
    md = VS.render_markdown(te)

    # the CONTRACT is present: op, dtypes, acceptance policy, epilogue, coverage
    assert "matmul" in spec["ops"]
    assert spec["ops"]["matmul"]["dtypes"] == ["i8 -> i32"]
    assert any("exact_int" in a for a in spec["ops"]["matmul"]["accept"])
    assert "relu" in spec["ops"]["matmul"]["epilogues"]
    assert "MVIN" in spec["ops"]["matmul"]["coverage"] and "MVOUT" in spec["ops"]["matmul"]["coverage"]
    # the ANSWER is absent: no golden output value, no outputs payload key
    assert "424242" not in md
    assert "outputs:" not in md
    # 'golden' appears only as the reframing prose ("no answer key / no stored golden"), never a value
    assert "no answer key" in md.lower()


def test_hidden_labelled_capsule_is_excluded(tmp_path):
    for name, label in (("P0", "public"), ("H0", "hidden")):
        d = tmp_path / "isa" / name
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "label": label,
                    "inputs": [{"name": "A", "dtype": "i8"}],
                    "operation": {"op": "matmul", "attributes": {"output_dtype": "i32"}},
                    "numeric_policy": {"compare": "exact_int"},
                }
            )
        )
    spec = VS.build_spec(_synth_te(tmp_path))
    assert spec["n_capsules"] == 1  # the hidden-labelled capsule is not counted


def test_real_target_ops_and_policy_derive():
    """A real target's spec derives its ops + acceptance policy from its committed corpus, no code change."""
    from merlin.common.paths import merlin_dir
    from merlin.targetgen.target_experiment import load_target_experiment

    p = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    if not p.is_file():
        pytest.skip("gemmini descriptor absent")
    spec = VS.build_spec(load_target_experiment(p))
    if not spec["ops"]:
        pytest.skip("gemmini corpus not generated")
    assert "matmul" in spec["ops"]
    assert any("exact_int" in a for a in spec["ops"]["matmul"]["accept"])
    md = VS.render_markdown(load_target_experiment(p))
    assert "acceptance contract for `gemmini`" in md and "outputs:" not in md


# --------------------------------------------------------------------------- the JSON sibling


def _schema():
    import json

    from merlin.common.paths import merlin_dir

    return json.loads((merlin_dir() / "contract/schemas/verification_spec.schema.json").read_text(encoding="utf-8"))


def test_the_json_sibling_is_written_and_schema_valid(tmp_path):
    """The markdown is for the agent to read; the JSON is what a checker or the agent's own tooling can
    consume without parsing prose. There was no JSON sibling and no schema, so nothing downstream could
    read the acceptance contract at all."""
    import json

    import jsonschema

    cap = tmp_path / "isa" / "T0_matmul"
    cap.mkdir(parents=True)
    (cap / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "T0_matmul",
                "label": "public",
                "inputs": [{"name": "A", "role": "input", "dtype": "i8"}],
                "operation": {"op": "matmul", "attributes": {"output_dtype": "i32"}},
                "numeric_policy": {"compare": "exact_int"},
                "expected": {"instruction_classes": ["MVIN", "MVOUT"]},
                "required_oracle_tiers": ["L0", "L2"],
                "semantic": {"must_accelerate": True},
            }
        )
    )
    te = _synth_te(tmp_path)
    ws = tmp_path / "ws"
    VS.write_spec(te, ws)
    assert (ws / "verification_spec.md").is_file()
    payload = json.loads((ws / "verification_spec.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, _schema())
    assert payload["ops"]["matmul"]["checked_by"] == {
        "oracle_tiers": ["L0", "L2"],
        "datapath_coverage": True,
        "must_accelerate": True,
    }


def test_not_checked_names_the_obligations_nothing_enforces(tmp_path):
    """The spec's value is telling the agent what will be CHECKED. An op whose capsules declare no
    tier, no instruction classes and no `must_accelerate` is one where the stated acceptance policy,
    the stated coverage requirement and the whole point of the accelerator are all unenforced -- and
    before this the spec printed the same confident prose for it as for a fully checked op."""
    for name, declaration in (
        (
            "T0_checked",
            {
                "operation": {"op": "matmul", "attributes": {"output_dtype": "i32"}},
                "expected": {"instruction_classes": ["MVIN"]},
                "required_oracle_tiers": ["L2"],
                "semantic": {"must_accelerate": True},
            },
        ),
        ("T1_bare", {"operation": {"op": "gelu", "attributes": {"output_dtype": "i8"}}}),
    ):
        d = tmp_path / "isa" / name
        d.mkdir(parents=True)
        declaration.update({"name": name, "label": "public", "numeric_policy": {"compare": "exact_int"}})
        (d / "capsule.yaml").write_text(yaml.safe_dump(declaration))

    spec = VS.build_spec(_synth_te(tmp_path))
    unchecked = {(e["obligation"], e["scope"]) for e in spec["not_checked"]}
    # the fully-declared op raises NO finding
    assert not any(scope == "`matmul`" for _, scope in unchecked)
    # the bare op raises all three
    assert ("numeric acceptance", "`gelu`") in unchecked
    assert ("datapath coverage", "`gelu`") in unchecked
    assert ("work lands on the accelerator", "`gelu`") in unchecked
    # and the standing admission is always present
    assert any(e["obligation"] == "this document" for e in spec["not_checked"])
    md = VS.render_markdown(_synth_te(tmp_path))
    assert "## What is NOT checked" in md and "NOT CHECKED" in md


def test_the_grammar_ops_no_capsule_exercises_are_named_but_the_decomposed_ones_are_not(tmp_path):
    """`resident_pack` / `matmul` / `commit` / `evict` are the residency decomposition every contraction
    capsule emits, so reporting them as never exercised would be a false finding in the agent's face."""
    d = tmp_path / "isa" / "T0"
    d.mkdir(parents=True)
    (d / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "T0",
                "label": "public",
                "operation": {"op": "matmul", "attributes": {"output_dtype": "i32"}},
                "numeric_policy": {"compare": "exact_int"},
            }
        )
    )
    spec = VS.build_spec(_synth_te(tmp_path))
    grammar_entry = [e for e in spec["not_checked"] if e["obligation"].startswith("interface ops")]
    assert len(grammar_entry) == 1
    scope = grammar_entry[0]["scope"]
    for decomposed in ("resident_pack", "commit", "evict"):
        assert f"`{decomposed}`" not in scope, f"{decomposed} is emitted by every contraction capsule"
    assert "`attention_pv`" in scope


def test_a_real_targets_spec_validates_and_admits_something(tmp_path):
    import jsonschema
    import pytest

    from merlin.common.paths import merlin_dir
    from merlin.targetgen.target_experiment import load_target_experiment

    p = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    if not p.is_file():
        pytest.skip("descriptor absent")
    spec = VS.build_spec(load_target_experiment(p))
    if not spec["ops"]:
        pytest.skip("corpus not generated")
    jsonschema.validate(spec, _schema())
    assert len(spec["not_checked"]) > 1, "a real corpus with 20 ops leaves nothing unchecked?"
