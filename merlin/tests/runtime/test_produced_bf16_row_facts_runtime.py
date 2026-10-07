"""Independent all-word maximum proof through the normal source emitter."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import data_path, merlin_dir
from merlin.llvmlower.produced_bf16_row_facts import ProducedBF16RowFactsContract
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy


@pytest.mark.parametrize(
    "case_name,marker",
    [
        ("produced_bf16_maximum_equivalence.c", b"PASS checks=65763"),
        ("produced_bf16_packing_equivalence.c", b"CHECKS 205059"),
    ],
)
def test_all_finite_bf16_maxima_mixed_equality_shapes_and_refusals_ubsan(tmp_path, case_name, marker):
    compiler = shutil.which("clang-18")
    if not compiler:
        pytest.skip("Clang18 UBSan compiler required")
    plan = SourceAttentionFrontierPlan(
        2,
        5,
        4,
        12,
        5,
        4,
        0.125,
        -87.3365478515625,
        1.4426950216293335,
        (-0.079204238951206207, -0.22433836758136749, 0.30354261398315430, 0.00010703434963943437),
        8388608.0,
        1065353216.0,
        127.0,
        float.fromhex("0x1.5p-17"),
        -127,
        127,
    )
    source = emit_source_attention_frontier(
        plan,
        symbol="independent_provider",
        prepare_product_domain=True,
        prepare_required_norms=True,
        separable_source_radius=True,
        prepare_softmax_domain=True,
        prepare_probability_bins=True,
        prepare_encoded_rows=True,
        prepare_softmax_spans=True,
        prepare_probability_points=True,
        prepare_readonly_rhs=True,
        fuse_encoded_witness=True,
        source_roundoff_estimate=ApproximateSourceRoundoffPolicy(*([True] * 8)),
        produced_bf16_row_facts=ProducedBF16RowFactsContract(*([True] * 12)),
    )
    (tmp_path / "provider.c").write_text(source)
    case_root = (
        Path(os.environ["MERLIN_PRODUCED_BF16_FACTS_TEST_DATA"])
        if "MERLIN_PRODUCED_BF16_FACTS_TEST_DATA" in os.environ
        else merlin_dir() / "tests/data"
    )
    case = case_root / case_name
    binary = tmp_path / "maximum_ubsan"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-fno-builtin",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(tmp_path),
            "-I",
            str(data_path("runtime", "c")),
            str(case),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    run = subprocess.run([str(binary)], check=True, capture_output=True)
    assert marker in run.stdout
