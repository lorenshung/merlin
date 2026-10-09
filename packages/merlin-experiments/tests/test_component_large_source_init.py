"""Large source initialization parses with bounded process memory and no data.

This covers ordinary source construction/parsing and small original numerics,
not candidate compile-through-link, index/resource legality or device execution.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import test_component_execution_budget as fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage, generation

from merlin.targetgen import component_program, golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

independent = generation_fixtures.independent


def test_large_original_source_parses_in_a_fixed_memory_and_cpu_budget():
    source_origin = Path(component_program.__file__).resolve()
    library_root = source_origin.parents[2]
    script = r"""
import json,resource,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(10,10))
from merlin.targetgen.component_program import build
from merlin.targetgen import component_program
assert Path(component_program.__file__).resolve()==Path(sys.argv[2])
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.corpus_spec import CorpusBinding
binding=CorpusBinding(target="independent",tile_dim=1,operand_dtype="int8",
    accum_dtype="i32",integer=True,tiers=[],compare="exact_int")
m,k,n=10**9,3,5
entry={"name":"Large", "kind":"model_slice", "source_role":"derived_sweep",
    "source_reference":"independent source extent", "program":{
        "inputs":[{"name":"A","role":"input","shape":[m,k],"dtype":"operand"},
                  {"name":"W","role":"weight","shape":[k,n],"dtype":"operand"}],
        "nodes":[{"name":"P","op":"matmul","inputs":["A","W"]}],
        "outputs":[{"name":"Y","value":"P"}]}}
cap,source=build(entry,binding)
assert len(source)<10000
parsed=parse_linalg_mlir(source)
assert parsed["args"]==[{"index":0,"shape":[m,k],"dtype":"i8"},
                         {"index":1,"shape":[k,n],"dtype":"i8"}]
assert parsed["results"]==[{"shape":[m,n],"dtype":"i32"}]
assert len(parsed["ops"])==1
assert parsed["ops"][0]["body_ops"]==["arith.extsi","arith.extsi","arith.muli","arith.addi"]
assert parsed["ops"][0]["outs"][0]["source"]==("init","fill")
print(json.dumps({"source_bytes":len(source),"shape":[m,k,n],
                  "memory_limit_bytes":512*1024*1024,"cpu_limit_seconds":10}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(library_root), str(source_origin)],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["source_bytes"] < 10000 and observed["shape"] == [10**9, 3, 5]


def test_bounded_normal_generation_keeps_original_complete_integer_outputs(independent):
    row = fixtures.contraction(independent)
    software = yaml.safe_load(independent["software_spec"].read_bytes())
    software["numerical_semantics"]["overflow"] = "bounded_exact"
    generation_fixtures.write(independent["software_spec"], software)
    fixtures.select(independent, [row])
    generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    member = report["obligations"][0]["members"][0]
    directory = independent["output_root"] / member["member"]
    capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    source = (directory / "capsule.interface.mlir").read_text()
    assert "%P_zero = arith.constant 0 : i32" in source
    assert "linalg.fill ins(%P_zero : i32) outs(%P_empty : tensor<2x2xi32>)" in source
    assert "dense<" not in source
    leaves = materialize_capsule_leaves(capsule)
    expected = [
        [sum(leaves["A"].data[i * 3 + p] * leaves["W"].data[p * 2 + j] for p in range(3)) for j in range(2)]
        for i in range(2)
    ]
    assert golden_store.load_golden(directory)["outputs"] == {"Y": expected}
    assert capsule["integer_partial_sum_bound"]["status"] == "proven_safe"
