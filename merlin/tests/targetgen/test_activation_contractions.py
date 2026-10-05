"""The contractions a dynamic int8 scheme cannot reach are quantized in that scheme's own form, on request.

WHY THIS EXISTS. ``int8_dyn_act_int8_weight`` replaces the weight of every ``nn.Linear`` and nothing else.
A whole-model capture under it therefore keeps two kinds of contraction in floating point: a matmul of
two ACTIVATIONS (attention's ``Q @ K^T`` and ``P @ V``, which have no module and no weight) and a
convolution (which ``quantize_`` does not transform). Measured on SmolVLA: the vision tower's attention
alone is 19.3 GMAC of f32 batched matmul, and on an int8-only unit every one of those contractions is
refused (``input_dtype``) and runs on the host.

``quant_activation_contractions`` on a model entry asks the capture to quantize them too, in the
scheme's own form (per-row dynamic first operand, per-output-column second operand, int8 x int8 into
int32, dequantized by both scales). What this pins:

* the request reaches the worker, and a capture made with it is never served for one made without it;
* such an entry is captured under its NAMED scheme -- a derived recipe replaces the scheme's form, so
  it must not silently take over;
* the capsule records the form and the per-site census the capture saw;
* the capture's leaf constants (buffers, lifted tensors) ship with the capsule and are found by the
  open-model builder without a second capture's files;
* the pure-python halves of the quantizer (which convolution is a matmul, the census) behave.

``torch`` is absent from this interpreter by design (the capture venv owns it). The numeric form is
checked against a numpy statement of it in the capture venv when that venv is present.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.targetgen import capsule_source as CSrc

_TARGETGEN = Path(CSrc.__file__).parent
_WORKER = _TARGETGEN / "_m2m_capture_worker.py"
_QUANTIZER = _TARGETGEN / "_activation_contractions.py"

# A W8A8 capture is admitted only with an i8 x i8 -> i32 contraction in its program; the fixture carries one.
_LINALG = (
    "builtin.module {\n"
    "  func.func @forward(%0: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
    "    %a = arith.constant dense<1> : tensor<2x2xi8>\n"
    "    %c = arith.constant dense<0> : tensor<2x2xi32>\n"
    "    %1 = linalg.matmul ins(%a, %a : tensor<2x2xi8>, tensor<2x2xi8>) outs(%c : tensor<2x2xi32>)"
    " -> tensor<2x2xi32>\n"
    "    return %0 : tensor<2x2xf32>\n  }\n}\n"
)


def _import_quantizer():
    sys.path.insert(0, str(_TARGETGEN))
    try:
        import _activation_contractions as AC  # noqa: PLC0415 -- a sibling imported by bare name
    finally:
        sys.path.remove(str(_TARGETGEN))
    return AC


# --- the quantizer's torch-free halves ------------------------------------------------------------


def test_the_quantizer_imports_without_torch_and_names_no_target():
    """It runs in the capture venv, but is read (and tested) from this one: torch stays a lazy import."""
    tree = ast.parse(_QUANTIZER.read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names} | {
        (n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)
    }
    assert not names & {"torch", "torchao", "merlin", "re"}, names
    assert _import_quantizer().MIN_REDUCTION == 16


class _Kernel:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)


@pytest.mark.parametrize(
    ("kwargs", "ok"),
    [
        ({"stride": 16}, True),  # a patch embedding: non-overlapping windows
        ({"stride": (16, 16), "padding": "valid"}, True),
        ({"stride": 8}, False),  # overlapping windows are not a re-layout
        ({"stride": 16, "padding": 1}, False),
        ({"stride": 16, "dilation": 2}, False),
        ({"stride": 16, "groups": 3}, False),
    ],
)
def test_only_a_non_overlapping_convolution_is_a_matmul(kwargs, ok):
    AC = _import_quantizer()
    verdict, why = AC.conv_as_matmul((None, _Kernel((768, 3, 16, 16))), kwargs)
    assert verdict is ok, why
    assert ok or why, "a refusal says why"


@pytest.mark.parametrize(
    ("kwargs", "ok"),
    [
        ({}, True),
        ({"is_causal": True, "scale": 0.125}, True),
        ({"dropout_p": 0.1}, False),  # a random mask is not two contractions
        ({"enable_gqa": True}, False),
    ],
)
def test_a_fused_attention_is_spelled_as_its_two_matmuls_only_when_it_is_exactly_that(kwargs, ok):
    """A vision tower routed through `scaled_dot_product_attention` hides both contractions from the
    matmul interception; measured on SmolVLA, all 24 of its vision-tower attention matmuls (19.3 GMAC)
    stayed float until the fused call was spelled out."""
    AC = _import_quantizer()
    t = _Kernel((1, 12, 1024, 64))
    verdict, why = AC.sdpa_as_matmuls((t, t, t), kwargs)
    assert verdict is ok, why


def test_the_census_counts_each_site_and_totals_by_verdict():
    AC = _import_quantizer()
    census = AC.Census()
    census.add("matmul", ((1, 12, 8, 64), (1, 12, 64, 8)), AC.QUANTIZED)
    census.add("matmul", ((1, 12, 8, 64), (1, 12, 64, 8)), AC.QUANTIZED)
    census.add("matmul", ((4, 1), (1, 8)), AC.REDUCTION_TOO_SMALL)
    doc = census.to_dict()
    assert doc["by_verdict"] == {AC.QUANTIZED: 2, AC.REDUCTION_TOO_SMALL: 1}
    assert {row["count"] for row in doc["sites"]} == {2, 1}
    json.dumps(doc)  # it is written into meta.json and capsule.yaml


# --- the worker ------------------------------------------------------------------------------------


def test_the_worker_declares_the_flag_and_refuses_it_under_a_recipe():
    """A derived recipe has its own granularity; mixing it with the scheme's form in one program is refused."""
    text = _WORKER.read_text(encoding="utf-8")
    assert '"--quantize-activation-contractions"' in text
    assert "recipe is not None or scheme_name !=" in text
    tree = ast.parse(text)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "_write_leaf_constants" for n in tree.body)


# --- the request reaches the capture, and is part of its identity ---------------------------------


def _stub(monkeypatch, seen: dict, *, extra: bool = True, meta_extra: dict | None = None):
    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "linalg.mlir").write_text(_LINALG, encoding="utf-8")
        (out / "inputs.json").write_text(json.dumps([[[1.0, 2.0], [3.0, 4.0]]]), encoding="utf-8")
        (out / "golden.json").write_text(json.dumps([[1.0, 2.0], [3.0, 4.0]]), encoding="utf-8")
        (out / "weights.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
        (out / "weights.safetensors.manifest.json").write_text(
            json.dumps({"0": {"kind": "input", "name": "x"}}), encoding="utf-8"
        )
        if extra:
            np.savez(out / "extra.npz", **{"buf::rope.inv_freq": np.arange(3, dtype=np.float32)})
        meta = {
            "ok": True,
            "opaque": 0,
            "opaque_detail": {},
            "scheme": "int8_dyn_act_int8_weight",
            "input_abi": [{"shape": [2, 2], "dtype": "f32"}],
            "output_abi": [{"shape": [2, 2], "dtype": "f32"}],
            "weights": str(out / "weights.safetensors"),
            "weights_manifest": str(out / "weights.safetensors.manifest.json"),
            "extra": str(out / "extra.npz") if extra else None,
        }
        if "--quantize-activation-contractions" in cmd:
            meta["activation_contractions"] = {
                "form": {"first_operand": "per row"},
                "sites": [],
                "by_verdict": {"quantized": 3},
            }
        meta.update(meta_extra or {})
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(CSrc.subprocess, "run", fake_run)
    monkeypatch.setattr(CSrc.PytorchRefSource, "_cache_slot", lambda *a, **k: None)


def _source(tmp_path):
    root = tmp_path / "model2MLIR"
    (root / "m2m").mkdir(parents=True)
    (root / "m2m" / "__init__.py").write_text("", encoding="utf-8")
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    loader = tmp_path / "loader.py"
    loader.write_text("def get_model_and_inputs(): ...\n", encoding="utf-8")
    return CSrc.PytorchRefSource(m2m_dir=root, python=python), loader


def test_the_request_reaches_the_worker_and_only_when_asked(tmp_path, monkeypatch):
    seen: dict = {}
    _stub(monkeypatch, seen)
    src, loader = _source(tmp_path)
    src.capture_loader(loader, "i8", workdir=tmp_path / "a", scheme="int8_dyn_act_int8_weight")
    assert "--quantize-activation-contractions" not in seen["cmd"]
    src.capture_loader(
        loader, "i8", workdir=tmp_path / "b", scheme="int8_dyn_act_int8_weight", activation_contractions=True
    )
    assert "--quantize-activation-contractions" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--scheme") + 1] == "int8_dyn_act_int8_weight"


def test_a_capture_with_the_request_is_a_different_cache_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    src, _ = _source(tmp_path)
    for owner in (
        "api.py",
        "ir/import_fx.py",
        "capture/torchao_pipeline.py",
        "capture/torchao_schemes.py",
        "capture/torch_export.py",
        "capture/torch_mlir_bridge.py",
    ):
        path = tmp_path / "model2MLIR" / "m2m" / owner
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {owner}\n", encoding="utf-8")
    plain = src._cache_slot("model", "i8", "src", "int8_dyn_act_int8_weight")
    asked = src._cache_slot("model", "i8", "src", "int8_dyn_act_int8_weight", capabilities=("activation_contractions",))
    if plain is None or asked is None:
        pytest.skip("artifact cache unavailable in this environment")
    assert plain != asked, "a capture with float attention must never be served for one that asked for int8"


def _binding():
    from merlin.targetgen import corpus_spec as CS

    return CS.CorpusBinding(
        target="t",
        tile_dim=16,
        operand_dtype="i8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
        atol=0.03125,
        rtol=0.02,
        classes_for=lambda **_: [],
    )


def _model_entry(**more):
    return {"name": "SY_model_x", "kind": "model", "cat": "model", "op": "model", "loader": "", **more}


def _write(tmp_path, monkeypatch, entry, *, extra=True):
    seen: dict = {}
    _stub(monkeypatch, seen, extra=extra)
    src, loader = _source(tmp_path)
    entry = {**entry, "loader": str(loader)}
    recipes: list = []

    def fake_recipe(target, dtype):
        recipes.append((target, dtype))
        return {"status": "derived", "schema": "quant_recipe_v1"}

    monkeypatch.setattr(CSrc, "derived_recipe", fake_recipe)
    monkeypatch.setattr(CSrc, "model_accelerator_demand", lambda *a, **k: (None, []))
    d = CSrc.write_model_capsule(entry, _binding(), tmp_path / "out", source=src)
    return d, seen, recipes


def test_an_entry_asking_for_them_is_captured_under_its_named_scheme(tmp_path, monkeypatch):
    """A derived recipe would replace the scheme's form; the entry named the scheme, so it wins here."""
    entry = _model_entry(quant_scheme="int8_dyn_act_int8_weight", quant_activation_contractions=True)
    d, seen, recipes = _write(tmp_path, monkeypatch, entry)
    assert recipes == [], "a recipe was derived for an entry that asked for the scheme's own form"
    assert "--recipe" not in seen["cmd"] and "--quantize-activation-contractions" in seen["cmd"]
    attrs = yaml.safe_load((d / "capsule.yaml").read_text())["operation"]["attributes"]
    assert attrs["quant_scheme"] == "int8_dyn_act_int8_weight"
    assert attrs["quant_activation_contractions"]["by_verdict"] == {"quantized": 3}


def test_an_entry_that_does_not_ask_keeps_the_recipe_precedence(tmp_path, monkeypatch):
    d, seen, recipes = _write(tmp_path, monkeypatch, _model_entry(quant_scheme="int8_dyn_act_int8_weight"))
    assert recipes, "the ordinary precedence (a derived recipe outranks a named scheme) must be unchanged"
    attrs = yaml.safe_load((d / "capsule.yaml").read_text())["operation"]["attributes"]
    assert "quant_activation_contractions" not in attrs


def test_the_capture_leaf_constants_ship_with_the_capsule(tmp_path, monkeypatch):
    d, _, _ = _write(tmp_path, monkeypatch, _model_entry(quant_scheme="int8_dyn_act_int8_weight"))
    attrs = yaml.safe_load((d / "capsule.yaml").read_text())["operation"]["attributes"]
    assert attrs["extra"] == "capsule.extra.npz"
    assert list(np.load(d / "capsule.extra.npz").files) == ["buf::rope.inv_freq"]


def test_a_capture_without_leaf_constants_names_none(tmp_path, monkeypatch):
    d, _, _ = _write(tmp_path, monkeypatch, _model_entry(quant_scheme="int8_dyn_act_int8_weight"), extra=False)
    attrs = yaml.safe_load((d / "capsule.yaml").read_text())["operation"]["attributes"]
    assert "extra" not in attrs and not (d / "capsule.extra.npz").exists()


# --- the open-model builder finds a capsule's own leaves -----------------------------------------


def _safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, offset, blobs = {}, 0, []
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(len(text).to_bytes(8, "little") + text + b"".join(blobs))


def test_the_open_model_builder_reads_the_capsule_declared_leaves(tmp_path):
    from merlin.perf import whole_model_open as WO

    d = tmp_path / "SY_model_x"
    d.mkdir()
    (d / "capsule.interface.mlir").write_text(
        textwrap.dedent(
            """\
            builtin.module {
              func.func @forward(%0: tensor<2xf32>, %1: tensor<3xf32>, %2: tensor<2xf32>) -> tensor<2xf32> {
                return %0 : tensor<2xf32>
              }
            }
            """
        ),
        encoding="utf-8",
    )
    _safetensors(d / "capsule.weights.safetensors", {"w": np.array([1.0, 2.0], np.float32)})
    manifest = {
        "0": {"kind": "param", "weight": "w"},
        "1": {"kind": "input", "name": "c_lifted_tensor_0"},
        "2": {"kind": "input", "name": "x"},
    }
    (d / "capsule.weights.safetensors.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(d / "capsule.extra.npz", c_lifted_tensor_0=np.array([7.0, 8.0, 9.0], np.float32))
    cap = {
        "name": "SY_model_x",
        "kind": "model",
        "source_role": "pytorch_model_slice",
        "label": "public",
        "numeric_policy": {"compare": "tolerance_float", "dtype": "f32"},
        "expected": {"instruction_classes": []},
        "required_oracle_tiers": ["L0"],
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "I0", "role": "input", "shape": [2], "dtype": "f32"}],
        "operation": {"op": "model", "attributes": {"extra": "capsule.extra.npz"}},
    }
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap), encoding="utf-8")
    golden = {"oracle_provenance": {"inputs": {"I0": {"shape": [2], "decoded": [3.0, 4.0]}}}, "outputs": {}}
    (d / "golden.yaml").write_text(yaml.safe_dump(golden), encoding="utf-8")
    capsule = SimpleNamespace(
        directory=d,
        interface=d / "capsule.interface.mlir",
        weights=d / "capsule.weights.safetensors",
        weights_manifest=d / "capsule.weights.safetensors.manifest.json",
    )
    args, sources = WO.forward_arguments(capsule)
    assert [list(a) for a in args] == [[1.0, 2.0], [7.0, 8.0, 9.0], [3.0, 4.0]]
    assert sources["extra"]["path"] == str(d / "capsule.extra.npz")


# --- the numeric form, in the capture venv --------------------------------------------------------

_FORM_CHECK = r"""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
import _activation_contractions as AC

f32 = np.float32

def q(x, half_range, eps, lo, hi):  # float32 throughout, x * (1/scale), round half to even
    s = np.maximum(np.abs(x).max(-1) / f32(half_range), f32(eps)).astype(f32)
    return np.clip(np.rint(x * (f32(1) / s)[..., None]), lo, hi).astype(np.int64), s

torch.manual_seed(0)
a = torch.randn(2, 3, 9, 40); b = torch.randn(2, 3, 40, 7)
y = AC.quantized_matmul(a, b).numpy()
# first operand: per row, symmetric, [-127, 127], eps 1e-5
qa, sa = q(a.numpy(), 127, 1e-5, -127, 127)
# second operand: per row of its transpose (= per output column), symmetric, [-128, 127], eps = f32 eps
qb, sb = q(np.swapaxes(b.numpy(), -1, -2), 127.5, np.finfo(f32).eps, -128, 127)
acc = qa @ np.swapaxes(qb, -1, -2)                      # exact integer accumulation
ref = (acc.astype(f32) * sa[..., None]) * sb[..., None, :]
assert np.allclose(y, ref, rtol=1e-6, atol=1e-6), float(np.abs(y - ref).max())

# the fused attention is exactly its two quantized matmuls around a masked softmax
q, k, v = torch.randn(1, 2, 20, 32), torch.randn(1, 2, 24, 32), torch.randn(1, 2, 24, 32)
mask = torch.rand(1, 1, 20, 24) > 0.2
mask[..., 0] = True
got = AC.quantized_sdpa((q, k, v), {"attn_mask": mask})
s = AC.quantized_matmul(q, k.transpose(-2, -1)) * (32 ** -0.5)
want = AC.quantized_matmul(torch.softmax(s.masked_fill(~mask, float("-inf")), -1), v)
assert torch.equal(got, want)
fp = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
assert float((got - fp).abs().max()) < 0.1, float((got - fp).abs().max())
print("ok", float(np.abs(y - a.numpy() @ b.numpy()).max()))
"""


def test_the_numeric_form_is_the_documented_one():
    """Per-row activation and per-column second operand, int32 accumulation, both scales on the way out."""
    python = Path(os.environ.get("MERLIN_M2M_PYTHON") or CSrc._m2m_python())
    if not python.is_file():
        pytest.skip(f"the capture venv ({python}) is not present on this host")
    probe = subprocess.run([str(python), "-c", "import torch, torchao"], capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        pytest.skip(f"the capture venv at {python} has no torch/torchao")
    run = subprocess.run([str(python), "-c", _FORM_CHECK, str(_TARGETGEN)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "ok" in run.stdout, run.stderr[-2000:]
