"""Softmax, GELU and layer norm computed in integer arithmetic, on request, at capture time.

WHY THIS EXISTS. A program whose contractions are all int8 still evaluates ``exp``, ``erf`` and
``rsqrt`` per element in floating point between them. Measured on SmolVLA's two-hart build (spike,
ranking only): softmax alone is 14.2 G of 35 G host cycles, GELU 4.0 G, layer norm 0.9 G.
``_integer_nonlinear`` replaces the three with the I-BERT integer algorithms, so a capture made under it
is a different numerical model -- graded against its own reference, its accuracy against the floating
point model measured separately.

What this pins:

* the module imports without torch, merlin or ``re`` and names no target;
* its integer-exp constants are ``passes_quant_int``'s, so the two integer exps in the repo cannot
  drift apart, and its GELU polynomial is the least-squares fit it states;
* in the capture venv (skipped when absent): the integer exp's error, the softmax's int8 numerators
  being EXACTLY the int8 operand the next contraction's dynamic row quantization produces, the masked
  positions at zero, GELU and layer norm within their stated error, the integer square root exact, and
  the mode reaching the softmax a fused attention computes inside the contraction mode.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.llvmlower import passes_quant_int as PQI
from merlin.targetgen import capsule_source as CSrc

_TARGETGEN = Path(CSrc.__file__).parent
_MODULE = _TARGETGEN / "_integer_nonlinear.py"


def _import_module():
    sys.path.insert(0, str(_TARGETGEN))
    try:
        import _integer_nonlinear as NL  # noqa: PLC0415 -- a sibling imported by bare name
    finally:
        sys.path.remove(str(_TARGETGEN))
    return NL


def test_the_module_imports_without_torch_and_names_no_target():
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names} | {
        (n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)
    }
    assert not names & {"torch", "torchao", "merlin", "re"}, names
    _import_module()


def test_its_integer_exp_is_the_lowering_passes_own():
    NL = _import_module()
    pairs = {
        "IEXP_A": "_IEXP_A",
        "IEXP_B": "_IEXP_B",
        "IEXP_C": "_IEXP_C",
        "IEXP_SH": "_IEXP_SH",
        "IEXP_K": "_IEXP_K",
        "IEXP_CLAMP": "_IEXP_CLAMP",
        "IEXP_S": "_IEXP_S",
        "IEXP_BQ": "_IEXP_BQ",
        "IEXP_AQ": "_IEXP_AQ",
        "IEXP_CQ": "_IEXP_CQ",
        "IEXP_QMAX": "_IEXP_QMAX",
    }
    assert {k: getattr(NL, k) for k in pairs} == {k: getattr(PQI, v) for k, v in pairs.items()}


def test_the_gelu_polynomial_is_the_fit_it_says_it_is():
    """The coefficients are re-derived from their stated fit, so they cannot be edited to pass a model."""
    import math

    import numpy as np

    NL = _import_module()
    u = np.linspace(0.0, NL.GELU_U, 40001)
    forms = {
        "none": 0.5 * (1 + np.vectorize(math.erf)(-u / math.sqrt(2))),
        "tanh": 0.5 * (1 + np.tanh(math.sqrt(2 / math.pi) * (-u - 0.044715 * u**3))),
    }
    for name, h in forms.items():
        fit = np.polynomial.polynomial.polyfit(u, h, NL.GELU_DEGREE, w=np.maximum(u, 0.05))
        assert np.allclose(fit, NL.GELU_COEFFS[name], rtol=0, atol=1e-8), name
        err = np.abs(u * (np.polynomial.polynomial.polyval(u, NL.GELU_COEFFS[name]) - h)).max()
        assert err < 6e-4, (name, err)


_NUMERIC_CHECK = r"""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
import _activation_contractions as AC
import _integer_nonlinear as NL

torch.manual_seed(0)
# the integer exp on its grid: within the polynomial's own floor (0.404% over |x| < 10)
q = torch.arange(-int(10 / NL.IEXP_S), 1)
got = NL.iexp_int(q).double() / (1 << NL.IEXP_SH)
want = torch.exp(q.double() * NL.IEXP_S)
assert float(((got - want).abs() / want).max()) < 5e-3

# softmax: the int8 numerators, the masked zeros, the row sums, the error
x = torch.randn(3, 5, 40) * 4
x[..., 7] = float("-inf")
y = NL.integer_softmax(x, -1)
ref = torch.softmax(x, -1)
assert float((y - ref).abs().max()) < 0.02, float((y - ref).abs().max())
assert torch.allclose(y.sum(-1), torch.ones(3, 5), atol=1e-5)
assert bool((y[..., 7] == 0).all())
# the next contraction's dynamic row quantization of P is the identity on its numerators
data, scale = AC._quantize(y, activation=True)
p = torch.round(y / y.amax(-1, keepdim=True) * 127)  # the row's largest term is exactly 127
assert torch.equal(data.to(torch.int64), p.to(torch.int64))
assert int(data.max()) == 127 and int(data.min()) >= 0

# GELU and layer norm: within their stated error
g = torch.cat([torch.randn(4, 3072) * 3, torch.linspace(-8, 8, 3072).reshape(1, -1)])
for form in ("none", "tanh"):
    err = float((NL.integer_gelu(g, form) - torch.nn.functional.gelu(g, approximate=form)).abs().max())
    assert err < 1.5e-3, (form, err)
h = torch.randn(4, 768) * 5 + 1
w, b = torch.randn(768), torch.randn(768)
ln = NL.integer_layer_norm(h, 768, w, b, 1e-6)
assert float((ln - torch.nn.functional.layer_norm(h, (768,), w, b, 1e-6)).abs().max()) < 0.01
# the integer square root is exact over the range an int16 row can produce
n = torch.randint(0, (1 << 31) - 1, (20000,), dtype=torch.int64)
n = torch.cat([n, torch.arange(0, 5000, dtype=torch.int64)])
r = NL.isqrt_int(n)
assert all(int(a) == max(1, math.isqrt(int(v))) for a, v in zip(r.tolist(), n.tolist()))

# nested outside the contraction mode, the softmax a fused attention spells out is integer too
nl, ac = NL.Census(), AC.Census()
qq, kk, vv = torch.randn(1, 2, 20, 32), torch.randn(1, 2, 24, 32), torch.randn(1, 2, 24, 32)
with NL._mode_class()(nl), AC._mode_class()(ac, convolutions=True):
    out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv)
    _ = torch.nn.functional.layer_norm(out, (32,))
    _ = torch.nn.functional.gelu(out)
rows = {(r["function"], r["verdict"]): r["count"] for r in nl.to_dict()["sites"]}
assert rows == {("softmax", "integer"): 1, ("layer_norm", "integer"): 1, ("gelu", "integer"): 1}, rows
print("ok")
"""


def test_the_integer_numerics_hold_in_the_capture_venv():
    python = Path(os.environ.get("MERLIN_M2M_PYTHON") or CSrc._m2m_python())
    if not python.is_file():
        pytest.skip(f"the capture venv ({python}) is not present on this host")
    probe = subprocess.run([str(python), "-c", "import torch, torchao"], capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        pytest.skip(f"the capture venv at {python} has no torch/torchao")
    run = subprocess.run(
        [str(python), "-c", _NUMERIC_CHECK, str(_TARGETGEN)], capture_output=True, text=True, timeout=600
    )
    assert run.returncode == 0 and "ok" in run.stdout, run.stderr[-3000:]


# ---- the integer helpers: exact integer ops, bit-identical to the float64 ones they replaced -------
#
# The shift and floor-division helpers were float64 round trips (cast, divide, floor, cast back)
# while the capture bridge left the integer ops opaque. The fixture keeps those versions verbatim and
# compares them with the integer ones, exactly, over the operand domains the callers produce
# (exhaustively where they are enumerable). The fast sections run by default; the large enumerations
# are marked slow.

_EXACTNESS = Path(__file__).resolve().parents[1] / "fixtures" / "integer_nonlinear_exactness.py"
_FAST = ("iexp", "softmax_numerator", "gelu_horner", "end_to_end")
_SLOW = ("shift_const", "shift_var", "floordiv", "gelu_rescale", "isqrt", "layer_norm_integer_steps")


def _capture_python() -> Path:
    python = Path(os.environ.get("MERLIN_M2M_PYTHON") or CSrc._m2m_python())
    if not python.is_file():
        pytest.skip(f"the capture venv ({python}) is not present on this host")
    probe = subprocess.run([str(python), "-c", "import torch"], capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        pytest.skip(f"the capture venv at {python} has no torch")
    return python


def _run_exactness(section: str) -> None:
    run = subprocess.run(
        [str(_capture_python()), str(_EXACTNESS), str(_TARGETGEN), section],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert run.returncode == 0 and f"ok {section}" in run.stdout, (run.stdout[-2000:], run.stderr[-3000:])


@pytest.mark.parametrize("section", _FAST)
def test_the_integer_helpers_equal_the_float64_ones(section):
    _run_exactness(section)


@pytest.mark.slow
@pytest.mark.parametrize("section", _SLOW)
def test_the_integer_helpers_equal_the_float64_ones_over_their_whole_domain(section):
    _run_exactness(section)


_CAPTURE_SHAPE = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
import _integer_nonlinear as NL
from m2m.coverage import opaque_report
from m2m.ir.decompositions import DECOMPOSITION_TABLE
import m2m

missing = [n for n in NL.REQUIRED_DECOMPOSITIONS if n not in DECOMPOSITION_TABLE]
if missing:
    print("skip", missing)
    raise SystemExit(0)

class M(torch.nn.Module):
    def forward(self, x, y, z):
        return NL.integer_softmax(x, -1), NL.integer_gelu(y), NL.integer_layer_norm(z, 64, None, None, 1e-6)

r = m2m.convert(M().eval(), (torch.randn(2, 16, 32), torch.randn(4, 96), torch.randn(8, 64)), backend="fx_importer")
assert sum(opaque_report(r.mlir_text).values()) == 0
names = [o.name for o in r.module.walk()]
assert "arith.shrsi" in names and "arith.floordivsi" in names
# float64 is left only where _dyadic turns a row's scale into a mantissa and exponent: per ROW
for op in r.module.walk():
    if op.name == "linalg.generic":
        types = [v.type for v in (*op.operands, *op.results)]
        if any("f64" in str(t) for t in types):
            assert all(t.get_shape()[-1] == 1 for t in types), str(types)
assert names.count("math.floor") <= 1, names.count("math.floor")
open(sys.argv[2], "w").write(r.mlir_text)
print("ok")
"""


def _generic_bodies(text: str):
    """``(result type, body op names)`` of each linalg.generic in printed MLIR, read line by line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "linalg.generic" not in line or "outs(" not in line:
            continue
        ops, j = [], i + 1
        while j < len(lines) and not lines[j].strip().startswith("} ->"):
            ops += [w for w in lines[j].split() if w.startswith(("arith.", "math."))]
            j += 1
        yield (lines[j].strip()[len("} ->") :].strip() if j < len(lines) else ""), ops


def test_the_capture_has_no_per_element_float64_round_trip(tmp_path):
    """Captured through the selected bridge, the three functions keep their integer arithmetic integer:
    shifts and floor divisions are ``arith.shrsi``/``arith.floordivsi``, and float64 appears only per row."""
    python = _capture_python()
    m2m_dir = CSrc._m2m_dir()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(m2m_dir), os.environ.get("PYTHONPATH", "")) if p),
    }
    captured = tmp_path / "integer_nonlinear.mlir"
    run = subprocess.run(
        [str(python), "-c", _CAPTURE_SHAPE, str(_TARGETGEN), str(captured)],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    if run.stdout.startswith("skip"):
        pytest.skip(f"the selected Model2MLIR at {m2m_dir} does not decompose {run.stdout.split(' ', 1)[1].strip()}")
    assert "ok" in run.stdout, run.stdout[-2000:]

    # After the lowering's own elementwise fusion, the per-row work (the square root's Newton steps, the
    # dyadic rescale's log/pow/floor) stays in per-row ops: no op over a whole row repeats it per element.
    from merlin.llvmlower import pipeline as PL

    stages = PL._UPSTREAM_PASSES[: PL._UPSTREAM_PASSES.index("func.func(linalg-fuse-elementwise-ops)") + 1]
    fused = PL.apply_passes(captured.read_text(encoding="utf-8"), ",".join(stages))
    per_element = [ops for result, ops in _generic_bodies(fused) if not result.split("x")[-2:-1] == ["1"]]
    assert per_element
    for ops in per_element:
        assert not {"math.log", "math.powf", "math.floor"} & set(ops), ops
        assert ops.count("arith.floordivsi") <= 1, ops
