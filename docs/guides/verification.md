---
title: Verify a compiler transformation
kind: guide
status: current
owner: verification
last_verified: 2026-09-27
related: [phase0_specification, model_lowering, simulator_selection]
code_refs:
  - src/merlin/verify/receipts.py
  - src/merlin/verify/refine.py
  - src/merlin/verify/linalg_semantics.py
  - src/merlin/verify/smt_semantics.py
  - src/merlin/verify/cb_semantics.py
  - src/merlin/verify/model_coverage.py
---

# Verify a compiler transformation

Merlin can check two concrete transformation boundaries: a `linalg` source module against its emitted
`interface` module, and an `interface` module against its emitted command buffer. The verifier encodes
both sides over the same symbolic inputs, asks Z3 whether any output can differ, and records the
answer. `unsat` verifies equivalence for **all input bit patterns at that concrete shape** under the
encoded semantics. `sat` supplies a counterexample. `unknown`, unsupported syntax, and missing tools
do not verify anything.

## Produce and replay a receipt

For saved in-tree `interface` MLIR and its emitted command buffer, the installed
entry point writes a new receipt directly:

```sh
merlin-verify compile-receipt --interface /generated/interface.mlir \
  --command-buffer /generated/command_buffer.json \
  --translator /selected/mlir-translate \
  --translator-sha256 "$MERLIN_VERIFY_TRANSLATOR_SHA256" \
  --output /generated/transform-receipt.json
```

Exit 0 means verified, 1 refuted, and 2 unsupported, unavailable or unknown.
The command does not accept a capsule's custom `merlin_iface` assembly file.
`merlin-verify capture-coverage /generated/model.mlir` separately prints a
conservative textual inventory of the current SMT source subset; eligibility
there is not a proof.

Install Merlin with its `verify` extra, provide a compatible LLVM `mlir-translate`, and select that
binary explicitly. The toolchain is an external input. Set `MERLIN_VERIFY_TRANSLATOR` to its absolute
path and `MERLIN_VERIFY_TRANSLATOR_SHA256` to the approved SHA-256 digest. For example:

```python
import json
import os
from pathlib import Path

from merlin.verify.receipts import TransformReceipt, qualify_receipt, verify_transformation

translator = os.environ["MERLIN_VERIFY_TRANSLATOR"]
expected_sha = os.environ["MERLIN_VERIFY_TRANSLATOR_SHA256"]

# `source_module` and `interface_module` are the actual before/after xDSL modules
# retained from the compiler invocation being checked.
receipt = verify_transformation(
    "linalg_to_interface",
    source_module,
    interface_module,
    translator=translator,
    expected_translator_sha256=expected_sha,
    timeout_ms=60_000,
)
path = Path("out/artifacts/verification/transform-receipt.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")

loaded = TransformReceipt.from_dict(json.loads(path.read_text(encoding="utf-8")))
assert qualify_receipt(loaded, source_module, interface_module, translator=translator)
```

For the downstream boundary, use `"interface_to_command_buffer"` with the actual interface module
and emitted command-buffer dictionary. Replay runs the semantic check again and requires the same
source, target, query, verifier implementation, translator binary, solver version, and verdict. An
IR parser or syntax verifier can establish well-formedness but cannot produce a semantic receipt.
Generated receipts should be kept with run artifacts, not substituted for the input artifacts.

Each receipt records SHA-256 identities of the generic xDSL IR text and canonical command-buffer
JSON, both sides' typed signatures, the verifier's source digest, translator path/version/hash,
xDSL and Z3 versions, timeout, assumptions, solver-query digest, outcome, and any counterexample.
The receipt's `status` is one of
`verified`, `refuted`, `unknown`, `unsupported`, `unavailable`, or `error`. Only `verified` passes the
qualification gate. A counterexample is a diagnostic input and should also be replayed through the
independent numerical oracle or simulator before attributing the defect to a particular pass.

## What the proof covers

The currently encoded source subset is one function with rank-2, positive, concrete-shape integer
tensors (`i8`, `i16`, `i32`, `i64` where width constraints permit), `arith.constant`, `tensor.empty`
used as the destination of a defining `linalg.fill`, `linalg.fill` with an integer constant,
`linalg.matmul`/`linalg.quantized_matmul`, and `func.return`. Quantized zero points must be resolvable
integer constants. The interface subset covers resident packing as value preservation, matmul,
commit without epilogue stages, eviction, and return. Its return values must be exactly its commits
in order. The command-buffer encoder supports its declared opcode subset and abstains on unknown
opcodes or numerical behavior. Shapes, widths, leaf binding, output count, and supported operations
are checked before an `unsat` result can be reported.

The source interpreter refuses a contraction whose `outs` operand is bare `tensor.empty`. [MLIR says
its contents are unspecified](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-tensoremptyop),
while [a linalg contraction][mlir-matmul] reads its output accumulator.
An explicit integer `linalg.fill` defines the init; zero fill makes the common `A @ B` case eligible
for an unconditional source-to-target value proof. The older repeated-RHS example currently uses a
bare `tensor.empty` init and therefore receives `unsupported`, even when its emitted target happens
to agree with a zero-init interpretation. Integer arithmetic uses signed bitvectors and modular
accumulation at the declared accumulator width. A verified receipt establishes equality in this
modeled value domain,
not physical packing, memory behavior, actual RTL execution, performance, or independent correctness
of the semantics encoder. The encoder and command-buffer meaning require separate review and
conformance checks against an independent implementation.

[mlir-matmul]: https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop

Floating-point reassociation, dynamic shapes, arbitrary PyTorch operators, symbolic zero points,
nonempty epilogues, multi-function modules, host effects, and unencoded target opcodes currently
abstain. These need explicit semantics and a sound correspondence relation before they can be called
formally verified. A finite shape sweep gives one theorem per checked shape, not one theorem for all
shapes. The proof does not by itself establish that Phase 1 handles every operator or model named in
Phase 0; that broader claim needs coverage accounting and executable model-level evidence.

Adding BF16 requires an exact floating-point semantics for each operation and conversion, including
rounding mode, reduction order, signed zero, NaNs, infinities, and subnormal policy. An IEEE BF16
format can be represented with an 8-bit exponent and 8-bit significand in an SMT floating-point
theory, but the target's actual flush/saturation behavior must be modeled separately. FP8 variants
such as finite-only E4M3 need a format-specific bitvector or circuit semantics. A tolerance claim
requires a specified relational error bound and a sound bound proof, including accumulation and
nonlinear approximations. Sampled tolerance tests are useful numerical evidence; they are not that
proof. Until those semantics and their independent conformance checks exist, the receipt returns
`unsupported` for these programs.

To inventory the current captured models against the source-side subset, run:

```sh
python -m merlin.verify.model_coverage \
  merlin/contract/capsules/model/SY_model_tiny_llama/capsule.interface.mlir \
  merlin/contract/capsules/model/SY_model_smolvla/capsule.interface.mlir \
  merlin/contract/capsules/model/SY_model_resnet50/capsule.interface.mlir
```

The JSON includes each capture's SHA-256, operation counts, observed tensor ranks/dtypes, and
abstention reasons. It inventories printed MLIR; it does not parse or prove the model. The final
qualification still requires a real transformation receipt for each eligible compiler boundary.

## Connect this to Phase 0 and Phase 2

Phase 0 should freeze the software-visible operation signatures and numerical rules, workload
provenance, legal shape/layout/dtype domain, host placement, and test selection criteria. For every
admitted operation family, record which compiler boundary and semantic assumptions its proof uses.
An unsupported operation remains an explicit coverage gap; an `unsat` result on a different shape or
dtype cannot fill it. Keep a holdout set of model/operator signatures outside the corpus used to
write the compiler and encoder. Phase 1 evidence should pair each lowered program with its source,
target, receipt, and independent numerical differential test. Phase 2 consumes the frozen Phase 0
oracle definitions and holdouts, then adds actual target execution and simulator/RTL checks. This
separates a proved local transformation from the distinct questions of workload coverage, oracle
validity, and target conformance.
