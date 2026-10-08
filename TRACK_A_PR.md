# Draft PR: Track A — fix silent precision bugs in Merlin's TorchAO FP8 path

**Branch:** `track-a` → base `main` (`4a7cb1821`, ucb-bar/merlin = lorenshung/merlin)
**Status:** draft for review — not opened. Decide keep / drop / split.

## Goal

Track A of `MERLIN_PLAN.md`: take a BF16 model (target: Pi0 and SmolVLA) to FP8 for the Atlas accelerator by
deriving an operation-scoped quantization policy from the target's hardware/software contract, realising it with
public TorchAO APIs, and preserving scales, layout and numeric types through model2MLIR capture and lowering.

This PR covers only the first piece of that: making Merlin's existing recipe → TorchAO → model2MLIR front end do
what a recipe says, or refuse. It does **not** add Atlas support, an FP8 compiler path, or results on Pi0/SmolVLA.

## What changed

Four fixes in two files, each with the failure it removes. All four failed silently before: the run succeeded and
recorded a result that did not match the request.

| Commit | File | Before | After |
|---|---|---|---|
| `fix(targetgen): pass recipe float8 dtypes and granularities to TorchAO` | `_recipe_quantizer.py` | An `fp8_e5m2` recipe produced `float8_e4m3fn` tensors (TorchAO defaults). Activations used the weight's granularity. | Both dtypes and an `(activation, weight)` granularity pair are passed explicitly; unsupported combinations raise `RecipeError` naming the fields. |
| `fix(targetgen): give static float8 specs the format's finite range` | `_recipe_quantizer.py` | A derived FP8 recipe (no `quant_min/max`) let TorchAO fall back to `(0, 15)`: every negative value clamped to 0. | Float8 specs without a range use `torch.finfo` (±448 for E4M3). |
| `fix(targetgen): count only layers TorchAO actually quantized` | `_recipe_quantizer.py`, `_m2m_capture_worker.py` | TorchAO silently skips Linears whose dims aren't multiples of 16; the capture still reported them quantized and labelled an FP32 graph as FP8. | Only layers whose weight was actually replaced are counted; skipped ones are recorded with a reason; a run that quantized nothing is refused. The dynamic FP8 scheme is now recorded in capture metadata. |
| `fix(targetgen): refuse capture dtypes with no known scheme` | `_m2m_capture_worker.py` | `--dtype fp8_e5m2` (not in the scheme table) captured the FP32 model with no error. | Unknown dtype tokens are refused at argument parsing unless `--recipe`/`--scheme`/`--already-quantized` decides. |

Plus 12 tests (`merlin/tests/targetgen/test_recipe_quantizer_dynamic.py`, `test_capture_dtype_is_known.py`) and
two working logs (`TRACK_A_JOURNAL.md`, `TRACK_A_BUGS_AND_FIXES.md`).

## How it was verified (on base `4a7cb1821`)

- **New tests:** 12/12 pass on this branch; against unmodified upstream `src/` (isolated pytest run that asserts
  which source tree it imported) 9 of the 12 fail.
- **Existing tests:** the same 13 targetgen test files run against pristine upstream and against this branch, with
  the model2MLIR capture interpreter enabled: upstream `3 failed, 68 passed`, branch `3 failed, 80 passed`. The 3
  failures are identical on both (`test_quantization_spec.py`, pre-existing). `ruff` clean.
- **End to end through model2MLIR** (`d671200`, CPU torch 2.10 / torchao 0.17), small BF16 MLP, same recipe:

  | | upstream | this branch |
  |---|---|---|
  | Static FP8 range in captured MLIR | `quant_min = 0, quant_max = 15` | `quant_min = -448, quant_max = 448` |
  | Agreement with unquantized model | cosine 0.45, argmax 0 % | cosine 0.997, argmax 100 % |
  | Dynamic FP8, 8-wide layers (all skipped by TorchAO) | exits 0, labelled FP8 | refused, names the layers |

  The dynamic-route checks needed TorchAO's GPU check bypassed (this machine's GPU is SM 7.5; TorchAO requires
  ≥ 8.9), so treat those as diagnostic.

## Not in this PR / not done

- **Atlas-specific numerics.** TorchAO picks fp32 `amax/448` scales; Atlas's scale register holds powers of two
  only. TorchAO has no public power-of-two option for inference. Unresolved.
- **The software spec.** Atlas's spec leaves model scale encoding and block size `unknown`, so Merlin cannot derive
  an FP8 recipe for Atlas; the recipes used here were hand-written. Open questions for the spec owners are listed
  in `TRACK_A_BUGS_AND_FIXES.md` (B9–B17, B20, B22).
- **Compiler.** Nothing lowers the captured FP8 MLIR to Atlas (or any target). Captured MLIR stores FP8 as f32 with
  quantize/dequantize ops around an f32 matmul.
- **Real models.** Only a 4-layer MLP was captured in FP8. SmolVLA was captured in BF16 only; Pi0 was not run.
- **`--dtype bf16` + `--recipe`** still fails on both routes (B36); BF16 → FP8 works only when the model is BF16
  at source.
- **Other bugs found but not fixed here** (fail-open admission, per-channel axis after transpose, model2MLIR's
  dynamic FP8 cast lowered as identity, etc.) are in the bug log with status.

## Side investigations (logs only, not code)

Recorded in the journal; their artifacts are local to `garden` under `out/artifacts/probes/` (gitignored):
Atlas MXU RTL vs the Atlas team's C model under VCS (bit-exact on the tested vectors), a numerics study on
synthetic Pi0/SmolVLA-shaped layers, an atlas-mlir build/test run, and a hand-written FP8 linear run on npu-model
(a performance model — not numerical evidence).

## Review notes

- **Known CI failure:** `build_tools/scripts/check_structure.py` fails "module size" —
  `_m2m_capture_worker.py` is 1514 lines against a 1500 limit. Upstream is at exactly 1500, so any change to the
  worker trips it. Options: split the worker first (its owner's call), move the dtype check and scheme recording
  into a helper module, or drop the two worker commits and keep only the `_recipe_quantizer.py` fixes. All other
  structure checks, `check_doc_paths` and `check_artifact_layout` pass.
- The two `.md` logs at the repo root are working notes, not durable docs; drop them or move them before merging
  if they don't belong in the tree.
- Commits are independent and can be cherry-picked separately.
- To reproduce the end-to-end checks you need a model2MLIR checkout + capture venv (`MERLIN_M2M_DIR`,
  `MERLIN_M2M_PYTHON`); `pyyaml` must be installed in that venv.
