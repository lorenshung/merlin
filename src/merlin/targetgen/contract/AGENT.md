# AGENT.md — merlin/python/merlin/targetgen/contract

## Purpose

Experiment-ABI contract layer.

## Modules

- `compile.py` — Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.
- `build_recipe.py` — Pure compile/link recipe; the runtime backend re-exports this same class.
- `build_service.py` — Typed host-only build service and isolated pure target-package loader; no backend discovery, execution or reference imports.
- `readback_policy.py` — Explicit invocation-only full-value output transport and selected build-byte receipt; never source this choice from a candidate command buffer or ambient environment.
- `interface_emit.py` — ``merlin_iface`` interface-grammar: emit a Merlin command buffer as contract text, and
- `schemas.py` — Fail-closed JSON-Schema validation against the ``merlin/contract/schemas/`` bundle.
- `toolchain.py` — MLIR toolchain resolution for the experiment ABI (env-overridable).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

`materialize.py` owns public copying, explicit-ceiling cohort publication, admission
validation, coverage and leases. It never selects evaluator adapters. Use core
`materialize_public_cohort` with an explicit ceiling; default evaluated selection is
`merlin_experiments.corpus.admission.public_capsules_for`, not a lazy core export.

`build_recipe.named_object_paths` shares deterministic object naming across
contract and layer builds. Equal basenames from caller and provider sources must
never overwrite one object; imported objects are reserved and link order is
retained. Unique basenames keep their original object names and commands.

`HarnessBuildRecipe.header_dependencies` declares direct harness headers for
ELF-cache identity. Exact bytes and first-match include-root resolution are
checked; a missing, unreadable or shadowed declaration disables caching. This
does not establish a complete transitive compiler-header closure.

The opt-in `ReadbackPolicy` changes only the trusted harness readback format.
Default calls retain their old build/cache path. A selected full-value build is
cache-free, requires a renderer that explicitly accepts the policy, and records
the unchanged command buffer, selected codec/recipe/source, generated harness,
object and ELF bytes. Recheck those pins after execution before any numerical
result. This is not complete toolchain closure or a numerical-support grant.
The binary alternative is separately selected and stages its own length-aware
packer plus the existing range helper; a raw byte console is archived before
strict parsing. B64 and absent-policy behavior remain unchanged.

The opt-in coherent-memory policy stages no serial codec and uses a distinct
v2 build receipt. Execution requires an explicitly supplied trusted memory
reader, selected-engine revalidator and provider request ABI: bound physical output storage is admitted
before launch; normal exit, exactly one DONE, no serial substitutes, complete
logical values and unchanged build bytes are required afterward. Optional
evaluators own the memory decoder; core never imports a grader to select one.
An output-readback admission does not prove numerical or compiler correctness.

The separately selected packet-memory policy stages all three generic codec
headers and uses a distinct v3 build receipt. Its trusted reader checks the
ELF-bound bounded arena and complete logical output frames, not output padding.
Existing raw-memory, binary, B64 and absent-policy build identities stay unchanged.
