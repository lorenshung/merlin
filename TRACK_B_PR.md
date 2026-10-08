# Draft PR: Track B — per-item coverage inventory + M0 audit

**Branch:** `track-b` · **Base:** `upstream/main` @ `4a7cb1821` (ucb-bar/merlin = lorenshung/merlin main)
**Status:** draft for review — not pushed, not opened.

## Goal

Track B of `MERLIN_PLAN.md` asks whether a small, representative test suite can produce a correct
compiler and improve full-model performance with less iteration cost. Before writing experiment code,
this branch does two things:

1. **M0 orientation** — find out what already exists (evidence, coverage machinery, infrastructure) and
   what is missing, so B1 ("freeze inputs and claims") starts from facts.
2. **First B2 building block** — one table, per target, that says for every thing the compiler must
   cover: where it must run, which tests cover it, and whether that was observed to pass in the right place.

## What changed (code)

| File | Change |
| --- | --- |
| `src/merlin/targetgen/coverage_inventory.py` | **New.** `build_inventory(target, spec_doc, contract, witnesses, *, results=None, workloads=None)` — a pure join producing one `CoverageRow` per required item. Thin readers: `collect_witnesses`, `results_from_dir`, `inventory_for_target`. |
| `src/merlin/targetgen/contract/materialize.py` | Extract `capsule_cell_rows` out of `cert_capsule_cover` (behavior-preserving) so the cert cover and the inventory share one definition of "capsule X exercises cell Y". |
| `src/merlin/targetgen/claim_models.py` | Add `mentioned_claim_model(name)`: whole-token match at any offset (no regex), because capsule names carry the model mid-name (`<prefix>_<model>_<suffix>`). |
| `merlin/tests/targetgen/test_coverage_inventory.py` | **New.** 22 tests on gemmini and atlas tracked data. |
| `TRACK_B_JOURNAL.md`, `TRACK_B_PR.md` | Working notes (separate commit; drop before merge — reports belong under `out/artifacts/`, not the repo root). |

### How a row is built

Each row joins existing sources rather than adding new logic:

- **Required item** — from the target's tracked conformance spec (`merlin/contract/capsules/conformance/<target>.yaml`):
  `(family, dtype, tile alignment)` cells, host lanes, host-only items, compositions, epilogues, shape geometry.
- **Required placement** (accelerator / host / `UNKNOWN`) — `eligibility.is_eligible` over the target's contract.
- **Composition class** — `boundary` (`A`, `A->H->A`, `H->A->H`, routing, …).
- **Witnesses** — the capsules that exercise the item, using the same readers the conformance gate uses.
- **Observed placement + verdict** — from `placement_census` in supplied capsule grading results; `not_run` otherwise.

**Status is fail-closed:** `covered` only if a witness *passed* **and** its observed placement matches the
required one. `UNKNOWN` placement, `not_run`, a pass with no measured placement, spec/contract disagreement,
or held-out-only evidence → `unknown`. Silent host fallback, a failure, or no witness → `missing`.
Supplying a held-out claim model (`merlin/contract/claim_models.yaml`) as an input raises `HoldoutInputError`.

### What it reports today (no grading results exist locally)

| Target | Rows | Covered | Missing | Unknown |
| --- | --- | --- | --- | --- |
| gemmini | 44 | 0 | 3 | 41 |
| atlas | 46 | 0 | 7 | 39 |

"0 covered" is expected: nothing has been graded on this machine yet. Notable: the tracked atlas
conformance spec requires bf16/elementwise cells on the accelerator that the atlas contract does not
declare (it declares only fp8 contraction) — the spec looks stale. Flagged, not changed.

## Verification (on the rebased branch)

- `pytest merlin/tests/targetgen/test_coverage_inventory.py` — 22 passed.
- `check_no_target_name`, `check_no_regex`, `check_structure`, `check_artifact_layout`, `check_docs`,
  `check_doc_paths` — all exit 0. `ruff check` clean.
- Neighbouring tests (cover, claim-set, conformance, promotion wiring) were run on the old base
  (`183cbb09f`): 5 pre-existing failures that do not reach the changed code (missing recapture store;
  tracked `atlas.yaml` cites a held-out model). **Not re-run on the new base.**

## M0 audit findings (not code — context for B1)

1. **The "token efficiency vs Claude Code" claim is not supported by any like-for-like pair.**
   Runs live outside the repo (`/scratch/agustin/projects/oscar-merlin/out/runs/gemmini/capsule-bench/`).
   - The headline 6.6M vs 92.9M compares gpt-5.6/Codex (Merlin arm) with Opus 4.8/Claude Code (baseline):
     model, driver, arm and repo sha all differ.
   - Same model + same driver (Opus 4.8, Claude Code, `gemrecreate1`): Merlin arms used 33–45% fewer tokens
     but qualified lower (19/20 at L1 vs 20/20 at L3); hidden never graded; no budget recorded.
   - The only fully matched pairs (g3arm, gpt-5.6/Codex, hidden graded): Merlin used *more* tokens
     (70.4M vs ≥59.3M) and qualified lower.
   - An sha256-pinned index of these runs was written locally (gitignored, not in this branch):
     `out/artifacts/audits/gemmini/v1/audits_gemmini_v1_20261002T085651Z_183cbb0/` in the worktree.
2. **Held-out models have leaked into development.** `merlin/contract/capsules/_model_layers/G_*` are
   ResNet-50-derived dev capsules; perf-bench kernels `M00–M02` (SmolVLA) and `M05–M07` (TinyLlama);
   the ResNet-50 FireSim results (q534–q536) were used during Phase 2 development. No dev/holdout split
   file exists.
3. **Coverage machinery mostly exists; the join did not** (this PR). Jack's ATen coverage
   (`/scratch/jack/PyTorchFullCoverage/`) is scalar-Spike only, not tied to accelerator placement, and
   uses `re` plus a hardcoded gemmini profile, so it can't be dropped in as-is.
4. **Infrastructure is installed but not wired.** VCS, Verilator, Spike, firtool, RISC-V GCC and bwrap are
   present; no `.env`, so every adapter reports unavailable. Only gemmini pins have matching checkouts
   (atlas, saturn, radiance, gsim do not). No PowerSpec anywhere in the repo. No FireSim runner installed.
   bwrap cannot start nested inside an agent shell.

## Not done / out of scope

- No B1 scope or holdout-split file (needs decisions below).
- No test reduction (B3), capsule selection (B4), or optimization runs (B5/B6).
- No `.env`, no simulator runs, no pin changes.
- No CLI or `out/` product for the inventory (would need a reviewed `storage.yaml` entry).

## Decisions needed before continuing

1. **Scope (B1):** which targets the claims name (gemmini is the only one runnable end-to-end here), the
   baseline model + driver for a matched token comparison, budgets, number of seeds.
2. **Holdouts:** keep ResNet-50 held out (then drop `_model_layers/G_*` and `M0x` kernels from selection and
   discard the q534–q536 numbers as evidence) or name a fresh held-out model.
3. **`.env` + canonical chipyard tree.**
4. **Pins:** check out atlas/saturn/radiance/gsim at their pinned commits, or re-pin with a rationale.
5. **Jack's coverage work:** port it here, or wait for him to upstream it.

## Keep or drop

- **Keep the code:** drop the `docs(track-b)` commit (journal + this file) and take `feat(targetgen)` alone.
- **Drop everything:** `git branch -D track-b` and `git worktree remove /scratch2/loren/merlin-track-b`.
