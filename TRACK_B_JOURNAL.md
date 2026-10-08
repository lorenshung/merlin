# Track B Journal — compiler coverage and representative capsules

Working log for MERLIN_PLAN.md §3. "Verified" means a command was run and its output observed.
Commits live on local branch `track-b` only (not pushed). PR draft: `TRACK_B_PR.md`.

Baseline: merlin `183cbb09f2cd`, worktree `/scratch2/loren/merlin-track-b` (branch `track-b`),
separate from the main tree so Track A's uncommitted edits are untouched. Host `garden`.

---

## 2026-10-02

### B-E1. Environment
- **Changed:** worktree-local `.venv` (gitignored), same install recipe as Track A E1.
- **Verified:** `merlin` imports from the worktree `src/`.
- **Baseline suite (pre-existing, not caused by Track B):** `merlin/tests/infra` 180 failed / 6456 passed /
  84 errors; `packages/merlin-experiments/tests` 14 failed / 2601 passed / 22 errors. Dominant causes: no
  `.env` (`MERLIN_EXT_CHIPYARD` unset), no `third_party/llvm-install`, nested bwrap unavailable, tests out
  of step with code. Two failures are worktree-specific (`test_stop_hook_gates_can_block.py` copies
  `.git/index`). Failing ids: scratchpad `failing_ids.txt` (session-local).

### B-E2. M0 audits (read-only)
- **Token-efficiency claim:** runs live in `/scratch/agustin/projects/oscar-merlin/out/runs/gemmini/capsule-bench/`,
  not this repo. No pair supports "fewer tokens than Claude Code at equal qualification": the 6.6M vs 92.9M
  ratio changes model+driver+arm+sha; the Claude-Code-vs-Claude-Code ladder (gemrecreate1) has Merlin arms
  at 19/20 L1 vs baseline 20/20 L3 with hidden never graded; the only like-for-like pairs (g3arm, Codex)
  show Merlin using *more* tokens. Indexed with sha256 in
  `out/artifacts/audits/gemmini/v1/audits_gemmini_v1_20261002T085651Z_183cbb0/` (no hardware verdict;
  `check_provenance.py` OK).
- **Holdout contamination:** `merlin/contract/capsules/_model_layers/G_*` are ResNet-50-derived dev
  capsules; perf-bench kernels `M00–M02` (smolvla), `M05–M07` (tiny_llama); ResNet-50 FireSim q534–q536
  used during Phase 2 development. No frozen dev/holdout split file exists.
- **Coverage seams:** eligibility, conformance cells, boundary classes, placement census exist; no per-item
  join. Jack's ATen coverage (`/scratch/jack/PyTorchFullCoverage/`) is Spike-scalar only and uses `re` +
  a hardcoded gemmini profile.
- **Infra:** VCS/Verilator/Spike/firtool/GCC/bwrap present but not wired (no `.env`). Only gemmini pins
  have matching checkouts; atlas/saturn/radiance/gsim do not. No PowerSpec in repo. No FireSim runner.

### B-E3. B2 coverage inventory join
- **Changed:** `src/merlin/targetgen/coverage_inventory.py` (new); `contract/materialize.py` (extract
  `capsule_cell_rows` from `cert_capsule_cover`, behavior-preserving); `claim_models.py`
  (`mentioned_claim_model`); `merlin/tests/targetgen/test_coverage_inventory.py` (new, 22 tests).
- **Verified:** 22 passed; `check_no_target_name`, `check_no_regex`, `check_structure`,
  `check_artifact_layout` exit 0. Neighbouring tests: 5 pre-existing failures not reaching changed code.
- **Current output (no grading results supplied):** gemmini 44 rows (0 covered / 3 missing / 41 unknown);
  atlas 46 rows (0 / 7 / 39). Atlas: tracked conformance spec demands bf16/elementwise cells the tracked
  contract does not declare → spec likely stale (flagged, not changed).

---

## 2026-10-07

### B-E4. Packaged on a branch
- **Changed:** `origin` → `git@github.com:lorenshung/merlin.git`; old remote kept as `upstream`
  (ucb-bar). Upstream history had moved: `upstream/main` = fork main = `4a7cb1821`, and the 160 local
  commits under `183cbb09f` are not on it. Moved only the Track B commits onto `upstream/main`
  (`git rebase --onto upstream/main 183cbb09f track-b`), clean.
- **Verified on new base:** inventory tests 22 passed; `check_no_target_name`, `check_no_regex`,
  `check_structure`, `check_artifact_layout`, `check_docs`, `check_doc_paths` exit 0; ruff clean.
  Neighbouring tests not re-run on the new base.

## Commits on `track-b`
1. `feat(targetgen): add per-item Phase 1 coverage inventory join` — the four code/test files.
2. `docs(track-b): add working journal and PR draft` — this file and `TRACK_B_PR.md` (drop before merge).
