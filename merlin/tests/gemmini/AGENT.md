# AGENT.md — merlin/tests/gemmini

## Purpose

Tests for the **gemmini** subsystem: Gemmini target: conformance/cert, RTL checks, OOT runner, bench contract.

## Invariants

- Backend-dependent tests require a selected support provider. Outside pytest an unset
  `MERLIN_TARGET_PATH` selects the vendored `examples/gemmini/support` (source revision in
  `examples/gemmini/SOURCE.yaml`); inside a test session `merlin/tests/conftest.py` makes it
  select nothing, so run these with
  `MERLIN_TARGET_PATH=$PWD/examples/gemmini/support`. Provide reviewed RTL facts separately
  for facts-dependent tests. Absence is not
  a passing target qualification. Do not run compiler/simulator tests merely to
  validate a source-layout change.
- Every test file is `merlin/tests/gemmini/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
