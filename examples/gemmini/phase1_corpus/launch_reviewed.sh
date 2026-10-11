#!/usr/bin/env bash
# Operator-only launch after merge and explicit corpus seal. Never seals.
# usage: launch_reviewed.sh <merged-sha> <release> <definition> <seed 0..3>
set -euo pipefail
[ "$#" -eq 4 ] || { echo 'expected merged-sha release definition seed' >&2; exit 64; }
MERGED_SHA=$1
RELEASE=$2
DEFINITION=$3
SEED=$4
case "$SEED" in 0|1|2|3) ;; *) echo 'seed must be 0..3' >&2; exit 64 ;; esac
REPO=${MERLIN_REPO_ROOT:?set MERLIN_REPO_ROOT to the merged checkout}
[ "$(git -C "$REPO" rev-parse HEAD)" = "$MERGED_SHA" ] || { echo 'checkout does not match merged SHA' >&2; exit 65; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo 'tracked source changes remain' >&2; exit 65; }
SEAL=$RELEASE/private/seal.json
BUNDLE=$RELEASE/payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml
[ -f "$SEAL" ] && [ -f "$BUNDLE" ] || { echo 'reviewed seal and release-owned bundle are required' >&2; exit 66; }
export PYTHONPATH=$REPO/src:$REPO/packages/merlin-experiments/src
export MERLIN_TARGET_CONTRACT=$RELEASE/payload/experiment/contracts/target_contract.yaml
export MERLIN_TARGET_EXPERIMENT=$RELEASE/payload/experiment/target_experiment.yaml
: "${MERLIN_TARGET_PATH:?explicit OOT support selection required}"
"$REPO/.venv/bin/python" - "$REPO" <<'PY'
from pathlib import Path
import sys
import merlin, merlin_experiments
root = Path(sys.argv[1])
assert Path(merlin.__file__).is_relative_to(root / 'src')
assert Path(merlin_experiments.__file__).is_relative_to(root / 'packages/merlin-experiments/src')
PY
ARGS=(--phase 1 --corpus-seal "$SEAL" --bundle-manifest "$BUNDLE"
      --phase1-driver codex --phase1-model gpt-6.1-sol --phase1-effort high --phase1-provider subscription)
"$REPO/.venv/bin/merlin" experiment preflight "$DEFINITION" "${ARGS[@]}"
RUN_DIR=${MERLIN_OUT_ROOT:?set MERLIN_OUT_ROOT}/runs/gemmini/phase1/p1-spec-int8-s$SEED-$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR=$MERLIN_OUT_ROOT/runs/gemmini/phase1/_logs
mkdir -p "$LOG_DIR"
# Seeds identify independent sessions, as in the previous release; the transport has no RNG seed flag.
LOG=$LOG_DIR/$(basename "$RUN_DIR").log
printf 'seed=%s merged_sha=%s schedule=continuous model=gpt-6.1-sol effort=high\n' "$SEED" "$MERGED_SHA" > "$LOG"
nohup "$REPO/.venv/bin/merlin" experiment run "$DEFINITION" "${ARGS[@]}" --run-dir "$RUN_DIR" >> "$LOG" 2>&1 &
printf '%s\n' "$!" > "$LOG.pid"
printf 'run_dir=%s log=%s pid=%s\n' "$RUN_DIR" "$LOG" "$!"
