# AGENT.md — src/merlin/capture

Owns shared capture bundle access, deterministic rewrites, and workload identities.
No dependency on experiment orchestration, baseline frameworks, or DSE analysis.
Preserve on-disk formats and numerical tolerances when moving code.

`integerization.py` validates complete integer/preserved-floating contraction
accounting without importing a framework. Preserved BF16/FP16 Q/DQ is source
floating arithmetic, never target admission or integer coverage; unresolved
rewrites and unmatched refusal inventories remain errors.

`safetensors.py` owns bounded header framing and JSON decoding for capture rewrites
and weight packers. It preserves metadata and entry order, never reads tensor data,
and rejects malformed, duplicate-key or non-finite JSON. Its default 16 MiB header
bound is a Merlin policy, not a format limit. Consumers still own tensor layout,
dtype, payload identity and execution-authority validation.
