# AGENT.md — build_tools/upstreams

## Purpose

Versioned provenance and qualification records for coordinated local companion-repository changes, and
for their vendored copies under `examples/*/support`.
These records are not discovery defaults, target hardware facts, or compiler certificates.

## Invariants

- Preserve source revisions and exact content hashes; distinguish public catalog, candidate, schedule,
  and support roles.
- A local companion commit is not a pushed upstream release. Never infer a qualified compiler from a
  schema-valid support contract or an ABI-looking publication wrapper.
- No canonical target source is removed until its consumers and behavioral qualifiers migrate.
- Each companion's provider root is vendored byte-identically at `examples/<example>/support`, with its
  source commit, tree and date in `examples/<example>/SOURCE.yaml`; the manifest's `vendored` block
  must agree with that record. That vendored copy is the runtime default when `MERLIN_TARGET_PATH` is
  unset; an explicit value always wins. Ignored local clone paths in manifests are operator records only.
- Re-vendoring is a new copy from a recorded companion commit, never an in-place edit of a vendored tree.
