# AGENT.md — build_tools/upstreams

## Purpose

Versioned provenance and qualification records for coordinated local companion-repository changes,
historical vendored snapshots, and canonical example support under `examples/*/support`.
These records are not discovery defaults, target hardware facts, or compiler certificates.

## Invariants

- Preserve source revisions and exact content hashes; distinguish public catalog, candidate, schedule,
  and support roles.
- A local companion commit is not a pushed upstream release. Never infer a qualified compiler from a
  schema-valid support contract or an ABI-looking publication wrapper.
- No canonical target source is removed until its consumers and behavioral qualifiers migrate.
- A historical vendored provider pins its companion commit, source tree and date in
  `examples/<example>/SOURCE.yaml`; the registry's `vendored` block must agree. Only files listed
  under `normalized`, with original companion blob IDs and changes, may differ from that tree.
  Re-vendoring requires a new recorded companion commit, never an in-place edit.
- Canonical example support instead pins its **current tracked tree** and file count in `SOURCE.yaml`
  and the registry's `canonical_example` block. Its external `origin` is historical provenance,
  never a runtime checkout or a claim that current bytes equal the origin commit. Changes require
  a new reviewed tree identity; do not fabricate a self-commit SHA.
- Either form is the host-private runtime default when `MERLIN_TARGET_PATH` is unset; an explicit
  value always wins. Ignored local clone paths are operator history, not runtime defaults.
