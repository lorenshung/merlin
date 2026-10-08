# Gemmini Core ATen 193 experiment

Catalog ID: `gemmini-core-aten-193-functional`. EL4 `capsule_bench` Phase 1 on
all 193 canonical Core ATen overload capsules (PUBLIC) with a held-out cohort
over the same overloads (HIDDEN), L2 only, continuous schedule, 12-hour budgets.
The device-focused sibling is `examples/gemmini/core_aten`.

Capsules declare `lane_expectation: any`: Merlin's host lane executes a call the
submitted device catalog does not cover, and a covered call is routed through
Merlin DeviceRouting. Every scored result keeps its observed `lane` and
`executed_instructions`; the verdict reports `host_lane_scored_pass` and
`device_lane_scored_pass` separately, the latter requiring retired accelerator
instructions matched to the linked ELF. Some calls (random-number generators)
are expected to fail on every lane and stay in the corpus.

Produce the corpus with a real Phase 0 run: write a recipe whose `core_aten`
block selects all 193 canonical cases as `public` and a same-overload cohort as
`hidden` (both rows with `lane_expectation: any`; `host_guard` is optional),
then `merlin experiment corpus derive`, `merlin experiment run <def> --phase 0`,
`corpus prepare --generated-only` and `corpus inspect`. Stage operator inputs with
`examples/gemmini/core_aten/stage_inputs.py --definition examples/gemmini/core_aten_193`.
Sealing and launching are operator steps.
