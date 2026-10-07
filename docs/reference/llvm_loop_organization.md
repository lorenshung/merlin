# Explicit LLVM loop organization

The default lowering keeps the existing function organization.
`outline_llvm_loops` extracts loops through LLVM's CodeExtractor and marks only
new helpers `noinline`. `outline_llvm_loops_merge_identical` additionally runs
LLVM `mergefunc`. Select one of these policies explicitly.

These are generic host compiler transformations in Merlin. They select neither
a CPU ISA nor an accelerator, and contain no workload or captured-input routing.
Every original function symbol must remain defined after the selected passes.
The actual optimizer commands and returned LLVM identity enter the normal
lowering recipe and optional IR audit.

Merging follows LLVM's function comparator and ordinary function-address rules.
It grants no approximate arithmetic, reassociation, effect removal or alias
permission. Numeric contracts and memory dependencies remain those of the source
IR. Empty loops and original helper inlining attributes are retained.

Helpers add calls, argument traffic and stack costs. Code size, retired
instructions and hardware cycles are separate measurements; reduced code size
alone must not promote a performance candidate. Measure complete source bodies
with their reads, writes, tails and ownership costs, then qualify the original
whole-model accuracy and final executable before hardware admission.
