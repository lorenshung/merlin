# `packages/merlin-experiments/src/merlin_experiments/phase1/feedback`

Host-only grading, redaction, certificate promotion and simulator dispatch live here.
Inputs are explicit invocation context and corpus/contract paths, never native imports.
Keep actual adapter construction, observation order, missing-oracle behavior and
enqueue-time certificate attribution unchanged. Public clients remain separate.
Dispatch owns only the closed async simulator policy shared by broker and promotion.
Source moves invalidate new implementation identities, never rewrite old evidence.

`private_capture_roster.py` binds every single- or multi-program capture stage to
the ordinary root contract and receipts. It refuses missing, extra, opaque or
indirect stage inputs; it neither selects a validation subset nor certifies code.

`caller_layout.py` gives an answer-free, layout-only projection from an explicitly
selected harness provider for a submission-owned command buffer. It never grades or
imports candidate code; absent provider inspection support refuses.

`private_prebuilt_receipt.py` admits only diagnostic inspection of an existing
whole-model build. The shared post-build verifier still checks linked bytes and
source obligations, but an old receipt without exact producer/toolchain closure
cannot become a full-roster gate result or a build-cache hit.

`private_source_freeze.py` owns host-only run copies of authored software and
host-capability inputs selected by a verified Phase 0 derivation export. Bind
original path, semantic role, digest and archive owner; distinct archives may
share one identical byte copy, but conflicting byte pins refuse. Fresh formal
completion requires the run-owned record, workspace snapshot and private masks.
Historical direct-path reads remain diagnostic only. The copy proves selected
source identity, not host-operation correctness or whole-model execution.

`private_control_support.py` proves typed source intervals for a narrow
mask-count assertion chain, its single-consumer comparison predicates, and exact
bounded mask-compaction/scatter cursors. Its index width is a caller-supplied
premise, not selected compiler evidence. The full-model consumer may discharge
only exact proven source ordinals after binding the same producer-owned compiler
observation to the linked build. All other index arithmetic and host math retain
their separate admissions. `private_source_support_join.py` requires all
source-only proofs to cite the same exact linked program; it does not infer
numerical equivalence or preserve arbitrary assertion abort paths.

`private_pointwise_support.py` records exact normalized source ordinals only
when a reviewed host declaration opts into a typed static pointwise body.
It rechecks each parsed operation and joins the same linked whole-program build;
this source witness does not establish numerical equivalence.
`private_linalg_support.py` is the current versioned, mandatory empty-or-
populated witness for exact pointwise, Boolean and unary f32 sine/cosine source-body declarations.
It keeps their schemas separate, rechecks every parsed ordinal, bounds each
static tensor's extent and byte span by the producer-selected signed index
width, and binds that exact lowering record to the linked program. This is
per-tensor source/build evidence, not a global arena proof; it grants no host
rule or selected-libm/PyTorch numerical equivalence.
`private_device_audit.py` also owns exact static board/DTS input checks for the
linked image; that check remains build-only and does not imply board execution.

`private_literal_arange.py` is a grant-none witness for fixed prepared i64
range literals and their closed typed source lowering. It records original
graph ancestry without claiming original-to-prepared equivalence; index width
is an explicit premise until separately bound to the linked compiler build.
`private_literal_arange_admission.py` requires that witness for each reviewed
host-admitted prepared range, binds the selected compiler index observation,
and joins the exact candidate, capture and ELF bytes after whole-program build.
It never creates a host declaration or proves numerical equivalence.

`private_integer_reduction_support.py` rechecks the exact source bodies of
reviewed integer sum, prefix-sum and paired minimum admissions. Its selected
index-width premise must match the producer-owned linked build observation.
`private_ordered_scan_support.py` independently checks a closed prepared f32
prefix scan with f64 carry, ordered loop/lane reset and exact source ancestry.
The v7 gate requires its empty-or-populated reviewed semantic tensor-root
admission roster, independently selected hardware exclusion, and exact proved
body-component ordinals before the selected-width/linked-image join; v6 records
cannot be upgraded. This does not prove original-to-prepared equivalence or
numerical correctness of the compiled loop.
`private_bucketize_support.py` joins the source/trace roster, proves literal
boundary ordering and requires an existing reviewed host placement for every
occurrence. Dynamic boundary ordering remains unproved. Both carry mandatory
empty-or-populated records into the v4 full-model build gate and bind the exact
candidate, capture and ELF. Neither proves frontend or executable numerical
equivalence, creates a host rule, or upgrades historical v3 gate records.

`rtlchecks.py` owns strict authored treatment selection, redaction and advisory
round/checkpoint RTL feedback. Its factory captures the explicit invocation context;
callbacks receive the actual graded public roots, including frozen promoted roots at
checkpoint. Never rediscover a live canonical corpus or select Chipyard at import.
Structural failures/errors do not change numerical pass/fail. The feedback subtree
already carries grader-only access identity; core RTL policy owners are explicit
members of the existing implementation source inventory.

`loop_grading.py` owns the authoring snapshot, language gate, numerical-grade call,
stage ledger, shape gate, promotion and verdict publication in their existing order.
GradingInputs keeps invocation roots separate; only the native diagnostic edge may
defer public-root materialization until after the submission gate. Certification
policy similarly resolves public roots only after finding additional cert tiers.
Keep archive-before-shape and publish-after-promotion ordering unchanged.

`codegen_scalability.py` observes public contractions at geometric multiples of
the selected fact-derived tile. It reads no validation capture or answer; the
ordinary round feedback records emitted text size, not executed work or a
correctness verdict. Size ratios never prescribe unrolling or reject smaller
looped code. Compilation, full-model and native numerical gates remain separate.

`brief.py` constructs and publishes the round brief from already-redacted round
verdicts, the candidate's notes and explicit operator errata. Preserve prompt bytes,
notes-staleness stamps, recognized resume prefixes and pre-launch refresh timing.
Refresh must not advance the notes stamp. This owner does not grade, inspect hidden
corpora or independently sanitize its trusted redacted inputs. Only its generated
brief is candidate-visible; the implementation remains in the private feedback tree.

`lifecycle.py` owns client staging, broker child lifetimes, the background-grading
cadence/single-flight handoff and durable channel-health accounting. BrokerConfig
supplies invocation context, resolved treatment tools, explicit timing storage and
distinct public/policy/schema roots; GradeCadence supplies only scheduler settings.
Grade callbacks are mandatory and contain the caller's existing numerical policy.
Background grading surrounds the agent launch; brokers live inside that launch.
Stop brokers first, then join the grader without a timeout before the authoritative
grade. Partial startup unwinds already-created children and closes parent log
handles. Shutdown attempts all owned processes despite signal/wait/kill failures;
after the 15-second graceful wait, forced stops use a bounded reap. Cleanup errors
are aggregated with unreaped PIDs. A propagating provider or startup exception
remains primary, with cleanup notes; otherwise cleanup failure raises. This owns
direct broker processes, not arbitrary detached descendants, and does not qualify
the full installed engine or unrelated orchestration cancellation.

`formal.py` owns public grade, iteration evidence, freeze/re-hash, hidden grade and
formal completion. `freeze.py` owns the existing freeze.json record using the shared
tree-hash and explicit repository identity. Preserve mandatory L3 simulator selection
even for no-oracle diagnostics. Only synthetic external execution is substituted in
installed lifecycle tests; those tier records are not hardware qualification.
