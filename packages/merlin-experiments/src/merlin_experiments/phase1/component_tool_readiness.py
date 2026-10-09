"""Original shared-tool probes through an explicitly selected command owner.

Tool readiness does not qualify compiler semantics, author isolation or runtime.
The fresh input owner supplies its already admitted exact transport and policy.
"""

import subprocess

from merlin.common import invocation_record
from merlin_experiments.execution.container_transport import PreparedContainerTransport, mounts_from_strict_policy
from merlin_experiments.phase2 import contracts as C


def probe_shared_tools(inputs, policy, *, compiler_transport):
    if compiler_transport is not None:
        if type(compiler_transport) is not PreparedContainerTransport:
            raise C.StageGateError("shared-tool readiness requires the actual prepared compiler command transport")
        compiler_transport.verify()
    probes = inputs.output / "readiness"
    probes.mkdir(mode=0o700)
    for probe in inputs.readiness:
        dependencies = tuple(row.source for row in inputs.runtime)
        if compiler_transport is None:
            with invocation_record.observe(
                probes / probe.capability,
                stage="fresh_phase1_" + probe.capability,
                argv=[*policy, "--", *probe.command],
                dependencies=dependencies,
            ) as observation:
                result = subprocess.run([*policy, "--", *probe.command], capture_output=True, timeout=45)
                observation.complete(result)
        else:
            mounts = mounts_from_strict_policy(policy)
            with invocation_record.observe_call(
                probes / probe.capability,
                stage="fresh_phase1_" + probe.capability,
                function=compiler_transport.execute,
                arguments={"command": probe.command, "transport_sha256": compiler_transport.sha256},
                dependencies=dependencies,
            ) as observation:
                result = compiler_transport.execute(
                    mounts=mounts,
                    command=probe.command,
                    cwd=str(inputs.candidate),
                    evidence_root=probes / probe.capability / "container-command",
                    timeout_s=45,
                )
                observation.returned(stdout=result.stdout, stderr=result.stderr)
        if result.returncode or C.sha256_file(observation.directory / "stdout.bin") != probe.stdout_sha256:
            raise C.StageGateError("fresh Phase 1 admitted tool readiness failed: " + probe.capability)
