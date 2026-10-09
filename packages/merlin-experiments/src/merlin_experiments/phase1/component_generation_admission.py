"""Fresh authoring admission for actual bounded Phase 0 reference generation.

Legacy coverage remains inspectable through its original report reader. A fresh
origin requires the complete budgeted ledger and ordinary source-cost replay.
These bounds do not qualify process heap, compiler execution or hardware timing.
"""

from pathlib import Path

from merlin_experiments.phase0.component_coverage import BUDGETED_REPORT_SCHEMA, verify_report
from merlin_experiments.phase2.contracts import StageGateError


def verify_bounded_generation(corpus_root: Path) -> dict:
    """Reopen actual source/member costs before fresh compiler-origin admission."""
    try:
        root = Path(corpus_root).absolute()
        if any(path.is_symlink() for path in (root, *root.parents)) or root.resolve() != root:
            raise StageGateError("fresh Phase 1 bounded generation refuses indirect paths")
        report = verify_report(root)
    except (OSError, TypeError, ValueError) as error:
        raise StageGateError("fresh Phase 1 bounded generation is unavailable: " + str(error)) from error
    if report["schema"] != BUDGETED_REPORT_SCHEMA:
        raise StageGateError("fresh Phase 1 requires verified budgeted Phase 0 coverage v2")
    return report
