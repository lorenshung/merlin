"""An emulator's stderr is bounded on disk (head and tail kept, the middle counted and stated) and the run
ends at its first hardware assertion, by its own handle, instead of at its cycle budget."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from merlin.perf import emulator_capture as EC

_FLOOD = """
import sys
w = sys.stderr.write
w("[tsi] load path: backdoor\\n")
line = "x" * 99 + "\\n"
for i in range({lines}):
    w(line)
w("[gsim-emu] FINISHED: cycles=60000000 wall=1s (1 cyc/s) done=0 exit_code=0\\n")
"""

_ASSERTS_FOREVER = """
import sys, time
w = sys.stderr.write
w("[gsim-emu] booted\\n")
sys.stderr.flush()
while True:
    w("\\x1b[1;31m\\x1b[0m\\nAssertion failed: pipeline stall\\n    at ReservationStation.scala:568 assert(x)\\n")
    sys.stderr.flush()
    time.sleep(0.001)
"""


def _run(tmp_path: Path, script: str, **kw) -> tuple[EC.Capture, bytes, float]:
    started = time.monotonic()
    capture = EC.run_capped(
        [sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr_path=tmp_path / "err.txt", **kw
    )
    return capture, (tmp_path / "err.txt").read_bytes(), time.monotonic() - started


def test_a_flooded_stderr_keeps_its_head_and_tail_and_states_what_it_dropped(tmp_path):
    capture, kept, _ = _run(tmp_path, _FLOOD.format(lines=50_000), head_bytes=64 * 1024, tail_bytes=64 * 1024)
    assert capture.total_bytes > 5_000_000 and capture.returncode == 0
    assert len(kept) < 64 * 1024 * 2 + 200 and capture.kept_bytes + capture.elided_bytes == capture.total_bytes
    text = kept.decode()
    assert text.startswith("[tsi] load path: backdoor\n")
    assert f"{EC.ELISION_PREFIX} {capture.elided_bytes} bytes" in text
    # The end-of-run line every reader parses survives, and the kept tail starts on a whole line.
    assert text.rstrip().endswith("done=0 exit_code=0")
    after = text.split(EC.ELISION_PREFIX, 1)[1].split("\n", 1)[1]
    assert after.startswith("x" * 99 + "\n")


def test_a_small_stderr_is_kept_whole_with_no_elision_line(tmp_path):
    capture, kept, _ = _run(tmp_path, _FLOOD.format(lines=10))
    assert capture.elided_bytes == 0 and capture.kept_bytes == capture.total_bytes == len(kept)
    assert EC.ELISION_PREFIX.encode() not in kept and capture.assertion is None


def test_the_first_hardware_assertion_stops_the_run_and_is_recorded(tmp_path):
    capture, kept, wall = _run(tmp_path, _ASSERTS_FOREVER, timeout_s=60, grace_s=5)
    assert wall < 30 and not capture.timed_out
    assert capture.stopped_on_assertion and capture.returncode != 0
    assert capture.assertion["message"] == "pipeline stall"
    assert capture.assertion["location"].startswith("ReservationStation.scala:568")
    assert kept.startswith(b"[gsim-emu] booted\n") and capture.assertion["at_byte"] > 0


def test_an_assertion_is_recorded_but_not_acted_on_when_stopping_is_off(tmp_path):
    script = "import sys; sys.stderr.write('Assertion failed: x\\n    at A.scala:1 assert(c)\\nafter\\n')"
    capture, kept, _ = _run(tmp_path, script, stop_on_assertion=False)
    assert capture.returncode == 0 and not capture.stopped_on_assertion
    assert capture.assertion == {"message": "x", "location": "A.scala:1 assert(c)", "at_byte": 0}
    assert kept.endswith(b"after\n")


def test_a_line_that_quotes_the_prefix_is_not_an_assertion(tmp_path):
    script = "import sys; sys.stderr.write('note: Assertion failed lines are counted\\n')"
    capture, _, _ = _run(tmp_path, script)
    assert capture.assertion is None and not capture.stopped_on_assertion


def test_an_assertion_split_across_reads_is_still_found(tmp_path):
    script = (
        "import sys, time\n"
        "sys.stderr.write('Assert'); sys.stderr.flush(); time.sleep(0.2)\n"
        "sys.stderr.write('ion failed: split\\n'); sys.stderr.flush(); time.sleep(30)\n"
    )
    capture, _, wall = _run(tmp_path, script, timeout_s=60, grace_s=5)
    assert capture.assertion["message"] == "split" and capture.stopped_on_assertion and wall < 30


def test_the_wall_clock_budget_still_ends_a_silent_run(tmp_path):
    capture, _, wall = _run(tmp_path, "import time; time.sleep(30)", timeout_s=0.5, grace_s=2)
    assert capture.timed_out and capture.returncode is None and wall < 15


def test_the_child_pid_is_handed_to_the_caller(tmp_path):
    seen = []
    capture, _, _ = _run(tmp_path, "pass", on_start=seen.append)
    assert capture.returncode == 0 and len(seen) == 1 and isinstance(seen[0], int)
