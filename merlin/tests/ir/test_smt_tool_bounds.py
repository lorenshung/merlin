"""The external SMT translator cannot hang or fill verifier memory indefinitely."""

from __future__ import annotations

import sys

import pytest

from merlin.verify.smt_export import SmtUnavailable, bounded_tool_output


def test_external_tool_timeout_is_unavailable():
    with pytest.raises(SmtUnavailable, match="timed out"):
        bounded_tool_output([sys.executable, "-c", "import time; time.sleep(2)"], timeout_s=0.05)


def test_external_tool_output_is_bounded():
    returncode, stdout, _ = bounded_tool_output([sys.executable, "-c", "print('x' * 100000)"], max_output_bytes=1024)
    assert returncode != 0
    assert len(stdout.encode("utf-8")) <= 1024


def test_external_tool_input_is_bounded():
    with pytest.raises(SmtUnavailable, match="input exceeds"):
        bounded_tool_output([sys.executable, "-c", "pass"], input_text="x" * (16 * 1024 * 1024 + 1))
