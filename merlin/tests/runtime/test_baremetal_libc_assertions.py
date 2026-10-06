"""Bare-metal MLIR assertion symbols must print and fail, never silently return."""

import subprocess

from merlin.common.paths import runtime_dir


def test_assertion_console_and_abort(tmp_path):
    harness = runtime_dir() / "baremetal/spike"
    source = tmp_path / "console.c"
    source.write_text(r"""
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
void htif_putc(char c) { (void)write(1, &c, 1); }
void htif_puts(const char *s) { while (*s) htif_putc(*s++); }
__attribute__((noreturn)) void htif_exit(int code) { _Exit(code & 255); }
int main(int argc, char **argv) {
  if (argc > 1) abort();
  if (puts("hello") < 0 || puts("") < 0) return 2;
  return 0;
}
""")
    exe = tmp_path / "console"
    subprocess.run(
        [
            "cc",
            "-O0",
            "-ffreestanding",
            "-fno-builtin",
            "-I",
            str(harness),
            str(source),
            str(harness / "libc_min.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
    )
    ok = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ok.returncode == 0
    assert ok.stdout == "hello\n\n"
    failed = subprocess.run([str(exe), "abort"], capture_output=True, text=True)
    assert failed.returncode == 1
    assert failed.stdout == "FATAL: model assertion abort\n"
    assert "DONE" not in failed.stdout
