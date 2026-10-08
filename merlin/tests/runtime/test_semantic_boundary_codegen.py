"""Native boundary reconstruction preserves shared storage and strided logical values."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.semantic_io import generate_observer


def test_strided_alias_is_materialized_and_read_back_from_one_storage(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    contract = dict(
        input_indices=[0],
        post_indices=[1],
        outputs=[
            dict(
                dtype="float32", shape=[2, 2], stride=[1, 2], storage_offset=0, requires_grad=False, alias_outputs=[1]
            ),
            dict(dtype="float32", shape=[4], stride=[1], storage_offset=0, requires_grad=False, alias_outputs=[]),
        ],
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    generate_observer(
        path,
        tmp_path / "semantic_io.h",
        [([2, 2], "f32"), ([4], "f32")],
        {0: (np.arange(4, dtype=np.float32), "f32")},
        {"f32": 4},
    )
    source = tmp_path / "main.c"
    source.write_text("""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static float input[]={1,2,3,4}, view[]={1,3,2,4}, post[]={1,2,3,4};
static void *MERLIN_INPUT_PTR[]={input};
static void *MERLIN_OUTPUT_PTR[]={view,post};
static size_t MERLIN_OUTPUT_NBYTES[]={16,16};
static void htif_puts(const char *s) { fputs(s,stdout); }
static void htif_putd(long v) { printf("%ld",v); }
static void htif_putc(char v) { putchar(v); }
#include "semantic_io.h"
int main(void) {
 merlin_semantic_before(); merlin_semantic_after();
 for(int i=0;i<2;i++) { printf("VALUES %d",i); for(int j=0;j<4;j++) printf(" %.0f",((float*)MERLIN_OUTPUT_PTR[i])[j]); puts(""); }
 return 0;
}
""")
    subprocess.run([compiler, str(source), "-o", str(tmp_path / "run")], check=True, capture_output=True)
    result = subprocess.run([str(tmp_path / "run")], check=True, capture_output=True, text=True).stdout
    assert "OUT_META 0 0 0 2 2 1 2 2" in result
    assert "OUT_ALIAS 0 1" in result
    assert "VALUES 0 1 3 2 4" in result
    assert "VALUES 1 1 2 3 4" in result
