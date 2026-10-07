"""Multi-output integer families require complete exact bounds and private output spans."""

import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.llvmlower.integer_product_family import (
    IntegerProductFamily,
    IntegerProductGroup,
    c_header,
    family_from_radix_plan,
    reference_outputs,
)
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups


def fixture():
    return IntegerProductFamily(
        3,
        5,
        7,
        2,
        2,
        (IntegerProductGroup(((0, 0),), 7 * 128**2), IntegerProductGroup(((1, 0), (0, 1), (1, 0)), 3 * 7 * 128**2)),
        output_plane_stride=19,
    )


def test_all_prefix_bounds_and_distinct_output_groups():
    f = fixture()
    lhs = [(-128, 127, 0, -17)[i % 4] for i in range(f.lhs_elements)]
    rhs = [(127, -128, 3)[i % 3] for i in range(f.rhs_elements)]
    out = reference_outputs(f, lhs, rhs)
    for group, values in zip(f.groups, out, strict=True):
        independent = []
        for r in range(f.rows):
            for c in range(f.columns):
                partial = 0
                for z in reversed(range(f.reduction_length)):
                    for ap, bp in reversed(group.pairs):
                        partial += (
                            lhs[(ap * f.rows + r) * f.reduction_length + z]
                            * rhs[(bp * f.reduction_length + z) * f.columns + c]
                        )
                        assert abs(partial) <= group.absolute_bound
                independent.append(partial)
        assert tuple(independent) == values
    assert f.output_elements == 34 and f.plane_stride == 19


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": True},
        {"columns": 0},
        {"reduction_length": -1},
        {"lhs_planes": 0},
        {"lhs_magnitude_bound": 129},
        {"rhs_magnitude_bound": -1},
        {"output_plane_stride": 14},
        {"groups": ()},
        {"groups": (IntegerProductGroup((), 0),)},
        {"groups": (IntegerProductGroup(((2, 0),), 1000000),)},
        {"groups": (IntegerProductGroup(((True, 0),), 1000000),)},
        {"groups": (IntegerProductGroup(((0, 0),), 7 * 128**2 - 1),)},
        {"groups": (IntegerProductGroup(((0, 0),), 1 << 31),)},
        {"output_plane_stride": 1 << 63},
    ],
)
def test_malformed_or_unproved_domains_refuse(changes):
    with pytest.raises(ValueError):
        replace(fixture(), **changes)


def test_radix_adapter_revalidates_existing_exact_proof():
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=192)
    family = family_from_radix_plan(plan, rows=17, columns=23)
    assert len(family.groups) == 5
    assert tuple(len(g.pairs) for g in family.groups) == (1, 2, 3, 2, 1)
    assert family.groups[2].absolute_bound == 3 * 192 * 127**2
    with pytest.raises(ValueError, match="proof changed"):
        family_from_radix_plan(replace(plan, weighted_absolute_bound=0), rows=17, columns=23)


def test_oracle_rejects_incomplete_or_unproved_operands():
    family = replace(fixture(), lhs_magnitude_bound=1)
    with pytest.raises(ValueError, match="complete"):
        reference_outputs(family, [], [])
    with pytest.raises(ValueError, match="domain"):
        reference_outputs(family, [2] * family.lhs_elements, [0] * family.rhs_elements)


def test_checked_callback_and_output_holes(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler required")
    f = fixture()
    (tmp_path / "family.h").write_text(c_header(f, symbol="apply_family"))
    source = tmp_path / "check.c"
    source.write_text(r"""
#include <assert.h>
#include "family.h"
static int calls;
static int cb(void *opaque,const int8_t*a,const int8_t*b,int32_t*c,int m,int n,int k,size_t stride){
 assert(opaque==(void*)7 && m==3 && n==5 && k==7 && stride==19);calls++;
 for(int g=0;g<2;g++)for(int i=0;i<m*n;i++)c[g*stride+i]=100*g+i;
 (void)a;(void)b;return 1;
}
int main(void){
 int8_t a[42]={0},b[70]={0};int32_t c[36];for(int i=0;i<36;i++)c[i]=-9;
 assert(apply_family(cb,(void*)7,a,sizeof(a),b,sizeof(b),c,34));assert(calls==1);
 for(int i=0;i<15;i++)assert(c[i]==i&&c[19+i]==100+i);
 for(int i=15;i<19;i++)assert(c[i]==-9);assert(c[34]==-9&&c[35]==-9);
 assert(!apply_family(cb,(void*)7,a,41,b,70,c,34));assert(calls==1);
 assert(!apply_family(cb,(void*)7,a,42,b,70,c,33));assert(calls==1);
 assert(!apply_family(cb,(void*)7,(int8_t*)c,42,b,70,c,34));assert(calls==1);
 assert(!apply_family(0,(void*)7,a,42,b,70,c,34));assert(calls==1);
 return 0;
}
""")
    exe = tmp_path / "check"
    subprocess.run(
        [cc, "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all", str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="identifier"):
        c_header(f, symbol="bad-name")
