"""Native exact-family admission requires source closure and independent bits."""

import hashlib

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_exact_dtype import bitwise_catalog_coverage
from merlin.targetgen.core_aten_host_floor import grade_native_bits
from merlin.targetgen.software_spec import admit_operation


def module_and_manifest(*, arithmetic=False):
    body = '''%zero = arith.constant 0 : i64
      %input = tensor.splat %zero : tensor<2xi64>
      %e = tensor.empty() : tensor<2xi64>
      %copy = linalg.copy ins(%input : tensor<2xi64>) outs(%e : tensor<2xi64>) -> tensor<2xi64>
      func.return %copy : tensor<2xi64>'''
    module = parse_mlir_text('builtin.module { func.func @forward() -> tensor<2xi64> {' + body + '} }')
    operations = list(module.walk())
    root = next(op for op in operations if op.name == 'linalg.copy')
    splat = next(op for op in operations if op.name == 'tensor.splat')
    constant = next(op for op in operations if op.name == 'arith.constant')
    kernel = dict(symbol='copy_bytes', meta=dict(proof='closed_static_element_selectors', readout='scratchpad_bytes'))
    numeric = dict(family='F1', quantization='none', exact_arithmetic=True, byte_order='little')
    bindings = [dict(root_ordinal=operations.index(root), symbol='copy_bytes', numeric=numeric,
                     operands={'x': dict(op_ordinal=operations.index(splat), result=0)})]
    if not arithmetic:
        bindings.append(dict(root_ordinal=operations.index(splat), symbol='copy_bytes', numeric=numeric,
                             operands={'literal': dict(op_ordinal=operations.index(constant), result=0)}))
    manifest = dict(source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
                    bindings=bindings, kernels=[kernel])
    return module, manifest


def test_closed_bitwise_lineage_and_source_identity():
    module, manifest = module_and_manifest()
    assert len(bitwise_catalog_coverage(module, manifest)) == 2
    manifest['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='different source'):
        bitwise_catalog_coverage(module, manifest)


def test_a_device_copy_does_not_admit_unbound_producer_work():
    module, manifest = module_and_manifest(arithmetic=True)
    with pytest.raises(ValueError, match='unbound tensor.splat'):
        bitwise_catalog_coverage(module, manifest)


@pytest.mark.parametrize('key,value', [('proof', 'unchecked'), ('readout', 'float_scale')])
def test_incomplete_kernel_proof_refuses(key, value):
    module, manifest = module_and_manifest()
    manifest['kernels'][0]['meta'][key] = value
    with pytest.raises(ValueError, match='kernel proof'):
        bitwise_catalog_coverage(module, manifest)


def test_family_constraint_is_executable_and_cannot_admit_ordinary_sites():
    declaration = dict(schema='merlin.software_spec.v1', target='test', status='reviewed',
        operations=[dict(id='bytes', families=['movement'], placement='accelerator',
                         signature=dict(numerical_family=['F1']))])
    assert admit_operation(declaration, 'movement', dict(family='movement'), 'accelerator')['status'] != 'admitted'
    assert admit_operation(declaration, 'movement', dict(family='movement', numerical_family='F1'),
                           'accelerator')['status'] == 'admitted'
    assert admit_operation(declaration, 'movement', dict(family='movement', numerical_family='F2'),
                           'accelerator')['status'] == 'unsupported'


@pytest.mark.parametrize('reference,changed', [
    ('00000000', '00000080'),  # signed zero
    ('0100c07f', '0200c07f'),  # different quiet NaN payloads
    ('ffffffffffffff7f', '0000000000000080'),  # int64 limits
    ('0100803f00000080', '0100803f00000000'),  # complex imaginary zero
])
def test_host_floor_compares_storage_bits(reference, changed):
    record = dict(output_indices=[0], native_eager_indices=dict(outputs=[0], posts=[1]),
                  native_eager_boundary=dict(authority='untransformed_native_cpu_eager',
                                             output_bytes=[reference], post_input_bytes=['a5']))
    assert grade_native_bits(record, [bytes.fromhex(reference), b'\xa5'])['status'] == 'pass'
    assert grade_native_bits(record, [bytes.fromhex(changed), b'\xa5'])['status'] == 'mismatch'
    assert grade_native_bits(record, [bytes.fromhex(reference), b'\xa4'])['status'] == 'mismatch'
    assert grade_native_bits(record, [bytes.fromhex(reference)])['status'] == 'mismatch'
