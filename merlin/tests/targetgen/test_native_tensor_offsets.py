"""Tensor literals and intervening casts cannot conceal biased rounding."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def source(offset, rounding, *, splat=False):
    constant = ('%s = arith.constant '+offset+' : f32\n%c = tensor.splat %s : tensor<4xf32>' if splat else
                '%c = arith.constant '+offset+' : tensor<4xf32>')
    return parse_mlir_text('''module { func.func @f(%x: tensor<4xf32>) -> tensor<4xf32> {
      ''' + constant + '''
      %e = tensor.empty() : tensor<4xf32>
      %r = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
        ins(%x,%c : tensor<4xf32>,tensor<4xf32>) outs(%e : tensor<4xf32>) {
        ^bb0(%a:f32,%b:f32,%unused:f32):
          %sum = arith.addf %a,%b : f32
          ''' + rounding + '''
          linalg.yield %v : f32
        } -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    }}''')


ROUNDINGS = [
    '%n = arith.fptosi %sum : f32 to i32\n%v = arith.sitofp %n : i32 to f32',
    '%v = math.trunc %sum : f32',
    '%v = math.round %sum : f32',
    '%v = math.roundeven %sum : f32',
    '%n = arith.truncf %sum : f32 to f16\n%v = arith.extf %n : f16 to f32',
    '%wide = arith.extf %sum : f32 to f64\n%n = arith.fptosi %wide : f64 to i32\n%v = arith.sitofp %n : i32 to f32',
    '%one = arith.constant 1.0 : f32\n%p = arith.mulf %sum,%one : f32\n%v = math.floor %p : f32',
]


@pytest.mark.parametrize('rounding', ROUNDINGS)
@pytest.mark.parametrize('offset', ['dense<0.5>', 'dense<[0.0,0.5,0.0,-0.5]>'])
def test_dense_offsets_before_every_rounding_observation(rounding, offset):
    m = source(offset, rounding)
    m.verify()
    result = construction_evidence(m, excluded=['readout_offset_rounding'])
    assert result['status'] == 'host', result
    assert any(e['reason'] == 'readout_offset_rounding' for e in result['evidence'])


@pytest.mark.parametrize('rounding', ROUNDINGS)
def test_scalar_splat_offset_is_not_an_escape(rounding):
    assert construction_evidence(source('0.5', rounding, splat=True),
                                 excluded=['readout_offset_rounding'])['status'] == 'host'


@pytest.mark.parametrize('rounding', ROUNDINGS)
def test_zero_tensor_offset_is_not_biased_rounding(rounding):
    assert construction_evidence(source('dense<-0.0>', rounding),
                                 excluded=['readout_offset_rounding'])['status'] == 'eligible'


def test_literal_affine_offset_without_rounding_remains_eligible():
    assert construction_evidence(source('dense<0.5>', '%v = arith.addf %sum,%a : f32'))['status'] == 'eligible'
