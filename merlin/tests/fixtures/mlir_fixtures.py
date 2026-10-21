"""MLIR text builders shared by the IR and targetgen tests.

Most fixtures are one ``@forward`` function around identity-map ``linalg.generic`` ops. These
helpers write that boilerplate, so a test states only its types and scalar body.
"""

import re

from merlin.frontends.linalg_mlir import parse_mlir_text


def element(typ):
    """Element type of ``tensor<...xT>`` (a scalar type is its own element)."""
    match = re.fullmatch(r"tensor<(?:[\d?]+x)*(.+)>", typ)
    return match.group(1) if match else typ


def generic(ins, result, body, *, yields="%v", name="%r", iterators=None, args="abcdefgh"):
    """``name = linalg.generic`` with identity maps over ``ins`` ([(value, type)]) into a fresh empty.

    Block arguments are ``%a, %b, ...`` for the inputs and ``%unused`` for the output; ``body`` is
    the scalar computation and ``yields`` its result.
    """
    rank = len(re.findall(r"[\d?]+x", result))
    dims = ",".join(f"d{i}" for i in range(rank))
    maps = ",".join([f"affine_map<({dims})->({dims})>"] * (len(ins) + 1))
    kinds = ",".join(f'"{kind}"' for kind in iterators or ["parallel"] * rank)
    empty, out = f"{name}_init", element(result)
    params = "".join(f"%{arg}:{element(typ)}," for arg, (_, typ) in zip(args, ins))
    return f"""{empty}=tensor.empty() : {result}
{name}=linalg.generic {{indexing_maps=[{maps}],iterator_types=[{kinds}]}}
  ins({",".join(v for v, _ in ins)}:{",".join(t for _, t in ins)}) outs({empty}:{result}) {{
^bb0({params}%unused:{out}):
{body}
linalg.yield {yields} : {out}
}} -> {result}"""


def forward(params, result, body, *, ret="%r"):
    """Parse ``module { func.func @forward(params) -> result { body; return ret } }``."""
    return parse_mlir_text(
        f"module {{func.func @forward({params}) -> {result} {{\n{body}\nreturn {ret} : {result}\n}} }}"
    )


def pointwise(inputs, result, body, *, prelude="", **kwargs):
    """``@forward(%x, %y, ...)`` over ``inputs`` types returning one identity-map generic of them."""
    ins = list(zip(("%x", "%y", "%z", "%w"), inputs))
    params = ",".join(f"{value}:{typ}" for value, typ in ins)
    return forward(params, result, prelude + "\n" + generic(ins, result, body, **kwargs))


def returned(module):
    """The first value the module's ``func.return`` returns."""
    return next(op for op in module.walk() if op.name == "func.return").operands[0]


def assert_unchanged(module, transform, expected=0):
    """A refusing ``transform`` reports ``expected`` and leaves the module text intact."""
    before = str(module)
    outcome = transform(module)
    assert outcome == expected, outcome
    assert str(module) == before, "refused transform changed the module"
