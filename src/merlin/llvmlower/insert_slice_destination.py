"""Exact destination passing for a pointwise producer inserted into a splat.

The scalar body is cloned unchanged. Static unit-stride slices, an unused output
block argument, a sole tensor-result use, and pure arithmetic are required.
A fresh fill gives the padded destination writable storage; an extract-slice is
then the producer's destination. Bufferization owns alias legality.

The explicit ``initialize_border_only`` option fills the disjoint rectangular
complement instead. The unread interior is completely overwritten by the same
scalar producer. A live original splat remains immutable and separate. This
option changes no arithmetic, input layout, call ABI, or default selection.
"""

from xdsl.dialects import tensor
from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.dialects.linalg.ops import FillOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap


def exterior_slabs(shape, offsets, sizes):
    """Disjoint rectangular complement of one in-bounds rectangular interior.

    Earlier dimensions are restricted to the interior; the current dimension
    selects its lower or upper exterior. Every exterior point therefore belongs
    to the slab for its first out-of-interior dimension, exactly once.
    """
    if not (len(shape) == len(offsets) == len(sizes)):
        raise ValueError("rank mismatch")
    if any(d <= 0 or n <= 0 or o < 0 or o + n > d for d, o, n in zip(shape, offsets, sizes)):
        raise ValueError("invalid static interior")
    result = []
    for axis, (dim, offset, size) in enumerate(zip(shape, offsets, sizes)):
        for start, extent in ((0, offset), (offset + size, dim - offset - size)):
            if not extent:
                continue
            slab_offsets = tuple(offsets[:axis]) + (start,) + (0,) * (len(shape) - axis - 1)
            slab_sizes = tuple(sizes[:axis]) + (extent,) + tuple(shape[axis + 1 :])
            result.append((slab_offsets, slab_sizes))
    return result


def _border_destination(empty, value, shape, offsets, sizes):
    current = empty.tensor
    operations = [empty]
    for slab_offsets, slab_sizes in exterior_slabs(shape, offsets, sizes):
        view = tensor.ExtractSliceOp.from_static_parameters(current, slab_offsets, slab_sizes, (1,) * len(shape))
        fill = FillOp([value], [view.result], [view.result.type])
        insert = tensor.InsertSliceOp.from_static_parameters(
            fill.results[0], current, slab_offsets, slab_sizes, (1,) * len(shape)
        )
        operations.extend((view, fill, insert))
        current = insert.result
    return operations, current


def rewrite_module(module, *, initialize_border_only=False):
    accepted = []
    for insert in list(module.walk()):
        if not isinstance(insert, tensor.InsertSliceOp):
            continue
        producer = insert.source.owner
        pad = insert.dest.owner
        if (
            not isinstance(producer, Operation)
            or producer.name != "linalg.generic"
            or not isinstance(pad, tensor.SplatOp)
        ):
            continue
        if (
            producer.parent_block() is not insert.parent_block()
            or len(producer.results) != 1
            or len(producer.outputs) != 1
        ):
            continue
        init_owner = producer.outputs[0].owner
        if (
            len(list(insert.source.uses)) != 1
            or not isinstance(init_owner, Operation)
            or init_owner.name != "tensor.empty"
        ):
            continue
        source_type = insert.source.type
        dest_type = insert.dest.type
        if any(
            not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr) or any(d <= 0 for d in t.get_shape())
            for t in (source_type, dest_type)
        ):
            continue
        rank = len(source_type.get_shape())
        if len(dest_type.get_shape()) != rank or source_type.get_element_type() != dest_type.get_element_type():
            continue
        offsets = tuple(insert.static_offsets.get_values())
        sizes = tuple(insert.static_sizes.get_values())
        strides = tuple(insert.static_strides.get_values())
        if (
            len(offsets) != rank
            or sizes != source_type.get_shape()
            or strides != (1,) * rank
            or any(o < 0 or o + n > d for o, n, d in zip(offsets, sizes, dest_type.get_shape()))
        ):
            continue
        if insert.offsets or insert.sizes or insert.strides or pad.dynamicSizes:
            continue
        if producer.properties["indexing_maps"].data[-1].data != AffineMap.identity(rank):
            continue
        if any(x.data.value != "parallel" for x in producer.properties["iterator_types"].data):
            continue
        body = producer.body.block
        if body.args[-1].uses:
            continue
        if any(
            op.regions or not (op.name.startswith(("arith.", "math.")) or op.name == "linalg.yield") for op in body.ops
        ):
            continue
        empty = tensor.EmptyOp([], dest_type)
        if initialize_border_only:
            initialization, destination = _border_destination(empty, pad.input, dest_type.get_shape(), offsets, sizes)
        else:
            fill = FillOp([pad.input], [empty.tensor], [dest_type])
            initialization, destination = [empty, fill], fill.results[0]
        view = tensor.ExtractSliceOp.from_static_parameters(destination, offsets, sizes, strides)
        clone = producer.clone()
        clone.operands = [*producer.inputs, view.result]
        replacement = tensor.InsertSliceOp.from_static_parameters(
            clone.results[0], destination, offsets, sizes, strides
        )
        replacement.attributes.update(insert.attributes)
        block = insert.parent_block()
        block.insert_ops_before([*initialization, view, clone, replacement], insert)
        insert.result.replace_all_uses_with(replacement.result)
        block.erase_op(insert)
        block.erase_op(producer)
        accepted.append(
            {
                "source_shape": list(source_type.get_shape()),
                "destination_shape": list(dest_type.get_shape()),
                "offsets": list(offsets),
                "scalar_body": "unchanged clone",
                "storage": (
                    "fresh exterior slabs with fully overwritten interior"
                    if initialize_border_only
                    else "fresh fill with producer writing exact interior slice"
                ),
            }
        )
    module.verify()
    return accepted


def rewrite_prepared_file(source, work, *, initialize_border_only=False):
    import hashlib
    import json
    from pathlib import Path

    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects._common import text

    source, work = Path(source), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    original = source.read_bytes()
    module = parse_mlir_file(source)
    routes = rewrite_module(module, initialize_border_only=initialize_border_only)
    target = work / "model.mlir"
    target.write_text(text(module))
    parse_mlir_file(target).verify()
    report = {
        "schema": "exact_insert_slice_destination_v1",
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "rewritten_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "routes": routes,
        "scalar_arithmetic_changed": False,
        "call_abi_changed": False,
    }
    (work / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[destination_reuse] pointwise insert-slice rewrites={len(routes)}")
    return target
