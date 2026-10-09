"""Original small RTL semantics and stimuli; no target, model or cycle inputs."""

import json


def stimuli(inputs, outputs, rows):
    return json.dumps(
        {
            "byte_order": "little",
            "inputs": inputs,
            "outputs": outputs,
            "samples": [
                {"id": i, "values": values, "evaluations": 2, "observe": observe}
                for i, (values, observe) in enumerate(rows)
            ],
        }
    ).encode()


def rw_memory():
    source = b"""FIRRTL version 4.0.0
circuit NativeMemory :
  public module NativeMemory :
    input clock : Clock
    input addr : UInt<2>
    input en : UInt<1>
    input write : UInt<1>
    input mask : UInt<1>
    input data : UInt<8>
    output result : UInt<8>
    mem storage :
      data-type => UInt<8>
      depth => 4
      read-latency => 1
      write-latency => 1
      readwriter => rw
      read-under-write => undefined
    connect storage.rw.addr, addr
    connect storage.rw.en, en
    connect storage.rw.clk, clock
    connect storage.rw.wmode, write
    connect storage.rw.wmask, mask
    connect storage.rw.wdata, data
    connect result, storage.rw.rdata
"""
    rows = []
    for addr, value in enumerate((17, 34, 51, 68)):
        rows.extend((([0, addr, 1, 1, 1, value], False), ([1, addr, 1, 1, 1, value], False)))
    rows.extend(
        (
            ([0, 2, 1, 0, 1, 0], False),
            ([1, 2, 1, 0, 1, 0], True),
            ([0, 3, 1, 0, 1, 0], True),
            ([1, 3, 1, 0, 1, 0], True),
            ([1, 0, 1, 0, 1, 0], True),
            ([0, 0, 1, 0, 1, 0], True),
            ([1, 0, 1, 0, 1, 0], True),
            ([0, 3, 1, 1, 0, 99], False),
            ([1, 3, 1, 1, 0, 99], False),
            ([0, 3, 1, 0, 1, 0], False),
            ([1, 3, 1, 0, 1, 0], True),
            ([0, 3, 0, 1, 1, 99], False),
            ([1, 3, 0, 1, 1, 99], False),
            ([0, 3, 1, 0, 1, 0], False),
            ([1, 3, 1, 0, 1, 0], True),
        )
    )
    expected = ((9, 51), (10, 51), (11, 68), (12, 68), (13, 68), (14, 17), (18, 68), (22, 68))
    return source, stimuli(["clock", "addr", "en", "write", "mask", "data"], ["result"], rows), expected


def separate_memory():
    source = b"""FIRRTL version 4.0.0
circuit NativeMemory :
  public module NativeMemory :
    input clock : Clock
    input addr : UInt<2>
    input read_enable : UInt<1>
    input write_enable : UInt<1>
    input mask : UInt<1>
    input data : UInt<8>
    output result : UInt<8>
    mem storage :
      data-type => UInt<8>
      depth => 4
      read-latency => 0
      write-latency => 1
      reader => r
      writer => w
      read-under-write => undefined
    connect storage.r.addr, addr
    connect storage.r.en, read_enable
    connect storage.r.clk, clock
    connect storage.w.addr, addr
    connect storage.w.en, write_enable
    connect storage.w.clk, clock
    connect storage.w.mask, mask
    connect storage.w.data, data
    connect result, storage.r.data
"""
    rows = []
    for addr, value in enumerate((17, 34, 51, 68)):
        rows.extend((([0, addr, 0, 1, 1, value], False), ([1, addr, 0, 1, 1, value], False)))
    rows.extend(
        (
            ([0, 0, 1, 0, 1, 0], True),
            ([0, 3, 1, 0, 1, 0], True),
            ([0, 3, 0, 1, 0, 99], False),
            ([1, 3, 0, 1, 0, 99], False),
            ([0, 3, 1, 0, 1, 0], True),
            ([0, 3, 0, 0, 1, 99], False),
            ([1, 3, 0, 0, 1, 99], False),
            ([0, 3, 1, 0, 1, 0], True),
            # A write input without an edge must not change memory.
            ([0, 0, 0, 1, 1, 99], False),
            ([0, 0, 1, 0, 1, 0], True),
        )
    )
    expected = ((8, 17), (9, 68), (12, 68), (15, 68), (17, 17))
    return source, stimuli(["clock", "addr", "read_enable", "write_enable", "mask", "data"], ["result"], rows), expected


def two_clocks():
    source = b"""FIRRTL version 4.0.0
circuit NativeClock :
  public module NativeClock :
    input clock_a : Clock
    input clock_b : Clock
    input reset : UInt<1>
    output a : UInt<8>
    output b : UInt<8>
    regreset count_a : UInt<8>, clock_a, reset, UInt<8>(0)
    regreset count_b : UInt<8>, clock_b, reset, UInt<8>(0)
    connect count_a, bits(add(count_a, UInt<8>(1)), 7, 0)
    connect count_b, bits(add(count_b, UInt<8>(1)), 7, 0)
    connect a, count_a
    connect b, count_b
"""
    rows = [
        ([0, 0, 1], False),
        ([1, 1, 1], True),
        ([0, 0, 0], True),
        ([1, 0, 0], True),
        ([1, 1, 0], True),
        ([1, 1, 0], True),
        ([0, 1, 0], True),
        ([1, 1, 0], True),
        ([1, 0, 1], True),
        ([1, 1, 1], True),
        ([1, 1, 0], True),
    ]
    expected = (
        (1, 0, 0),
        (2, 0, 0),
        (3, 1, 0),
        (4, 1, 1),
        (5, 1, 1),
        (6, 1, 1),
        (7, 2, 1),
        (8, 2, 1),
        (9, 2, 0),
        (10, 2, 0),
    )
    return source, stimuli(["clock_a", "clock_b", "reset"], ["a", "b"], rows), expected


def pipelined_read():
    source, _, _ = separate_memory()
    # This creates a distinct original control. It never rewrites selected RTL.
    source = source.replace(b"read-latency => 0", b"read-latency => 2")
    rows = []
    for addr, value in enumerate((17, 34, 51, 68)):
        rows.extend((([0, addr, 0, 1, 1, value], False), ([1, addr, 0, 1, 1, value], False)))
    rows.extend(
        (
            ([0, 0, 1, 0, 1, 0], False),
            ([1, 0, 1, 0, 1, 0], False),
            ([0, 3, 1, 0, 1, 0], False),
            ([1, 3, 1, 0, 1, 0], True),
            ([0, 2, 1, 0, 1, 0], True),
            ([1, 2, 1, 0, 1, 0], True),
            ([0, 1, 1, 0, 1, 0], True),
            ([1, 1, 1, 0, 1, 0], True),
            ([0, 0, 0, 0, 1, 0], True),
            ([1, 0, 0, 0, 1, 0], True),
            ([0, 0, 0, 0, 1, 0], True),
            ([1, 0, 0, 0, 1, 0], False),
            ([0, 0, 1, 0, 1, 0], False),
            ([1, 0, 1, 0, 1, 0], False),
            ([0, 2, 1, 0, 1, 0], False),
            ([1, 2, 1, 0, 1, 0], True),
            ([1, 3, 1, 0, 1, 0], True),
        )
    )
    expected = ((11, 17), (12, 17), (13, 68), (14, 68), (15, 51), (16, 51), (17, 34), (18, 34), (23, 17), (24, 17))
    return source, stimuli(["clock", "addr", "read_enable", "write_enable", "mask", "data"], ["result"], rows), expected
