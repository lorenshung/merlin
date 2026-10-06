"""Compile the actual target hasher and compare canonical bytes to hashlib."""

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def hasher(tmp_path_factory):
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    work = tmp_path_factory.mktemp("sha256")
    source = work / "probe.c"
    source.write_text("""#include "output_sha256.h"
void hash_bytes(const unsigned char *p, size_t n, unsigned char *out) {
 merlin_sha256 s; merlin_sha256_init(&s);
 /* Exercise streaming across every block/padding boundary. */
 for(size_t i=0;i<n;i+=7) merlin_sha256_update(&s,p+i,n-i<7?n-i:7);
 merlin_sha256_final(&s,out);
}
void hash_floats(const float *p,size_t n,unsigned char *out) {
 merlin_output_sha256(p,n,out);
}
""")
    lib = work / "probe.so"
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-Wall",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I",
            str(root / "merlin/runtime/baremetal/spike"),
            str(source),
            "-o",
            str(lib),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(lib))
    for name in ("hash_bytes", "hash_floats"):
        getattr(dll, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return dll


@pytest.mark.parametrize(
    "data",
    [b"", b"abc", b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", b"a" * 1_000_000]
    + [bytes(range(n)) for n in (55, 56, 63, 64, 65, 119, 120, 127, 128)],
)
def test_known_vectors_and_padding(hasher, data):
    output = ctypes.create_string_buffer(32)
    hasher.hash_bytes(data, len(data), output)
    assert output.raw.hex() == hashlib.sha256(data).hexdigest()


def test_f32_encoding_preserves_every_bit(hasher):
    # Signed zero, infinities, NaN payloads, subnormal and ordinary finite bits.
    words = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 1, 0x3F800000]
    floats = (ctypes.c_uint32 * len(words))(*words)
    output = ctypes.create_string_buffer(32)
    hasher.hash_floats(floats, len(words), output)
    canonical = b"".join(w.to_bytes(4, "little") for w in words)
    assert len(canonical) == 4 * len(words)
    assert output.raw.hex() == hashlib.sha256(canonical).hexdigest()


def test_record_validation():
    import numpy as np

    from merlin.runtime.output_digest import verify_output_sha256

    reference = np.array([[1, -0.0, 3]], dtype="<f4")
    digest = hashlib.sha256(reference.tobytes()).hexdigest()
    record = f"OUT_SHA256 f32le 3 12 {digest}\n"
    assert verify_output_sha256(record, reference)["bytes"] == 12
    assert verify_output_sha256(record, reference.astype(">f4"))["sha256"] == digest
    for bad in (
        "",
        record + record,
        record.replace("3 12", "2 12"),
        record.replace("3 12", "3 8"),
        record.replace(digest, "0" * 64),
    ):
        with pytest.raises(ValueError):
            verify_output_sha256(bad, reference)
    with pytest.raises(ValueError):
        verify_output_sha256(record, reference.astype("float64"))
