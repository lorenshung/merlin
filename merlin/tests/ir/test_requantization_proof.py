import unittest

import numpy as np

from merlin.llvmlower.requantization import prove_scale, quantized, synthesize_bias


class TestRequantizationProof(unittest.TestCase):
    def test_full_i32_domain_excludes_overflowing_integer_bias(self):
        proof = synthesize_bias([1.0], [0.0, 1.0, -1.0], 1.0, -(1 << 31), (1 << 31) - 1)
        self.assertEqual(proof["integer_bias"], [0, None, None])
        self.assertEqual(proof["accepted_channels"], 1)
        self.assertEqual([x["channel"] for x in proof["refused_channels"]], [1, 2])

    def test_transition_proof_matches_exhaustive_domain(self):
        for scales in [[0.5, 0.0625], [0.7, 0.7], [0.123456, 0.234567]]:
            proof = prove_scale(scales, -10000, 10000)
            self.assertTrue(all(quantized(a, scales) == quantized(a, [proof["scale"]]) for a in range(-10000, 10001)))

    def test_float_reassociation_refused_with_concrete_witness(self):
        with self.assertRaisesRegex(ValueError, "-12650"):
            prove_scale([0.1, 0.1], -100000, 100000)

    def test_nonintegral_bias_cannot_be_silently_rounded(self):
        proof = synthesize_bias([0.5, 0.0625], [0, 1, 0.01], 1, -10000, 10000)
        self.assertEqual(proof["integer_bias"], [0, 32, None])
        for j, bias in enumerate([0, 1]):
            for acc in range(-10000, 10001):
                source = int(
                    np.clip(np.rint(np.float32(np.float32(acc) * np.float32(0.03125)) + np.float32(bias)), -128, 127)
                )
                self.assertEqual(source, quantized(acc + proof["integer_bias"][j], [proof["scale"]]))
