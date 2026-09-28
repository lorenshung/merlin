"""The memory-regime dtype is reproducible when required-cell counts tie."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from merlin.targetgen import conformance as CF


def test_regime_dtype_tie_is_stable_across_python_hash_seeds():
    # This is the exact six/six cell-count tie seen in the Atlas derivation. The old
    # `max(set(dtypes), key=dtypes.count)` selected bf16 or fp8 by hash seed.
    script = """
import json
from merlin.targetgen.conformance import Cell, regime_dtype_selection
cells = [Cell('contraction', 'bf16', None) for _ in range(6)] + [
    Cell('contraction', 'fp8_e4m3', None) for _ in range(6)
]
print(json.dumps(regime_dtype_selection(cells), sort_keys=True))
"""
    results = []
    for seed in ("0", "1", "2", "4", "17"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        run = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)
        results.append(json.loads(run.stdout))
    assert all(result == results[0] for result in results)
    selected, evidence = results[0]
    assert selected == "fp8_e4m3"
    assert evidence["counts"] == {"bf16": 6, "fp8_e4m3": 6}
    assert evidence["storage_bits"] == {"bf16": 16, "fp8_e4m3": 8}
    assert evidence["tied_by_count"] == ["bf16", "fp8_e4m3"]


def test_regime_dtype_frequency_precedes_width_and_unknown_width_is_last():
    choose = CF.regime_dtype_selection
    cells = [CF.Cell("contraction", "bf16", None)] * 2 + [CF.Cell("contraction", "fp8_e4m3", None)]
    assert choose(cells)[0] == "bf16"
    tied = [CF.Cell("contraction", dtype, None) for dtype in ("not_a_dtype", "bf16")]
    assert choose(tied)[0] == "bf16"
    assert choose([])[0] is None


def test_host_only_dtype_map_is_stable_across_python_hash_seeds():
    # The full Atlas r11 cross-seed derivations differed only in these two map
    # keys' order. Exercise the real host_only_dtypes call with captured regions
    # substituted, so this stays fast and independent of capture-file parsing.
    script = """
import json
from merlin.targetgen import conformance as cf, model_coverage as mc
class Region:
    in_dtype = 'f32'
    def __init__(self, family):
        self.family = family
    def resolved_family(self):
        return self.family
mc.load_module = lambda path: None
mc.regions_from_module = lambda module: [Region('normalization'), Region('reduction')]
result = cf.host_only_dtypes({'capture': 'unused'}, ['normalization', 'reduction'])
print(json.dumps(list(result.items())))
"""
    results = []
    for seed in ("0", "1", "2", "4", "17"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        run = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)
        results.append(json.loads(run.stdout))
    assert all(result == results[0] for result in results)
    assert results[0] == [["normalization", "f32"], ["reduction", "f32"]]
