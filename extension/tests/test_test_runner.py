import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from extension.lib import bootstrap, freeze


class TestFreeze(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.a = self.dir / "a.md"
        self.b = self.dir / "b.json"
        self.a.write_text("prereg", encoding="utf-8")
        self.b.write_text("{}", encoding="utf-8")
        self.sums = self.dir / "FREEZE.sha256"
        freeze.write_freeze(self.sums, [self.a, self.b])

    def test_freeze_passes_when_unchanged(self):
        freeze.check_freeze(self.sums)

    def test_freeze_detects_change(self):
        self.b.write_text('{"changed": true}', encoding="utf-8")
        with self.assertRaises(SystemExit):
            freeze.check_freeze(self.sums)

    def test_guard_refuses_existing_summary(self):
        summary = self.dir / "summary.md"
        freeze.guard_single_run(summary)
        summary.write_text("done", encoding="utf-8")
        with self.assertRaises(SystemExit):
            freeze.guard_single_run(summary)


class TestBootstrap(unittest.TestCase):
    def test_paired_bootstrap_detects_a_real_gap(self):
        families = np.repeat([f"f{i}" for i in range(10)], 20)
        y = np.r_[np.ones(200, dtype=int), np.zeros(200, dtype=int)]
        family = np.r_[families, np.array(["benign"] * 200)]
        pred_a = y.copy()
        pred_b = y.copy()
        pred_b[:200:2] = 0  # B misses half of the DGA domains
        out = bootstrap.paired_f1_bootstrap(pd.Series(y), pd.Series(family), pd.Series(pred_a), pd.Series(pred_b), n_boot=300, seed=7)
        self.assertGreater(out["diff"], 0)
        self.assertGreater(out["low"], 0)
        self.assertLessEqual(out["low"], out["diff"])
        self.assertGreaterEqual(out["high"], out["diff"])

    def test_paired_bootstrap_is_zero_for_identical_systems(self):
        y = np.r_[np.ones(50, dtype=int), np.zeros(50, dtype=int)]
        family = np.r_[np.repeat(["f1", "f2"], 25), np.array(["benign"] * 50)]
        pred = y.copy()
        pred[:5] = 0
        out = bootstrap.paired_f1_bootstrap(pd.Series(y), pd.Series(family), pd.Series(pred), pd.Series(pred), n_boot=100, seed=7)
        self.assertEqual(out["diff"], 0.0)
        self.assertEqual(out["low"], 0.0)
        self.assertEqual(out["high"], 0.0)


if __name__ == "__main__":
    unittest.main()
