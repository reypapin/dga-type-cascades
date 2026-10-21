import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from extension.lib import decision as dm
from extension.tests.test_cacic import synthetic_wide


def dm_frame(wide: pd.DataFrame, domains: list[str], p_wl: list[float], p_dga: list[float]) -> pd.DataFrame:
    rows = wide.set_index("domain").loc[domains].reset_index()
    return pd.DataFrame(
        {
            "domain": rows["domain"],
            "family": rows["family"],
            "dga_type": rows["dga_type"],
            "split": rows["split"],
            "y_true": rows["y_true"],
            "model": "laya",
            "p_wl": p_wl,
            "p_dga": p_dga,
            "latency_ms": [30.0] * len(domains),
        }
    )


class TestDecisionModel(unittest.TestCase):
    def setUp(self):
        self.wide = synthetic_wide()
        # Logit confidence |p - 0.5| * 2: a.com 0.8, b.com 0.6, c.com 0.2, d.com 0.9 -> doubtful: b.com, c.com

    def write(self, frame: pd.DataFrame) -> Path:
        path = Path(tempfile.mkdtemp()) / "dm.csv"
        frame.to_csv(path, index=False)
        return path

    def test_load_dm_aligns_by_domain(self):
        path = self.write(dm_frame(self.wide, ["c.com", "b.com"], [0.1, 0.9], [0.2, 0.8]))
        table = dm.load_dm(path, self.wide)
        self.assertListEqual(list(table.index), list(self.wide.index))
        self.assertAlmostEqual(table.loc[self.wide.index[self.wide.domain.eq("b.com")][0], "p_wl"], 0.9)
        self.assertTrue(np.isnan(table.loc[self.wide.index[self.wide.domain.eq("a.com")][0], "p_wl"]))

    def test_load_dm_raises_on_missing_doubtful_domain(self):
        path = self.write(dm_frame(self.wide, ["b.com"], [0.9], [0.8]))
        with self.assertRaises(ValueError):
            dm.load_dm(path, self.wide)

    def test_c3_routes(self):
        path = self.write(dm_frame(self.wide, ["c.com", "b.com"], [0.1, 0.9], [0.2, 0.8]))
        table = dm.load_dm(path, self.wide)
        routes = dm.c3_routes(self.wide, table["p_wl"], tau=0.5)
        by_domain = dict(zip(self.wide["domain"], routes))
        self.assertEqual(by_domain["a.com"], "Logit")
        self.assertEqual(by_domain["d.com"], "Logit")
        self.assertEqual(by_domain["b.com"], "ModernBERT")
        self.assertEqual(by_domain["c.com"], "DomURLs-BERT")

    def test_c4_uses_decision_model_only_on_doubtful(self):
        path = self.write(dm_frame(self.wide, ["c.com", "b.com"], [0.1, 0.9], [0.2, 0.8]))
        table = dm.load_dm(path, self.wide)
        pred, latency = dm.c4_pred(self.wide, table, theta=0.5)
        by_domain = dict(zip(self.wide["domain"], pred))
        self.assertEqual(by_domain["a.com"], 1)  # Logit decides (confident)
        self.assertEqual(by_domain["b.com"], 1)  # judge p_dga 0.8
        self.assertEqual(by_domain["c.com"], 0)  # judge p_dga 0.2
        lat = dict(zip(self.wide["domain"], latency))
        self.assertAlmostEqual(lat["a.com"], 0.06)
        self.assertAlmostEqual(lat["b.com"], 30.06)


if __name__ == "__main__":
    unittest.main()
