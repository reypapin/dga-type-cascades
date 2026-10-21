import unittest

import numpy as np
import pandas as pd

from extension.lib import diagnostics as dg
from extension.tests.test_cacic import synthetic_wide


def allfail_wide() -> pd.DataFrame:
    wide = synthetic_wide()
    # row with index 1 ("d.com"): a DGA domain that every expert misses
    for model in ("Logit", "ModernBERT", "DomURLs-BERT"):
        wide.loc[1, f"{model}__pred"] = 0
    wide["tier"] = "gate1"
    return wide


class TestAllFail(unittest.TestCase):
    def test_allfail_kept_in_evaluation(self):
        wide = allfail_wide()
        mask = dg.allfail_mask(wide)
        self.assertListEqual(list(mask[mask].index), [1])
        table = dg.family_table(wide, {"AlwaysLogit": wide["Logit__pred"].astype(int)})
        f1_row = table[(table.family == "f1") & (table.system == "AlwaysLogit")].iloc[0]
        self.assertEqual(int(f1_row["n"]), 2)
        self.assertAlmostEqual(float(f1_row["recall"]), 0.5)


class TestThresholds(unittest.TestCase):
    def test_threshold_ties(self):
        y = pd.Series([1, 0, 1, 0])
        score = pd.Series([0.9, 0.9, 0.9, 0.1])
        thr = dg.threshold_at_fpr(y, score, 0.5)
        self.assertLessEqual(dg.rate_at_threshold(y, score, thr)["fpr"], 0.5)
        self.assertAlmostEqual(dg.rate_at_threshold(y, score, thr)["recall"], 1.0)

    def test_threshold_never_splits_a_tie(self):
        y = pd.Series([1, 1, 0, 0])
        score = pd.Series([0.9, 0.9, 0.9, 0.1])
        thr = dg.threshold_at_fpr(y, score, 0.4)
        rate = dg.rate_at_threshold(y, score, thr)
        self.assertLessEqual(rate["fpr"], 0.4)
        self.assertEqual(rate["recall"], 0.0)


class TestRescueDamage(unittest.TestCase):
    def test_rescue_damage_counts(self):
        wide = pd.DataFrame(
            {
                "dga_type": ["NW", "NW", "NW"],
                "y_true": [1, 1, 1],
                "Logit": [0.45, 0.55, 0.55],
                "Logit__pred": [0, 1, 1],
                "DomURLs-BERT__pred": [1, 0, 1],
            }
        )
        table = dg.rescue_damage(wide, strong="DomURLs-BERT", bins=[0, 0.2, 1.0001])
        row = table[(table.dga_type == "NW") & (table.n > 0)].iloc[0]
        self.assertEqual(int(row["n"]), 3)
        self.assertEqual(int(row["rescue"]), 1)
        self.assertEqual(int(row["damage"]), 1)
        self.assertEqual(int(row["net"]), 0)


class TestRouterChoices(unittest.TestCase):
    def test_router_choice_partition(self):
        wide = allfail_wide()
        routes = pd.Series(["Logit", "DomURLs-BERT", "Logit", "ModernBERT"], index=wide.index)
        table = dg.router_choice_table(wide, routes).set_index("dga_type")
        for dga_type, count in wide["dga_type"].value_counts().items():
            row = table.loc[dga_type]
            self.assertEqual(int(row["wrong_choice"] + row["forced_wrong"] + row["ok"]), int(count))
        self.assertEqual(int(table.loc["NW", "forced_wrong"]), 1)
        self.assertEqual(int(table.loc["WL", "wrong_choice"]), 1)
        self.assertEqual(int(table.loc["benign", "wrong_choice"]), 1)


class TestCalibration(unittest.TestCase):
    def test_calibration_perfect(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 10_000)
        y = (rng.uniform(0, 1, 10_000) < p).astype(int)
        _, ece, brier = dg.calibration(pd.Series(y), pd.Series(p))
        self.assertLess(ece, 0.02)
        self.assertGreater(brier, 0.0)
        _, ece_bad, brier_bad = dg.calibration(pd.Series(np.zeros(100, dtype=int)), pd.Series(np.full(100, 0.9)))
        self.assertAlmostEqual(ece_bad, 0.9, delta=1e-9)
        self.assertAlmostEqual(brier_bad, 0.81, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
