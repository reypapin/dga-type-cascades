import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from extension.lib import cacic

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"


def synthetic_wide() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "domain": ["a.com", "b.com", "c.com", "d.com"],
            "family": ["f1", "f2", "benign", "f1"],
            "dga_type": ["NW", "WL", "benign", "NW"],
            "split": ["s"] * 4,
            "y_true": [1, 1, 0, 1],
            "Logit": [0.9, 0.2, 0.6, 0.95],
            "ModernBERT": [0.8, 0.9, 0.1, 0.7],
            "DomURLs-BERT": [0.7, 0.3, 0.2, 0.6],
            "Logit__pred": [1, 0, 1, 1],
            "ModernBERT__pred": [1, 1, 0, 1],
            "DomURLs-BERT__pred": [1, 0, 0, 1],
            "Logit__latency": [0.06, 0.06, 0.06, 0.06],
            "ModernBERT__latency": [1.9, 1.9, 1.9, 1.9],
            "DomURLs-BERT__latency": [1.6, 1.6, 1.6, 1.6],
        }
    )
    return df.set_index(pd.Index([3, 0, 2, 1]))


class TestApplyRoutes(unittest.TestCase):
    def test_apply_routes_alignment(self):
        wide = synthetic_wide()
        routes = pd.Series(["Logit", "ModernBERT", "DomURLs-BERT", "Logit"], index=wide.index)
        pred, latency, score = cacic.apply_routes(wide, routes, probe_logit=True)
        self.assertListEqual(list(pred.index), list(wide.index))
        for idx in wide.index:
            chosen = routes.loc[idx]
            self.assertEqual(pred.loc[idx], wide.loc[idx, f"{chosen}__pred"])
            self.assertAlmostEqual(score.loc[idx], wide.loc[idx, chosen])
            expected = wide.loc[idx, f"{chosen}__latency"]
            if chosen != "Logit":
                expected += wide.loc[idx, "Logit__latency"]
            self.assertAlmostEqual(latency.loc[idx], expected)


class TestGuards(unittest.TestCase):
    def test_unknown_route_name_raises(self):
        wide = synthetic_wide()
        routes = pd.Series(["Logit", "DomURLs_BERT", "Logit", "Logit"], index=wide.index)
        with self.assertRaises(ValueError):
            cacic.apply_routes(wide, routes, probe_logit=True)

    def test_feature_cache_follows_row_order(self):
        wide = cacic.load_split("gate1")
        shuffled = wide.sample(frac=1.0, random_state=0)
        self.assertEqual(shuffled.attrs.get("split"), "gate1")
        expected = cacic.logit_aware_features(shuffled.head(5))
        self.assertEqual(cacic._features(shuffled)[:5], expected)


class TestReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wide = cacic.load_split("final_test")
        cls.systems = cacic.cacic_systems(cls.wide)

    def f1(self, pred: pd.Series) -> float:
        return cacic.binary_metrics(self.wide["y_true"], pred)["f1"]

    def test_reproduces_cacic_final_test(self):
        router_f1 = np.mean([self.f1(self.systems[f"Router_s{s}"][0]) for s in cacic.SEEDS])
        self.assertAlmostEqual(router_f1, 0.9174707335503044, delta=1e-6)
        cascade_f1 = self.f1(self.systems["CascadeLogitDomURLs_t0.80"][0])
        self.assertAlmostEqual(cascade_f1, 0.8703434114347887, delta=1e-6)
        ref = pd.read_csv(RESULTS / "final_test_system_metrics.csv", dtype={"seed": str})
        row = ref[(ref.system == "RandomCostMatched") & (ref.seed == "MEAN") & (ref.group == "ALL")]
        random_f1 = np.mean([self.f1(self.systems[f"Random_s{s}"][0]) for s in cacic.SEEDS])
        self.assertAlmostEqual(random_f1, float(row["f1"].iloc[0]), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
