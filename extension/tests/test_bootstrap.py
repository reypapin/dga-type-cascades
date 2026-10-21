import unittest

import numpy as np
import pandas as pd

from extension.lib.bootstrap import paired_f1_bootstrap
from extension.lib.bootstrap_multi import paired_f1_bootstrap_multi


def toy(seed: int = 0, n_fam: int = 6, per_fam: int = 40, n_neg: int = 200):
    rng = np.random.default_rng(seed)
    family = np.concatenate([np.repeat([f"f{i}" for i in range(n_fam)], per_fam), np.repeat("benign", n_neg)])
    y = np.concatenate([np.ones(n_fam * per_fam, dtype=int), np.zeros(n_neg, dtype=int)])
    preds = [pd.Series((rng.random(len(y)) < np.where(y == 1, 0.8, 0.1)).astype(int)) for _ in range(4)]
    return pd.Series(y), pd.Series(family), preds


def f1(y: pd.Series, p: pd.Series) -> float:
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn)


class MultiSeedBootstrapTest(unittest.TestCase):
    def test_single_vectors_match_the_paired_bootstrap(self):
        y, family, (a, b, _, _) = toy()
        self.assertEqual(paired_f1_bootstrap_multi(y, family, [a], [b], n_boot=200, seed=7),
                         paired_f1_bootstrap(y, family, a, b, n_boot=200, seed=7))

    def test_point_estimate_is_the_mean_of_the_seed_f1(self):
        y, family, (a1, a2, b, _) = toy()
        result = paired_f1_bootstrap_multi(y, family, [a1, a2], [b], n_boot=50, seed=7)
        self.assertAlmostEqual(result["diff"], (f1(y, a1) + f1(y, a2)) / 2 - f1(y, b), places=12)

    def test_a_repeated_seed_equals_one_seed(self):
        y, family, (a, b, _, _) = toy()
        self.assertEqual(paired_f1_bootstrap_multi(y, family, [a, a], [b], n_boot=200, seed=7),
                         paired_f1_bootstrap_multi(y, family, [a], [b], n_boot=200, seed=7))


if __name__ == "__main__":
    unittest.main()
