"""Paired two-stage bootstrap where each side may hold several prediction vectors (descriptive, after the test run).

Kept apart from bootstrap.py, which is frozen in extension/protocol/FREEZE.sha256.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from extension.lib.bootstrap import _f1


def paired_f1_bootstrap_multi(
    y: pd.Series, family: pd.Series, preds_a: list[pd.Series], preds_b: list[pd.Series], n_boot: int = 1000, seed: int = 7
) -> dict[str, float]:
    """Same bootstrap, but each side may hold several prediction vectors (for example the 5 router seeds).

    The F1 of a side is the mean F1 of its vectors, in the point estimate and in every replicate. With one
    vector per side it gives exactly the result of paired_f1_bootstrap (same resampling draws).
    """
    y = np.asarray(y, dtype=int)
    fam = np.asarray(family).astype(str)
    pos, neg = y == 1, y == 0
    families, fam_code = np.unique(fam[pos], return_inverse=True)
    n_fam = len(families)
    size = np.bincount(fam_code, minlength=n_fam).astype(float)

    def counts(preds):
        rows = [np.asarray(p, dtype=int) for p in preds]
        tp = np.stack([np.bincount(fam_code, weights=r[pos], minlength=n_fam) for r in rows])
        fp = np.stack([r[neg].astype(float) for r in rows])
        return tp, fp

    tp_a, fp_a = counts(preds_a)
    tp_b, fp_b = counts(preds_b)
    n_neg = fp_a.shape[1]

    def mean_f1(tp, fp, fam_w, neg_w):
        t, f, s = tp @ fam_w, fp @ neg_w, size @ fam_w
        return float(np.mean(_f1(t, f, s - t)))

    point = mean_f1(tp_a, fp_a, np.ones(n_fam), np.ones(n_neg)) - mean_f1(tp_b, fp_b, np.ones(n_fam), np.ones(n_neg))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        fam_counts = np.bincount(rng.integers(0, n_fam, n_fam), minlength=n_fam).astype(float)
        neg_counts = np.bincount(rng.integers(0, n_neg, n_neg), minlength=n_neg).astype(float) if n_neg else np.zeros(0)
        diffs[r] = mean_f1(tp_a, fp_a, fam_counts, neg_counts) - mean_f1(tp_b, fp_b, fam_counts, neg_counts)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "low": float(low), "high": float(high), "n_boot": n_boot, "n_families": n_fam}
