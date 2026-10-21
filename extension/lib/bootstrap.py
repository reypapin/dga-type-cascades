"""Paired two-stage bootstrap for F1 differences: DGA families and benign domains are resampled."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def paired_f1_bootstrap(
    y: pd.Series, family: pd.Series, pred_a: pd.Series, pred_b: pd.Series, n_boot: int = 1000, seed: int = 7
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    fam = np.asarray(family).astype(str)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    pos, neg = y == 1, y == 0

    families, fam_code = np.unique(fam[pos], return_inverse=True)
    n_fam = len(families)
    tp_a = np.bincount(fam_code, weights=a[pos], minlength=n_fam)
    tp_b = np.bincount(fam_code, weights=b[pos], minlength=n_fam)
    size = np.bincount(fam_code, minlength=n_fam).astype(float)
    fp_a_rows, fp_b_rows = a[neg].astype(float), b[neg].astype(float)
    n_neg = len(fp_a_rows)

    def diff(tpa, tpb, sz, fpa, fpb):
        return float(_f1(tpa, fpa, sz - tpa) - _f1(tpb, fpb, sz - tpb))

    point = diff(tp_a.sum(), tp_b.sum(), size.sum(), fp_a_rows.sum(), fp_b_rows.sum())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        fam_counts = np.bincount(rng.integers(0, n_fam, n_fam), minlength=n_fam)
        neg_counts = np.bincount(rng.integers(0, n_neg, n_neg), minlength=n_neg) if n_neg else np.zeros(0)
        diffs[r] = diff(
            (fam_counts * tp_a).sum(),
            (fam_counts * tp_b).sum(),
            (fam_counts * size).sum(),
            (neg_counts * fp_a_rows).sum(),
            (neg_counts * fp_b_rows).sum(),
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "low": float(low), "high": float(high), "n_boot": n_boot, "n_families": n_fam}
