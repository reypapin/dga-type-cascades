"""Decision-model configurations: C3 (type cascade with "¿es WL?") and C4 (doubtful-case judge)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from extension.lib.cacic import CASCADE_THRESHOLD, apply_routes


def doubtful_mask(wide: pd.DataFrame, t: float = CASCADE_THRESHOLD) -> pd.Series:
    return (wide["Logit"] - 0.5).abs() * 2 < t


def load_dm(path: Path, wide: pd.DataFrame, require_doubtful: bool = True, t: float = CASCADE_THRESHOLD) -> pd.DataFrame:
    """Decision-model outputs aligned to `wide` by domain (NaN where the model was not run)."""
    frame = pd.read_csv(path)
    frame["domain"] = frame["domain"].astype(str).str.strip().str.lower()
    if frame["domain"].duplicated().any():
        raise ValueError(f"{path}: duplicated domains")
    unknown = set(frame["domain"]) - set(wide["domain"])
    if unknown:
        raise ValueError(f"{path}: {len(unknown)} domains are not in this split")
    merged = wide[["domain"]].merge(frame[["domain", "p_wl", "p_dga", "latency_ms"]], on="domain", how="left")
    merged.index = wide.index
    if require_doubtful:
        missing = doubtful_mask(wide, t) & merged["p_wl"].isna()
        if missing.any():
            raise ValueError(f"{path}: {int(missing.sum())} doubtful domains have no decision-model output")
    return merged[["p_wl", "p_dga"]].assign(dm_latency_ms=merged["latency_ms"])


def c3_routes(wide: pd.DataFrame, p_wl: pd.Series, tau: float, t: float = CASCADE_THRESHOLD) -> pd.Series:
    confident = ~doubtful_mask(wide, t)
    strong = np.where(p_wl.to_numpy() >= tau, "ModernBERT", "DomURLs-BERT")
    return pd.Series(np.where(confident, "Logit", strong), index=wide.index, dtype=object)


def c3_apply(wide: pd.DataFrame, dm_table: pd.DataFrame, tau: float, t: float = CASCADE_THRESHOLD):
    routes = c3_routes(wide, dm_table["p_wl"], tau, t)
    pred, latency, score = apply_routes(wide, routes, probe_logit=True)
    doubtful = doubtful_mask(wide, t)
    latency = latency + np.where(doubtful, dm_table["dm_latency_ms"].fillna(0.0), 0.0)
    return pred, latency, routes


def c4_pred(wide: pd.DataFrame, dm_table: pd.DataFrame, theta: float, t: float = CASCADE_THRESHOLD):
    doubtful = doubtful_mask(wide, t)
    judge = (dm_table["p_dga"] >= theta).astype(int)
    pred = pd.Series(np.where(doubtful, judge, wide["Logit__pred"].astype(int)), index=wide.index, dtype=int)
    latency = wide["Logit__latency"].astype(float) + np.where(doubtful, dm_table["dm_latency_ms"].fillna(0.0), 0.0)
    return pred, pd.Series(latency, index=wide.index, dtype=float)
