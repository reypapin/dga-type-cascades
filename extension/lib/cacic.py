"""CACIC systems rebuilt from the saved per-domain predictions.

The CACIC functions live in scripts/train_final_router.py and are imported,
never edited, so every system here matches the published paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_final_router import (  # noqa: E402
    MODELS,
    SEEDS,
    binary_metrics,
    cheapest_correct_targets,
    logit_aware_features,
    make_router,
    pivot_predictions,
    route_predictions,
    sample_weights,
)

PREDICTIONS = ROOT / "experiments" / "predictions"
FINAL_DOMAINS = ROOT / "data" / "final_test_domains.csv"
SPLITS = ("router_train", "gate1", "final_test")
CASCADE_THRESHOLD = 0.80

__all__ = [
    "MODELS",
    "SEEDS",
    "binary_metrics",
    "load_split",
    "train_router",
    "router_routes",
    "cascade_routes",
    "random_routes",
    "apply_routes",
    "cacic_systems",
]

_WIDE: dict[str, pd.DataFrame] = {}
_FEATURES: dict[str, list[dict]] = {}


def load_split(split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split not in _WIDE:
        wide = pivot_predictions(pd.read_csv(PREDICTIONS / f"{split}_uniform_predictions.csv"))
        if split == "final_test":
            meta = pd.read_csv(FINAL_DOMAINS, usecols=["domain", "tier", "base_family"]).fillna("")
            wide = wide.merge(meta, on="domain", how="left")
            wide["tier"] = wide["tier"].fillna("")
            wide["base_family"] = wide["base_family"].fillna("")
        else:
            wide["tier"] = split
            wide["base_family"] = ""
        wide = wide.reset_index(drop=True)
        wide.attrs["split"] = split
        _WIDE[split] = wide
    return _WIDE[split].copy()


def _features(wide: pd.DataFrame) -> list[dict]:
    """Router features; the per-split cache is used only when rows are in the cached order."""
    split = wide.attrs.get("split")
    cached = _WIDE.get(split)
    if cached is None or not np.array_equal(wide["domain"].to_numpy(), cached["domain"].to_numpy()):
        return logit_aware_features(wide)
    if split not in _FEATURES:
        _FEATURES[split] = logit_aware_features(cached)
    return _FEATURES[split]


def train_router(seed: int, fp_weight: float = 0.0) -> Pipeline:
    train = load_split("router_train")
    target, resolved = cheapest_correct_targets(train)
    features = _features(train)
    keep = np.flatnonzero(resolved.to_numpy())
    weights = sample_weights(train)
    if fp_weight:
        any_fp = pd.concat([train[f"{m}__pred"].astype(int).eq(1) for m in MODELS], axis=1).any(axis=1)
        risky_benign = train["y_true"].astype(int).eq(0) & any_fp
        weights = weights * np.where(risky_benign, 1.0 + fp_weight, 1.0)
    model = make_router(seed)
    model.fit(
        [features[i] for i in keep],
        target.iloc[keep],
        model__sample_weight=weights.iloc[keep].to_numpy(),
    )
    return model


def router_routes(model: Pipeline, wide: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(_features(wide)), index=wide.index, dtype=object)


def cascade_routes(wide: pd.DataFrame, t: float = CASCADE_THRESHOLD, strong: str = "DomURLs-BERT") -> pd.Series:
    confident = (wide["Logit"] - 0.5).abs() * 2 >= t
    return pd.Series(np.where(confident, "Logit", strong), index=wide.index, dtype=object)


def random_routes(wide: pd.DataFrame, shares: dict[str, float], seed: int) -> pd.Series:
    p = np.array([shares.get(m, 0.0) for m in MODELS], dtype=float)
    chosen = np.random.default_rng(seed).choice(MODELS, size=len(wide), p=p / p.sum())
    return pd.Series(chosen, index=wide.index, dtype=object)


def apply_routes(wide: pd.DataFrame, routes: pd.Series, probe_logit: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
    unknown = set(routes.unique()) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown expert names in routes: {sorted(map(str, unknown))}")
    pred, latency = route_predictions(wide, routes, base_logit_cost=probe_logit)
    score = pd.Series(np.nan, index=wide.index, dtype=float)
    for model in MODELS:
        idx = routes.index[routes.eq(model)]
        score.loc[idx] = wide.loc[idx, model].astype(float)
    return pred, latency, score


def route_shares(routes: pd.Series) -> dict[str, float]:
    return {m: float(routes.eq(m).mean()) for m in MODELS}


def cacic_systems(wide: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series, pd.Series | None]]:
    systems: dict[str, tuple[pd.Series, pd.Series, pd.Series | None]] = {}
    for model in MODELS:
        routes = pd.Series(model, index=wide.index, dtype=object)
        pred, latency, _ = apply_routes(wide, routes, probe_logit=False)
        systems[f"Always{model}"] = (pred, latency, routes)
    ensemble_pred = (wide[MODELS].mean(axis=1) >= 0.5).astype(int)
    ensemble_latency = wide[[f"{m}__latency" for m in MODELS]].sum(axis=1)
    systems["EnsembleMean"] = (ensemble_pred, ensemble_latency, None)
    cascade = cascade_routes(wide)
    pred, latency, _ = apply_routes(wide, cascade, probe_logit=True)
    systems[f"CascadeLogitDomURLs_t{CASCADE_THRESHOLD:.2f}"] = (pred, latency, cascade)
    for seed in SEEDS:
        routes = router_routes(train_router(seed), wide)
        pred, latency, _ = apply_routes(wide, routes, probe_logit=True)
        systems[f"Router_s{seed}"] = (pred, latency, routes)
    shares = route_shares(cascade)
    for seed in SEEDS:
        routes = random_routes(wide, shares, seed)
        pred, latency, _ = apply_routes(wide, routes, probe_logit=False)
        systems[f"Random_s{seed}"] = (pred, latency, routes)
    return systems
