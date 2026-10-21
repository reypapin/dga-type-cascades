"""Error diagnosis of the CACIC systems (D1-D6) and fixed-FPR thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from extension.lib.cacic import MODELS
from train_final_router import domain_features

PROFILE_FEATURES = ("length", "entropy", "digit_ratio", "vowel_ratio")


def correct_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    y = wide["y_true"].astype(int)
    return pd.DataFrame({m: wide[f"{m}__pred"].astype(int).eq(y) for m in MODELS}, index=wide.index)


def allfail_mask(wide: pd.DataFrame) -> pd.Series:
    return ~correct_matrix(wide).any(axis=1)


def family_table(wide: pd.DataFrame, preds: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for family, fam in wide.groupby("family", sort=True):
        positive = fam["y_true"].astype(int).eq(1)
        for system, pred in preds.items():
            p = pred.loc[fam.index].astype(int)
            rows.append(
                {
                    "family": family,
                    "dga_type": fam["dga_type"].iloc[0],
                    "tier": fam["tier"].iloc[0],
                    "system": system,
                    "n": int(len(fam)),
                    "recall": float(p[positive].mean()) if positive.any() else np.nan,
                    "fpr": float(p[~positive].mean()) if (~positive).any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def allfail_profile(wide: pd.DataFrame) -> pd.DataFrame:
    fail = allfail_mask(wide)
    rows = []
    for section, key in (("by_type", "dga_type"), ("by_family", "family")):
        for value, idx in wide.groupby(key, sort=True).groups.items():
            rows.append(
                {
                    "section": section,
                    "key": value,
                    "n": int(len(idx)),
                    "n_allfail": int(fail.loc[idx].sum()),
                    "share": float(fail.loc[idx].mean()),
                }
            )
    feats = pd.DataFrame([domain_features(d) for d in wide["domain"]], index=wide.index)
    for dga_type, idx in wide.groupby("dga_type", sort=True).groups.items():
        for label, part in (("allfail", fail.loc[idx]), ("rest", ~fail.loc[idx])):
            sel = part[part].index
            row = {"section": "features", "key": f"{dga_type}:{label}", "n": int(len(sel))}
            for col in PROFILE_FEATURES:
                row[col] = float(feats.loc[sel, col].mean()) if len(sel) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def router_choice_table(wide: pd.DataFrame, routes: pd.Series) -> pd.DataFrame:
    """Per type: chosen expert right (ok), wrong while another was right, or every expert wrong."""
    correct = correct_matrix(wide)
    chosen_ok = pd.Series(False, index=wide.index)
    for model in MODELS:
        idx = routes.index[routes.eq(model)]
        chosen_ok.loc[idx] = correct.loc[idx, model]
    any_ok = correct.any(axis=1)
    frame = pd.DataFrame(
        {
            "dga_type": wide["dga_type"],
            "ok": chosen_ok.astype(int),
            "wrong_choice": (~chosen_ok & any_ok).astype(int),
            "forced_wrong": (~any_ok).astype(int),
        }
    )
    table = frame.groupby("dga_type").agg(
        n=("ok", "size"), ok=("ok", "sum"), wrong_choice=("wrong_choice", "sum"), forced_wrong=("forced_wrong", "sum")
    )
    return table.reset_index()


def benign_route_fpr(wide: pd.DataFrame, routes: pd.Series) -> pd.DataFrame:
    """Per route: FPR of the chosen expert on the benign rows sent to it vs on all benign rows."""
    benign = wide["y_true"].astype(int).eq(0)
    rows = []
    for model in MODELS:
        flagged = wide[f"{model}__pred"].astype(int).eq(1)
        sent = benign & routes.eq(model)
        rows.append(
            {
                "route": model,
                "n_benign": int(sent.sum()),
                "share_benign": float(sent.sum() / benign.sum()),
                "fpr_routed": float(flagged[sent].mean()) if sent.any() else np.nan,
                "fpr_expert_all": float(flagged[benign].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration(y: pd.Series, p: pd.Series, n_bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    """Equal-width reliability bins, expected calibration error (weighted by bin size) and Brier score."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        sel = which == b
        n = int(sel.sum())
        mean_p = float(p[sel].mean()) if n else np.nan
        frac_pos = float(y[sel].mean()) if n else np.nan
        if n:
            ece += n / len(p) * abs(mean_p - frac_pos)
        rows.append({"bin": b, "low": edges[b], "high": edges[b + 1], "n": n, "mean_p": mean_p, "frac_pos": frac_pos})
    brier = float(np.mean((p - y) ** 2))
    return pd.DataFrame(rows), float(ece), brier


CONFIDENCE_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def rescue_damage(wide: pd.DataFrame, strong: str, bins: list[float] = CONFIDENCE_BINS) -> pd.DataFrame:
    """Per Logit-confidence bin and type: how often escalating to `strong` fixes or breaks a decision."""
    y = wide["y_true"].astype(int)
    logit_ok = wide["Logit__pred"].astype(int).eq(y)
    strong_ok = wide[f"{strong}__pred"].astype(int).eq(y)
    conf = (wide["Logit"] - 0.5).abs() * 2
    frame = pd.DataFrame(
        {
            "dga_type": wide["dga_type"],
            "bin": pd.cut(conf, bins=bins, right=False),
            "rescue": (~logit_ok & strong_ok).astype(int),
            "damage": (logit_ok & ~strong_ok).astype(int),
        }
    )
    table = frame.groupby(["dga_type", "bin"], observed=False).agg(
        n=("rescue", "size"), rescue=("rescue", "sum"), damage=("damage", "sum")
    )
    table = table.reset_index()
    table["net"] = table["rescue"] - table["damage"]
    table["bin_low"] = table["bin"].map(lambda b: b.left).astype(float)
    table["bin_high"] = table["bin"].map(lambda b: b.right).astype(float)
    return table.drop(columns="bin").assign(strong=strong)


def threshold_at_fpr(y: pd.Series, score: pd.Series, target_fpr: float) -> float:
    """Lowest threshold (highest recall) whose FPR with `score >= threshold` stays <= target."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    negatives = max(int((y == 0).sum()), 1)
    order = np.argsort(-s, kind="stable")
    s_sorted, neg_sorted = s[order], (y[order] == 0)
    cum_fp = np.cumsum(neg_sorted)
    # last position of each group of tied scores
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    fpr_at = cum_fp[last_of_tie] / negatives
    candidates = s_sorted[last_of_tie][fpr_at <= target_fpr]
    return float(candidates.min()) if len(candidates) else float("inf")


def rate_at_threshold(y: pd.Series, score: pd.Series, thr: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(score, dtype=float) >= thr
    pos, neg = y == 1, y == 0
    return {
        "recall": float(flagged[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(flagged[neg].mean()) if neg.any() else float("nan"),
    }
