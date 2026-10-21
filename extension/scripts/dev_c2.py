#!/usr/bin/env python3
"""C2: type cascade with the dictionary WL rule, chosen without the test split.

Routes: Logit when its confidence |p - 0.5| * 2 >= 0.80 (CACIC); otherwise
ModernBERT when coverage(main_label) >= tau, else DomURLs-BERT.
tau is chosen on router_train DGA rows (WL vs NW) by balanced accuracy;
C2 and its route-matched random control are then measured on gate1.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib import wordlist  # noqa: E402
from extension.lib.frozen import update_frozen  # noqa: E402

OUT = ROOT / "extension" / "results"
WORDS = ROOT / "extension" / "data" / "american-english.txt"
TAUS = tuple(round(x, 2) for x in np.arange(0.20, 0.951, 0.05))


def coverages(wide: pd.DataFrame, words: frozenset[str]) -> pd.Series:
    return pd.Series([wordlist.coverage(wordlist.main_label(d), words) for d in wide["domain"]], index=wide.index)


def choose_tau(train: pd.DataFrame, cov: pd.Series) -> tuple[float, pd.DataFrame]:
    dga = train["dga_type"].isin(["WL", "NW"])
    is_wl = train.loc[dga, "dga_type"].eq("WL")
    rows = []
    for tau in TAUS:
        says_wl = cov.loc[dga] >= tau
        tpr = float(says_wl[is_wl].mean())
        tnr = float((~says_wl[~is_wl]).mean())
        rows.append({"tau": tau, "wl_recall": tpr, "nw_specificity": tnr, "balanced_accuracy": (tpr + tnr) / 2})
    table = pd.DataFrame(rows)
    best = table.sort_values(["balanced_accuracy", "tau"], ascending=[False, True]).iloc[0]
    return float(best["tau"]), table


def type_cascade_routes(wide: pd.DataFrame, cov: pd.Series, tau: float, t: float = cacic.CASCADE_THRESHOLD) -> pd.Series:
    confident = (wide["Logit"] - 0.5).abs() * 2 >= t
    strong = np.where(cov >= tau, "ModernBERT", "DomURLs-BERT")
    return pd.Series(np.where(confident, "Logit", strong), index=wide.index, dtype=object)


def summarize(wide: pd.DataFrame, name: str, seed, pred: pd.Series, latency: pd.Series, routes: pd.Series) -> dict:
    metrics = cacic.binary_metrics(wide["y_true"], pred)
    row = {"system": name, "seed": seed, "f1": metrics["f1"], "fpr": metrics["fpr"], "recall": metrics["recall"]}
    row["latency_mean_ms"] = float(latency.mean())
    for group in ("WL", "NW"):
        row[f"{group.lower()}_recall"] = float(pred[wide["dga_type"].eq(group)].mean())
    for model, share in cacic.route_shares(routes).items():
        row[f"share_{model}"] = share
    return row


def main() -> None:
    words = wordlist.load_words(WORDS)
    train = cacic.load_split("router_train")
    tau, tau_table = choose_tau(train, coverages(train, words))
    tau_table.to_csv(OUT / "dev_c2_tau.csv", index=False)
    print(tau_table.round(4).to_string(index=False))
    print(f"chosen tau = {tau}")

    dev = cacic.load_split("gate1")
    cov = coverages(dev, words)
    rows = []
    systems = {
        "C2_type_cascade": type_cascade_routes(dev, cov, tau),
        "CascadeLogitDomURLs_t0.80": cacic.cascade_routes(dev, strong="DomURLs-BERT"),
        "CascadeLogitModernBERT_t0.80": cacic.cascade_routes(dev, strong="ModernBERT"),
    }
    for name, routes in systems.items():
        pred, latency, _ = cacic.apply_routes(dev, routes, probe_logit=True)
        rows.append(summarize(dev, name, "FIXED", pred, latency, routes))
    shares = cacic.route_shares(systems["C2_type_cascade"])
    for seed in cacic.SEEDS:
        routes = cacic.random_routes(dev, shares, seed)
        pred, latency, _ = cacic.apply_routes(dev, routes, probe_logit=False)
        rows.append(summarize(dev, "C2_random_control", seed, pred, latency, routes))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dev_c2.csv", index=False)
    view = table.groupby("system", sort=False).mean(numeric_only=True)
    print(view[["f1", "fpr", "latency_mean_ms", "wl_recall", "nw_recall"]].round(4).to_string())

    update_frozen(
        "c2",
        {
            "system": "type cascade with dictionary WL rule",
            "t": cacic.CASCADE_THRESHOLD,
            "tau": tau,
            "tau_grid": list(TAUS),
            "tau_rule": "max balanced accuracy WL vs NW on router_train DGA rows; ties -> lower tau",
            "label": "longest label excluding the TLD",
            "word_file": str(WORDS.relative_to(ROOT)),
            "word_file_sha256": hashlib.sha256(WORDS.read_bytes()).hexdigest(),
            "min_word_len": 3,
            "random_control_shares_dev": shares,
            "dev": {k: float(v) for k, v in view.loc["C2_type_cascade"].items()},
        },
    )


if __name__ == "__main__":
    main()
