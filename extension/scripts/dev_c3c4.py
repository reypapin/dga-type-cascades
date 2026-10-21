#!/usr/bin/env python3
"""C3 (type cascade with a decision model) and C4 (doubtful-case judge), chosen without the test split.

Inputs (from the Colab runs, in experiments/predictions/):
- router_train_dm_sample_<model>_decision_predictions.csv  -> tau_dm for C3 (balanced accuracy WL vs NW)
- gate1_doubtful_<model>_decision_predictions.csv          -> C3 and C4 on development data
C4 judge threshold theta: maximum development F1 of the whole system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib import decision as dm  # noqa: E402
from extension.lib.frozen import FROZEN, update_frozen  # noqa: E402

PRED = ROOT / "experiments" / "predictions"
OUT = ROOT / "extension" / "results"
GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))


def metrics_row(wide, name, seed, pred, latency, routes=None) -> dict:
    m = cacic.binary_metrics(wide["y_true"], pred)
    row = {"system": name, "seed": seed, "f1": m["f1"], "fpr": m["fpr"], "recall": m["recall"], "latency_mean_ms": float(latency.mean())}
    for group in ("WL", "NW"):
        row[f"{group.lower()}_recall"] = float(pred[wide["dga_type"].eq(group)].mean())
    if routes is not None:
        for model, share in cacic.route_shares(routes).items():
            row[f"share_{model}"] = share
    return row


def main() -> None:
    model = json.loads(FROZEN.read_text())["decision_model"]["model"]

    train = cacic.load_split("router_train")
    sample = dm.load_dm(PRED / f"router_train_dm_sample_{model}_decision_predictions.csv", train, require_doubtful=False)
    rows = sample["p_wl"].notna() & train["dga_type"].isin(["WL", "NW"])
    is_wl = train.loc[rows, "dga_type"].eq("WL")
    tau_rows = []
    for tau in GRID:
        says = sample.loc[rows, "p_wl"] >= tau
        tpr, tnr = float(says[is_wl].mean()), float((~says[~is_wl]).mean())
        tau_rows.append({"tau": tau, "wl_recall": tpr, "nw_specificity": tnr, "balanced_accuracy": (tpr + tnr) / 2})
    tau_table = pd.DataFrame(tau_rows)
    tau_dm = float(tau_table.sort_values(["balanced_accuracy", "tau"], ascending=[False, True]).iloc[0]["tau"])
    tau_table.to_csv(OUT / "dev_c3_tau.csv", index=False)

    dev = cacic.load_split("gate1")
    table = dm.load_dm(PRED / f"gate1_doubtful_{model}_decision_predictions.csv", dev)
    results = []
    pred, latency, routes = dm.c3_apply(dev, table, tau_dm)
    results.append(metrics_row(dev, "C3_type_cascade_dm", "FIXED", pred, latency, routes))
    shares = cacic.route_shares(routes)
    for seed in cacic.SEEDS:
        r = cacic.random_routes(dev, shares, seed)
        p, lat, _ = cacic.apply_routes(dev, r, probe_logit=False)
        results.append(metrics_row(dev, "C3_random_control", seed, p, lat, r))

    theta_rows = []
    for theta in GRID:
        p, lat = dm.c4_pred(dev, table, theta)
        theta_rows.append({"theta": theta, **metrics_row(dev, "C4", "FIXED", p, lat)})
    theta_table = pd.DataFrame(theta_rows)
    theta = float(theta_table.sort_values(["f1", "theta"], ascending=[False, True]).iloc[0]["theta"])
    theta_table.to_csv(OUT / "dev_c4_theta.csv", index=False)
    p, lat = dm.c4_pred(dev, table, theta)
    results.append(metrics_row(dev, "C4_judge", "FIXED", p, lat))
    for strong in ("DomURLs-BERT", "ModernBERT"):
        r = cacic.cascade_routes(dev, strong=strong)
        p, lat, _ = cacic.apply_routes(dev, r, probe_logit=True)
        results.append(metrics_row(dev, f"CascadeLogit{strong.replace('-', '')}_t0.80", "FIXED", p, lat, r))

    out = pd.DataFrame(results)
    out.to_csv(OUT / "dev_c3c4.csv", index=False)
    view = out.groupby("system", sort=False).mean(numeric_only=True)
    print(f"tau_dm = {tau_dm}  theta = {theta}")
    print(view[["f1", "fpr", "latency_mean_ms", "wl_recall", "nw_recall"]].round(4).to_string())
    update_frozen("c3", {"system": "type cascade with decision-model WL answer", "t": cacic.CASCADE_THRESHOLD, "tau_dm": tau_dm,
                         "tau_rule": "max balanced accuracy WL vs NW on the router_train sample; ties -> lower tau",
                         "dev": {k: float(v) for k, v in view.loc["C3_type_cascade_dm"].items()}})
    update_frozen("c4", {"system": "doubtful-case judge (decision-model DGA answer)", "t": cacic.CASCADE_THRESHOLD, "theta": theta,
                         "theta_rule": "max development F1; ties -> lower theta",
                         "dev": {k: float(v) for k, v in view.loc["C4_judge"].items()}})


if __name__ == "__main__":
    main()
