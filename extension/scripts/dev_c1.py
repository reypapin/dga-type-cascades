#!/usr/bin/env python3
"""C1: FPR-aware router, chosen on development data (gate1) only.

Same router as CACIC (54 seen families, features, seeds, type weights); benign
training rows where some expert raises a false alarm get weight x (1 + fp_weight).
Selection rule (fixed before running): lowest mean dev FPR among fp_weight values
whose mean dev F1 >= mean dev F1 at fp_weight = 0 minus 0.005; ties -> lower latency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib import diagnostics as dg  # noqa: E402
from extension.lib.frozen import update_frozen  # noqa: E402

OUT = ROOT / "extension" / "results"
FP_WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0)
F1_TOLERANCE = 0.005
TARGET_FPRS = (0.001, 0.01)


def main() -> None:
    dev = cacic.load_split("gate1")
    rows, thresholds = [], {}
    for fp_weight in FP_WEIGHTS:
        for seed in cacic.SEEDS:
            routes = cacic.router_routes(cacic.train_router(seed, fp_weight=fp_weight), dev)
            pred, latency, score = cacic.apply_routes(dev, routes, probe_logit=True)
            metrics = cacic.binary_metrics(dev["y_true"], pred)
            row = {
                "fp_weight": fp_weight,
                "seed": seed,
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "recall": metrics["recall"],
                "latency_mean_ms": float(latency.mean()),
                "wl_recall": float(pred[dev["dga_type"].eq("WL")].mean()),
            }
            for target in TARGET_FPRS:
                thr = dg.threshold_at_fpr(dev["y_true"], score, target)
                row[f"thr_{target}"] = thr
                row[f"dev_recall_at_{target}"] = dg.rate_at_threshold(dev["y_true"], score, thr)["recall"]
                thresholds.setdefault(fp_weight, {}).setdefault(str(seed), {})[str(target)] = thr
            rows.append(row)
            print(f"fp_weight={fp_weight} seed={seed} F1={row['f1']:.4f} FPR={row['fpr']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dev_c1.csv", index=False)

    mean = table.groupby("fp_weight").mean(numeric_only=True).drop(columns="seed")
    floor = mean.loc[0.0, "f1"] - F1_TOLERANCE
    eligible = mean[mean["f1"] >= floor].sort_values(["fpr", "latency_mean_ms"])
    chosen = float(eligible.index[0])
    print(mean.round(4).to_string())
    print(f"chosen fp_weight = {chosen}")
    update_frozen(
        "c1",
        {
            "system": "FPR-aware router (CACIC router + benign false-alarm weight)",
            "fp_weight": chosen,
            "fp_weight_grid": list(FP_WEIGHTS),
            "seeds": list(cacic.SEEDS),
            "selection_rule": "lowest mean dev FPR with mean dev F1 >= F1(fp_weight=0) - 0.005; ties by lower latency",
            "score": "p_dga of the chosen expert",
            "thresholds_by_seed": thresholds[chosen],
            "dev_mean": {k: float(v) for k, v in mean.loc[chosen].items()},
            "dev_mean_fp_weight_0": {k: float(v) for k, v in mean.loc[0.0].items()},
        },
    )


if __name__ == "__main__":
    main()
