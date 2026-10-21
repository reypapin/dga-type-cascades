#!/usr/bin/env python3
"""Error diagnosis of the CACIC systems. Subcommands: d1d5, d2, d3d4, d6.

Routed systems are rebuilt from extension/results/routes_<split>.csv.gz,
written and checked against CACIC by reproduce_cacic.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib import diagnostics as dg  # noqa: E402

OUT = ROOT / "extension" / "results"
EVAL_SPLITS = ("gate1", "final_test")
SCORE_SYSTEMS = ("AlwaysLogit", "AlwaysModernBERT", "AlwaysDomURLs-BERT", "EnsembleMean")
TARGET_FPRS = (0.001, 0.01)


def load_systems(split: str) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series]]:
    """Return (wide, preds, routes) with routes taken from the reproduction run."""
    wide = cacic.load_split(split)
    preds: dict[str, pd.Series] = {f"Always{m}": wide[f"{m}__pred"].astype(int) for m in cacic.MODELS}
    preds["EnsembleMean"] = (wide[cacic.MODELS].mean(axis=1) >= 0.5).astype(int)
    routes_long = pd.read_csv(OUT / f"routes_{split}.csv.gz")
    routes: dict[str, pd.Series] = {}
    for system, part in routes_long.groupby("system", sort=False):
        if not part["domain"].reset_index(drop=True).equals(wide["domain"]):
            raise SystemExit(f"routes for {system} on {split} are not aligned with the wide table")
        route = pd.Series(part["route"].to_numpy(), index=wide.index, dtype=object)
        pred, _, _ = cacic.apply_routes(wide, route, probe_logit=not system.startswith("Random"))
        preds[system] = pred
        routes[system] = route
    return wide, preds, routes


def scores(wide: pd.DataFrame) -> dict[str, pd.Series]:
    out = {f"Always{m}": wide[m].astype(float) for m in cacic.MODELS}
    out["EnsembleMean"] = wide[cacic.MODELS].mean(axis=1)
    return out


def cmd_d1d5() -> None:
    family_rows, allfail_rows = [], []
    for split in EVAL_SPLITS:
        wide, preds, _ = load_systems(split)
        family_rows.append(dg.family_table(wide, preds).assign(split=split))
        allfail_rows.append(dg.allfail_profile(wide).assign(split=split))
    pd.concat(family_rows).to_csv(OUT / "diag_family.csv", index=False)
    pd.concat(allfail_rows).to_csv(OUT / "diag_allfail.csv", index=False)

    dev, test = cacic.load_split("gate1"), cacic.load_split("final_test")
    dev_scores, test_scores = scores(dev), scores(test)
    rows = []
    for system in SCORE_SYSTEMS:
        for target in TARGET_FPRS:
            thr_dev = dg.threshold_at_fpr(dev["y_true"], dev_scores[system], target)
            thr_test = dg.threshold_at_fpr(test["y_true"], test_scores[system], target)
            at_dev = dg.rate_at_threshold(test["y_true"], test_scores[system], thr_dev)
            at_test = dg.rate_at_threshold(test["y_true"], test_scores[system], thr_test)
            rows.append(
                {
                    "system": system,
                    "target_fpr": target,
                    "threshold_dev": thr_dev,
                    "test_recall_dev_threshold": at_dev["recall"],
                    "test_fpr_dev_threshold": at_dev["fpr"],
                    "threshold_test": thr_test,
                    "test_recall_test_threshold": at_test["recall"],
                    "test_fpr_test_threshold": at_test["fpr"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "diag_thresholds.csv", index=False)
    print(table.round(4).to_string(index=False))


def cmd_d2() -> None:
    parts = []
    for split in EVAL_SPLITS:
        wide = cacic.load_split(split)
        for strong in ("DomURLs-BERT", "ModernBERT"):
            parts.append(dg.rescue_damage(wide, strong=strong).assign(split=split))
    table = pd.concat(parts)
    table.to_csv(OUT / "diag_cascade_bins.csv", index=False)
    escalated = table[table["bin_high"] <= 0.8 + 1e-9]
    summary = escalated.groupby(["split", "strong", "dga_type"])[["n", "rescue", "damage", "net"]].sum()
    print(summary.to_string())


def cmd_d3d4() -> None:
    choice_parts, benign_parts = [], []
    for split in EVAL_SPLITS:
        wide, preds, routes = load_systems(split)
        for system, route in routes.items():
            if not system.startswith("Router_s"):
                continue
            seed = system.removeprefix("Router_s")
            choice_parts.append(dg.router_choice_table(wide, route).assign(split=split, seed=seed))
            benign = dg.benign_route_fpr(wide, route)
            observed = float(preds[system][wide["y_true"].astype(int).eq(0)].mean())
            independence = float((benign["share_benign"] * benign["fpr_expert_all"]).sum())
            benign_parts.append(benign.assign(split=split, seed=seed, router_fpr=observed, independence_fpr=independence))
    choice = pd.concat(choice_parts)
    benign = pd.concat(benign_parts)
    for table in (choice, benign):
        keys = ["split", "dga_type"] if "dga_type" in table else ["split", "route"]
        mean = table.drop(columns="seed").groupby(keys, as_index=False).mean(numeric_only=True).assign(seed="MEAN")
        table_all = pd.concat([table, mean], ignore_index=True)
        name = "diag_router_choice.csv" if "dga_type" in table else "diag_benign_routes.csv"
        table_all.to_csv(OUT / name, index=False)
        print(mean.round(4).to_string(index=False))


def cmd_d6() -> None:
    summary, bins = [], []
    for split in cacic.SPLITS:
        wide = cacic.load_split(split)
        for expert in cacic.MODELS:
            for group in ("ALL", "WL", "NW", "benign"):
                idx = wide.index if group == "ALL" else wide.index[wide["dga_type"].eq(group)]
                table, ece, brier = dg.calibration(wide.loc[idx, "y_true"], wide.loc[idx, expert])
                summary.append({"split": split, "expert": expert, "group": group, "n": len(idx), "ece": ece, "brier": brier})
                bins.append(table.assign(split=split, expert=expert, group=group))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "diag_calibration.csv", index=False)
    pd.concat(bins).to_csv(OUT / "diag_reliability.csv", index=False)
    print(summary.pivot_table(index=["expert", "group"], columns="split", values="ece").round(4).to_string())


COMMANDS = {"d1d5": cmd_d1d5, "d2": cmd_d2, "d3d4": cmd_d3d4, "d6": cmd_d6}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
