#!/usr/bin/env python3
"""Rebuild every CACIC system on gate1 and final_test and compare with the CACIC CSVs.

Writes:
- extension/results/routes_<split>.csv.gz  (domain, system, route)
- extension/results/reproduction_check.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402

OUT = ROOT / "extension" / "results"
CACIC_RESULTS = ROOT / "experiments" / "results"
REFERENCE = {
    "gate1": CACIC_RESULTS / "final_router_gate1_metrics.csv",
    "final_test": CACIC_RESULTS / "final_test_system_metrics.csv",
}
GROUPS = ("ALL", "NW", "WL", "benign")


def reference_key(split: str, system: str) -> tuple[str, str]:
    if system.startswith("Router_s"):
        return "FinalWeightedLogitAwareRouter", system.removeprefix("Router_s")
    if system.startswith("Random_s"):
        return "RandomCostMatched", system.removeprefix("Random_s")
    return system, ("FIXED" if split == "final_test" else "7")


def group_metrics(wide: pd.DataFrame, pred: pd.Series, latency: pd.Series) -> dict[str, dict]:
    out = {}
    for group in GROUPS:
        idx = wide.index if group == "ALL" else wide.index[wide["dga_type"].eq(group)]
        metrics = cacic.binary_metrics(wide.loc[idx, "y_true"], pred.loc[idx])
        metrics["latency_mean_ms"] = float(latency.loc[idx].mean())
        out[group] = metrics
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("gate1", "final_test"):
        print(f"[{split}] building systems", flush=True)
        wide = cacic.load_split(split)
        systems = cacic.cacic_systems(wide)
        ref = pd.read_csv(REFERENCE[split], dtype={"seed": str})
        routes_long = []
        for name, (pred, latency, routes) in systems.items():
            if routes is not None and not name.startswith("Always"):
                routes_long.append(pd.DataFrame({"domain": wide["domain"], "system": name, "route": routes}))
            ref_system, ref_seed = reference_key(split, name)
            for group, ours in group_metrics(wide, pred, latency).items():
                match = ref[(ref.system == ref_system) & (ref.seed == ref_seed) & (ref.group == group)]
                row = {"split": split, "system": name, "group": group}
                for col in ("f1", "fpr", "latency_mean_ms"):
                    row[f"{col}_ours"] = ours[col]
                    row[f"{col}_cacic"] = float(match[col].iloc[0]) if len(match) else float("nan")
                    row[f"{col}_abs_diff"] = abs(row[f"{col}_ours"] - row[f"{col}_cacic"])
                rows.append(row)
        pd.concat(routes_long).to_csv(OUT / f"routes_{split}.csv.gz", index=False)
    check = pd.DataFrame(rows)
    check.to_csv(OUT / "reproduction_check.csv", index=False)
    compared = check.dropna(subset=["f1_cacic"])
    worst = compared[["f1_abs_diff", "fpr_abs_diff", "latency_mean_ms_abs_diff"]].max()
    print(f"compared rows: {len(compared)} of {len(check)}")
    print("worst abs diff:", worst.to_dict())


if __name__ == "__main__":
    main()
