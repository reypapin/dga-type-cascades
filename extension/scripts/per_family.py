#!/usr/bin/env python3
"""Descriptive per-family detection rate on the test (27 families), from the saved per-domain predictions.

No decision is made from these numbers: the systems are the frozen ones, evaluated once, plus the post-hoc
Jev variants (C3_jev, C4_jev) from extension/results/predictions_posthoc_jev_final_test.csv.gz.
Writes extension/results/test_v2_per_family.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402

OUT = ROOT / "extension" / "results"


def main() -> None:
    test = pd.read_csv(OUT / "predictions_final_test.csv.gz")
    systems = {
        "Logit": test["pred_AlwaysLogit"],
        "ModernBERT": test["pred_AlwaysModernBERT"],
        "DomURLs-BERT": test["pred_AlwaysDomURLs-BERT"],
        "Ensemble": test["pred_EnsembleMean"],
        "CascadeCACIC": test["pred_CascadeCACIC"],
        "RouterCACIC": test[[f"pred_RouterCACIC_s{s}" for s in cacic.SEEDS]].mean(axis=1),
        "C1": test[[f"pred_RouterC1_s{s}" for s in cacic.SEEDS]].mean(axis=1),
        "C2": test["pred_C2"],
        "C3": test["pred_C3"],
        "C4": test["pred_C4"],
    }
    jev = pd.read_csv(OUT / "predictions_posthoc_jev_final_test.csv.gz")
    wide = cacic.load_split("final_test")
    if not ((jev["domain"].values == test["domain"].values).all() and (wide["domain"].values == test["domain"].values).all()):
        raise SystemExit("prediction files are not in the test row order")
    diag = cacic.apply_routes(wide, cacic.cascade_routes(wide, strong="ModernBERT"), probe_logit=True)[0]
    systems.update({"C3_jev": jev["pred_C3_jev"], "C4_jev": jev["pred_C4_jev"],
                    "Diag_LogitModernBERT": pd.Series(diag.values, index=test.index)})
    frame = test[["family", "dga_type", "tier", "y_true"]].assign(**systems)
    dga = frame[frame["y_true"].eq(1)]
    table = dga.groupby(["dga_type", "tier", "family"])[list(systems)].mean()
    table.insert(0, "n", dga.groupby(["dga_type", "tier", "family"]).size())
    table = table.reset_index()
    benign = frame[frame["y_true"].eq(0)]
    fpr = {"dga_type": "benign", "tier": "", "family": "benign (false positive rate)", "n": len(benign),
           **{k: float(benign[k].mean()) for k in systems}}
    table = pd.concat([table, pd.DataFrame([fpr])], ignore_index=True)
    table.to_csv(OUT / "test_v2_per_family.csv", index=False)
    print(f"{table['dga_type'].ne('benign').sum()} families")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
