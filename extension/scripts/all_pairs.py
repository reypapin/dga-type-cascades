#!/usr/bin/env python3
"""All-vs-all paired F1 comparisons on the test: 3 single models and configurations C1-C10 (descriptive).

Page names -> internal names:
  Logit, ModernBERT, DomURLs_BERT = AlwaysLogit, AlwaysModernBERT, AlwaysDomURLs-BERT
  C1 = EnsembleMean            C2 = CascadeLogitDomURLs_t0.80   C3 = Diag_CascadeLogitModernBERT_t0.80 (†)
  C4 = Router (5 seeds)        C5 = C1_fpr_router (5 seeds, "C1" in the preregistration)
  C6 = C2_type_cascade_rule    C7 = C3_type_cascade_dm          C8 = C3_type_cascade_jev (†)
  C9 = C4_judge_dm ("C4" in the preregistration)                C10 = C4_judge_jev (†)
Same paired two-stage bootstrap as the test run (DGA families x benign domains, 1000 replicates, seed 7).
C4 and C5 use the mean F1 of their 5 seeds in every replicate, as in the summary tables (paired_f1_bootstrap_multi).
The bootstrap is antisymmetric, so only the 78 unordered pairs are computed.
Writes extension/results/test_v2_all_pairs.csv.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib.bootstrap_multi import paired_f1_bootstrap_multi  # noqa: E402

OUT = ROOT / "extension" / "results"


def seeds(frame: pd.DataFrame, prefix: str) -> list[pd.Series]:
    return [frame[f"{prefix}{s}"] for s in cacic.SEEDS]


def main() -> None:
    test = pd.read_csv(OUT / "predictions_final_test.csv.gz")
    jev = pd.read_csv(OUT / "predictions_posthoc_jev_final_test.csv.gz")
    wide = cacic.load_split("final_test")
    if not ((jev["domain"].values == test["domain"].values).all() and (wide["domain"].values == test["domain"].values).all()):
        raise SystemExit("prediction files are not in the same row order")
    c3 = cacic.apply_routes(wide, cacic.cascade_routes(wide, strong="ModernBERT"), probe_logit=True)[0]
    systems = {
        "Logit": [test["pred_AlwaysLogit"]], "ModernBERT": [test["pred_AlwaysModernBERT"]], "DomURLs_BERT": [test["pred_AlwaysDomURLs-BERT"]],
        "C1": [test["pred_EnsembleMean"]], "C2": [test["pred_CascadeCACIC"]], "C3": [pd.Series(c3.values, index=test.index)],
        "C4": seeds(test, "pred_RouterCACIC_s"), "C5": seeds(test, "pred_RouterC1_s"),
        "C6": [test["pred_C2"]], "C7": [test["pred_C3"]], "C8": [jev["pred_C3_jev"]],
        "C9": [test["pred_C4"]], "C10": [jev["pred_C4_jev"]],
    }
    rows = []
    for a, b in itertools.combinations(systems, 2):
        r = paired_f1_bootstrap_multi(test["y_true"], test["family"], systems[a], systems[b], n_boot=1000, seed=7)
        rows.append({"a": a, "b": b, "diff": r["diff"], "low": r["low"], "high": r["high"],
                     "clear": bool(r["low"] > 0 or r["high"] < 0), "status": "descriptive, not preregistered"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_v2_all_pairs.csv", index=False)
    print(f"{len(table)} pairs, {int(table['clear'].sum())} clear")
    for name in systems:
        wins = int(((table["a"].eq(name)) & (table["low"] > 0)).sum() + ((table["b"].eq(name)) & (table["high"] < 0)).sum())
        losses = int(((table["a"].eq(name)) & (table["high"] < 0)).sum() + ((table["b"].eq(name)) & (table["low"] > 0)).sum())
        print(f"{name:13s} clear wins {wins:2d}  clear losses {losses:2d}")


if __name__ == "__main__":
    main()
