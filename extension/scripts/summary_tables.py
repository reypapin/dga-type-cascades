#!/usr/bin/env python3
"""One row per system on development (gate1) and test (final_test), plus per-domain post-hoc Jev predictions.

Frozen systems are rebuilt with the saved routers (extension/models/) and checked against the earlier outputs.
Post-hoc systems (marked † on the results page) use the thresholds in extension/config/posthoc_jev.json.
Writes extension/results/summary_<split>.csv and extension/results/predictions_posthoc_jev_<split>.csv.gz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic, wordlist  # noqa: E402
from extension.lib import decision as dm  # noqa: E402
from extension.lib.routers import load_router  # noqa: E402

PRED = ROOT / "experiments" / "predictions"
OUT = ROOT / "extension" / "results"
FROZEN = json.loads((ROOT / "extension" / "config" / "frozen_v2.json").read_text())
POSTHOC = json.loads((ROOT / "extension" / "config" / "posthoc_jev.json").read_text())
DM_RUN = {"gate1": "gate1_doubtful", "final_test": "final_test_doubtful"}
PRIOR = 1 / 101


def prior_precision(recall: float, fpr: float, prior: float = PRIOR) -> float:
    # same as adjusted_precision in the frozen run_test_v2.py: precision with 1 DGA domain per 100 benign ones
    num = recall * prior
    den = num + fpr * (1 - prior)
    return num / den if den else 0.0


def metrics(wide, pred, latency) -> dict:
    m = cacic.binary_metrics(wide["y_true"], pred)
    row = {"f1": m["f1"], "fpr": m["fpr"], "recall": m["recall"], "p_1_100": prior_precision(m["recall"], m["fpr"]),
           "latency_mean_ms": float(latency.mean())}
    for g in ("WL", "NW"):
        row[f"{g.lower()}_recall"] = float(pred[wide["dga_type"].eq(g)].mean())
    for t in ("new_family", "variant_of_seen"):
        mask = wide["tier"].eq(t) & wide["y_true"].eq(1)
        row[f"{t}_recall"] = float(pred[mask].mean()) if mask.any() else np.nan
    return row


def systems(split: str):
    wide = cacic.load_split(split)
    out: dict[str, list] = {}
    for model in cacic.MODELS:
        p, lat, _ = cacic.apply_routes(wide, pd.Series(model, index=wide.index, dtype=object), probe_logit=False)
        out[f"Always{model}"] = [(p, lat)]
    out["EnsembleMean"] = [((wide[cacic.MODELS].mean(axis=1) >= 0.5).astype(int), wide[[f"{m}__latency" for m in cacic.MODELS]].sum(axis=1))]
    for name, strong in (("CascadeLogitDomURLs_t0.80", "DomURLs-BERT"), ("Diag_CascadeLogitModernBERT_t0.80", "ModernBERT")):
        p, lat, _ = cacic.apply_routes(wide, cacic.cascade_routes(wide, strong=strong), probe_logit=True)
        out[name] = [(p, lat)]
    for kind, name in (("cacic", "Router"), ("c1", "C1_fpr_router")):
        out[name] = []
        for seed in cacic.SEEDS:
            model = load_router(f"router_{kind}_s{seed}")
            p, lat, _ = cacic.apply_routes(wide, cacic.router_routes(model, wide), probe_logit=True)
            out[name].append((p, lat))
    words = wordlist.load_words(ROOT / FROZEN["c2"]["word_file"], min_len=FROZEN["c2"]["min_word_len"])
    cov = pd.Series([wordlist.coverage(wordlist.main_label(d), words) for d in wide["domain"]], index=wide.index)
    confident = (wide["Logit"] - 0.5).abs() * 2 >= cacic.CASCADE_THRESHOLD
    routes = pd.Series(np.where(confident, "Logit", np.where(cov >= FROZEN["c2"]["tau"], "ModernBERT", "DomURLs-BERT")), index=wide.index, dtype=object)
    p, lat, _ = cacic.apply_routes(wide, routes, probe_logit=True)
    out["C2_type_cascade_rule"] = [(p, lat)]
    d1 = dm.load_dm(PRED / f"{DM_RUN[split]}_d1_decision_predictions.csv", wide)
    jev = dm.load_dm(PRED / f"{DM_RUN[split]}_jev_decision_predictions.csv", wide)
    out["C3_type_cascade_dm"] = [dm.c3_apply(wide, d1, FROZEN["c3"]["tau_dm"])[:2]]
    out["C4_judge_dm"] = [dm.c4_pred(wide, d1, FROZEN["c4"]["theta"])]
    out["C3_type_cascade_jev"] = [dm.c3_apply(wide, jev, POSTHOC["tau_jev"])[:2]]
    out["C4_judge_jev"] = [dm.c4_pred(wide, jev, POSTHOC["theta_jev"])]
    return wide, out


def check(table: pd.DataFrame, system: str, ref: dict, label: str) -> float:
    row = table.set_index("system").loc[system]
    return max(abs(float(row[k]) - float(ref[k])) for k in ("f1", "fpr"))


def main() -> None:
    tables = {}
    for split in ("gate1", "final_test"):
        wide, out = systems(split)
        rows = []
        for name, runs in out.items():
            ms = [metrics(wide, p, lat) for p, lat in runs]
            row = {"system": name, "n_seeds": len(ms), **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}}
            if len(ms) > 1:  # spread over the router seeds (sample SD)
                f1s, fprs = [m["f1"] for m in ms], [m["fpr"] for m in ms]
                row.update({"f1_sd": float(np.std(f1s, ddof=1)), "fpr_sd": float(np.std(fprs, ddof=1)),
                            "f1_min": float(min(f1s)), "f1_max": float(max(f1s))})
            rows.append(row)
        tables[split] = pd.DataFrame(rows)
        tables[split].to_csv(OUT / f"summary_{split}.csv", index=False)
        pd.DataFrame({"domain": wide["domain"], "pred_C3_jev": out["C3_type_cascade_jev"][0][0],
                      "pred_C4_jev": out["C4_judge_jev"][0][0]}).to_csv(OUT / f"predictions_posthoc_jev_{split}.csv.gz", index=False)
        print(split)
        print(tables[split].drop(columns=["n_seeds"]).round(4).to_string(index=False))

    worst = 0.0
    dev, test = tables["gate1"], tables["final_test"]
    worst = max(worst, check(dev, "Router", FROZEN["c1"]["dev_mean_fp_weight_0"], "dev router"))
    worst = max(worst, check(dev, "C1_fpr_router", FROZEN["c1"]["dev_mean"], "dev C1"))
    for system, block in (("C2_type_cascade_rule", "c2"), ("C3_type_cascade_dm", "c3"), ("C4_judge_dm", "c4")):
        worst = max(worst, check(dev, system, FROZEN[block]["dev"], f"dev {block}"))
    pj = pd.read_csv(OUT / "posthoc_jev_dev.csv").set_index("system")
    for system in ("C3_type_cascade_jev", "C4_judge_jev", "CascadeLogitDomURLs_t0.80"):
        worst = max(worst, check(dev, system, pj.loc[system], f"dev {system}"))
    ref = pd.read_csv(OUT / "test_v2_system_metrics.csv", dtype={"seed": str})
    ref = ref[ref["group"].eq("ALL")].groupby("system").mean(numeric_only=True)
    for system in ref.index.intersection(test["system"]):
        worst = max(worst, check(test, system, ref.loc[system], f"test {system}"))
        worst = max(worst, abs(float(test.set_index("system").loc[system, "p_1_100"]) - float(ref.loc[system, "p_1_100"])))
    pj = pd.read_csv(OUT / "posthoc_jev_test.csv").set_index("system")
    for system in ("C3_type_cascade_jev", "C4_judge_jev"):
        worst = max(worst, check(test, system, pj.loc[system], f"test {system}"))
    print(f"max |difference| in F1/FPR/precision at 1:100 against the earlier outputs: {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("rebuilt systems do not match the earlier outputs")


if __name__ == "__main__":
    main()
