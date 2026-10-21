#!/usr/bin/env python3
"""Preregistered v2 analysis of the 27-family test. Runs once.

Usage:
  python3 extension/scripts/run_test_v2.py            # the single test run (checks the freeze first)
  python3 extension/scripts/run_test_v2.py --dry-run  # same pipeline on gate1, written to extension/results/dryrun/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic, wordlist  # noqa: E402
from extension.lib import decision as dm  # noqa: E402
from extension.lib import diagnostics as dg  # noqa: E402
from extension.lib.bootstrap import paired_f1_bootstrap  # noqa: E402
from extension.lib.freeze import check_freeze, guard_single_run, sha256  # noqa: E402
from extension.lib.frozen import FROZEN  # noqa: E402

PRED = ROOT / "experiments" / "predictions"
RESULTS = ROOT / "extension" / "results"
FREEZE = ROOT / "extension" / "protocol" / "FREEZE.sha256"
PRIOR = 1 / 101
TARGET_FPRS = (0.001, 0.01)


def type_cascade(wide: pd.DataFrame, cov: pd.Series, tau: float) -> pd.Series:
    confident = (wide["Logit"] - 0.5).abs() * 2 >= cacic.CASCADE_THRESHOLD
    strong = np.where(cov >= tau, "ModernBERT", "DomURLs-BERT")
    return pd.Series(np.where(confident, "Logit", strong), index=wide.index, dtype=object)


def adjusted_precision(recall: float, fpr: float, prior: float = PRIOR) -> float:
    num = recall * prior
    den = num + fpr * (1 - prior)
    return num / den if den else 0.0


def summarize(wide, system, seed, pred, latency, routes=None) -> list[dict]:
    rows = []
    groups = [("ALL", wide.index)] + [(g, wide.index[wide["dga_type"].eq(g)]) for g in ("WL", "NW", "benign")]
    groups += [(f"tier:{t}", wide.index[wide["tier"].eq(t)]) for t in sorted(set(wide["tier"]) - {"benign", "gate1"})]
    for group, idx in groups:
        m = cacic.binary_metrics(wide.loc[idx, "y_true"], pred.loc[idx])
        row = {"system": system, "seed": seed, "group": group, **m, "latency_mean_ms": float(latency.loc[idx].mean())}
        if group == "ALL":
            row["p_1_100"] = adjusted_precision(m["recall"], m["fpr"])
            if routes is not None:
                row.update({f"share_{k}": v for k, v in cacic.route_shares(routes).items()})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    split = "gate1" if args.dry_run else "final_test"
    out_dir = RESULTS / "dryrun" if args.dry_run else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "test_v2_decision_summary.md"
    if not args.dry_run:
        check_freeze(FREEZE)
        guard_single_run(summary_path)

    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    words_path = ROOT / cfg["c2"]["word_file"]
    if sha256(words_path) != cfg["c2"]["word_file_sha256"]:
        raise SystemExit("word file changed after the freeze")
    model = cfg["decision_model"]["model"]
    dm_run = "gate1_doubtful" if args.dry_run else "final_test_doubtful"

    wide = cacic.load_split(split)
    dev = cacic.load_split("gate1")
    rows: list[dict] = []
    preds: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}

    # CACIC systems, rebuilt exactly
    for name, (pred, latency, routes) in cacic.cacic_systems(wide).items():
        seed = name.split("_s")[-1] if "_s" in name else "FIXED"
        rows += summarize(wide, name.split("_s")[0] if "_s" in name else name, seed, pred, latency, routes)
        preds[name] = pred
    scores.update({f"Always{m}": wide[m].astype(float) for m in cacic.MODELS})
    scores["EnsembleMean"] = wide[cacic.MODELS].mean(axis=1)

    # Diagnostic only (chosen with test hindsight in CACIC): Logit -> ModernBERT cascade
    r = cacic.cascade_routes(wide, strong="ModernBERT")
    p, lat, _ = cacic.apply_routes(wide, r, probe_logit=True)
    rows += summarize(wide, "Diag_CascadeLogitModernBERT_t0.80", "FIXED", p, lat, r)

    # C1: FPR-aware router (+ route-matched random control per seed)
    c1 = cfg["c1"]
    c1_preds = []
    for seed in c1["seeds"]:
        routes = cacic.router_routes(cacic.train_router(seed, fp_weight=c1["fp_weight"]), wide)
        pred, latency, score = cacic.apply_routes(wide, routes, probe_logit=True)
        rows += summarize(wide, "C1_fpr_router", seed, pred, latency, routes)
        preds[f"C1_s{seed}"], scores[f"C1_s{seed}"] = pred, score
        c1_preds.append(pred)
        rr = cacic.random_routes(wide, cacic.route_shares(routes), seed)
        pr, lr, _ = cacic.apply_routes(wide, rr, probe_logit=False)
        rows += summarize(wide, "C1_random_control", seed, pr, lr, rr)
    preds["C1_majority"] = (pd.concat(c1_preds, axis=1).mean(axis=1) >= 0.5).astype(int)
    preds["Router_majority"] = (pd.concat([preds[f"Router_s{s}"] for s in cacic.SEEDS], axis=1).mean(axis=1) >= 0.5).astype(int)

    # C2: type cascade with the dictionary rule (+ control)
    words = wordlist.load_words(words_path, min_len=cfg["c2"]["min_word_len"])
    cov = pd.Series([wordlist.coverage(wordlist.main_label(d), words) for d in wide["domain"]], index=wide.index)
    routes = type_cascade(wide, cov, cfg["c2"]["tau"])
    pred, latency, _ = cacic.apply_routes(wide, routes, probe_logit=True)
    rows += summarize(wide, "C2_type_cascade_rule", "FIXED", pred, latency, routes)
    preds["C2"] = pred
    for seed in cacic.SEEDS:
        rr = cacic.random_routes(wide, cacic.route_shares(routes), seed)
        pr, lr, _ = cacic.apply_routes(wide, rr, probe_logit=False)
        rows += summarize(wide, "C2_random_control", seed, pr, lr, rr)
        preds[f"C2_control_s{seed}"] = pr

    # C3 and C4: decision model on the doubtful domains
    table = dm.load_dm(PRED / f"{dm_run}_{model}_decision_predictions.csv", wide)
    pred, latency, routes = dm.c3_apply(wide, table, cfg["c3"]["tau_dm"])
    rows += summarize(wide, "C3_type_cascade_dm", "FIXED", pred, latency, routes)
    preds["C3"] = pred
    for seed in cacic.SEEDS:
        rr = cacic.random_routes(wide, cacic.route_shares(routes), seed)
        pr, lr, _ = cacic.apply_routes(wide, rr, probe_logit=False)
        rows += summarize(wide, "C3_random_control", seed, pr, lr, rr)
        preds[f"C3_control_s{seed}"] = pr
    pred, latency = dm.c4_pred(wide, table, cfg["c4"]["theta"])
    rows += summarize(wide, "C4_judge_dm", "FIXED", pred, latency)
    preds["C4"] = pred

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "test_v2_system_metrics.csv", index=False)

    # Fixed-FPR operating points with development thresholds
    fixed = []
    dev_scores = {f"Always{m}": dev[m].astype(float) for m in cacic.MODELS}
    dev_scores["EnsembleMean"] = dev[cacic.MODELS].mean(axis=1)
    for name, score in scores.items():
        for target in TARGET_FPRS:
            if name.startswith("C1_s"):
                thr = c1["thresholds_by_seed"][name.removeprefix("C1_s")][str(target)]
            else:
                thr = dg.threshold_at_fpr(dev["y_true"], dev_scores[name], target)
            rate = dg.rate_at_threshold(wide["y_true"], score, thr)
            fixed.append({"system": name, "target_fpr": target, "threshold_dev": thr, **rate,
                          "p_1_100": adjusted_precision(rate["recall"], rate["fpr"])})
    pd.DataFrame(fixed).to_csv(out_dir / "test_v2_fixed_fpr.csv", index=False)

    # Paired two-stage bootstrap (DGA families x benign domains), 1000 replicates, seed 7
    y, fam = wide["y_true"], wide["family"]
    pairs = [("C1_majority", "Router_majority"), ("C2", "CascadeLogitDomURLs_t0.80"), ("C3", "C2"), ("C4", "CascadeLogitDomURLs_t0.80")]
    pairs += [("C2", f"C2_control_s{s}") for s in cacic.SEEDS] + [("C3", f"C3_control_s{s}") for s in cacic.SEEDS]
    boot = [{"a": a, "b": b, **paired_f1_bootstrap(y, fam, preds[a], preds[b], n_boot=1000, seed=7)} for a, b in pairs]
    boot = pd.DataFrame(boot)
    boot.to_csv(out_dir / "test_v2_bootstrap.csv", index=False)

    # Decision rules (preregistro v2)
    allg = metrics[metrics["group"].eq("ALL")]
    mean = allg.groupby("system").mean(numeric_only=True)
    lows = boot.set_index(["a", "b"])["low"]
    rules = {
        "C1 FPR < 0.0455": mean.loc["C1_fpr_router", "fpr"] < 0.0455,
        "C1 F1 >= 0.9125": mean.loc["C1_fpr_router", "f1"] >= 0.9175 - 0.005,
        "C2 beats every control seed (CI low > 0)": all(lows[("C2", f"C2_control_s{s}")] > 0 for s in cacic.SEEDS),
        "C2 F1 >= 0.8703": mean.loc["C2_type_cascade_rule", "f1"] >= 0.8703,
        "C2 FPR <= 0.0320": mean.loc["C2_type_cascade_rule", "fpr"] <= 0.0320,
        "C3 beats every control seed (CI low > 0)": all(lows[("C3", f"C3_control_s{s}")] > 0 for s in cacic.SEEDS),
        "C3 better than C2 (CI low > 0)": lows[("C3", "C2")] > 0,
        "C4 better than the CACIC cascade (CI low > 0)": lows[("C4", "CascadeLogitDomURLs_t0.80")] > 0,
        "C4 FPR <= 0.0320": mean.loc["C4_judge_dm", "fpr"] <= 0.0320,
    }
    lines = [f"# Test v2 decision summary ({'DRY RUN on gate1' if args.dry_run else 'final test, single run'})", ""]
    lines += ["| System | F1 | FPR | Latency (ms) |", "|---|---|---|---|"]
    for system, row in mean.iterrows():
        lines.append(f"| {system} | {row['f1']:.4f} | {row['fpr']:.4f} | {row['latency_mean_ms']:.3f} |")
    lines += ["", "| Rule | Result |", "|---|---|"]
    lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in rules.items()]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
