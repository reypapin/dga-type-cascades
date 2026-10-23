#!/usr/bin/env python3
"""Checks asked for by the tutor review of 2026-10-23 (descriptive, added after the test run).

Built only from saved predictions; no detector or decision model is run again and no threshold changes.
  1. Doubtful-only random control: Logit keeps its confident domains, and each doubtful domain goes at
     random to ModernBERT or DomURLs_BERT with the shares that C6 used on the same set (5 seeds).
  2. WL/NW AUC of the word cues on doubtful DGA domains that were not used to choose the decision model.
  3. Intervals for pairs with C4 or C5 that also resample the router seed in every replicate.
  4. C4 and C5 retrained without the router-training domains that are also in the expert training data.
Writes extension/results/revision_*.csv. Page names as in all_pairs.py (C6 = C2_type_cascade_rule, ...).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extension.lib import cacic  # noqa: E402
from extension.lib.bootstrap import _f1, paired_f1_bootstrap  # noqa: E402
from extension.lib.bootstrap_multi import paired_f1_bootstrap_multi  # noqa: E402
from extension.lib.decision import doubtful_mask  # noqa: E402
from extension.lib.wordlist import coverage, load_words, main_label  # noqa: E402
from train_final_router import cheapest_correct_targets, make_router, sample_weights  # noqa: E402

OUT = ROOT / "extension" / "results"
PRED = ROOT / "experiments" / "predictions"
EXPERT_TRAIN = ROOT / "data" / "router_expert_overlap.csv"
if not EXPERT_TRAIN.is_file():
    EXPERT_TRAIN = ROOT / "paper_1" / "release" / "data" / "router_expert_overlap.csv"
WORDS = ROOT / "extension" / "data" / "american-english.txt"
SMOKE = ROOT / "extension" / "colab" / "smoke" / "smoke_domains.csv"


def f1_fpr(y, pred) -> tuple[float, float]:
    y, p = np.asarray(y, int), np.asarray(pred, int)
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    return float(_f1(tp, fp, fn)), fp / int((y == 0).sum())


def load_test() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[pd.Series]]]:
    test = pd.read_csv(OUT / "predictions_final_test.csv.gz")
    jev = pd.read_csv(OUT / "predictions_posthoc_jev_final_test.csv.gz")
    wide = cacic.load_split("final_test")
    if not ((jev["domain"].values == test["domain"].values).all() and (wide["domain"].values == test["domain"].values).all()):
        raise SystemExit("prediction files are not in the same row order")
    c3 = cacic.apply_routes(wide, cacic.cascade_routes(wide, strong="ModernBERT"), probe_logit=True)[0]
    seeds = lambda prefix: [test[f"{prefix}{s}"] for s in cacic.SEEDS]  # noqa: E731
    systems = {
        "Logit": [test["pred_AlwaysLogit"]], "ModernBERT": [test["pred_AlwaysModernBERT"]], "DomURLs_BERT": [test["pred_AlwaysDomURLs-BERT"]],
        "C1": [test["pred_EnsembleMean"]], "C2": [test["pred_CascadeCACIC"]], "C3": [pd.Series(c3.values, index=test.index)],
        "C4": seeds("pred_RouterCACIC_s"), "C5": seeds("pred_RouterC1_s"),
        "C6": [test["pred_C2"]], "C7": [test["pred_C3"]], "C8": [jev["pred_C3_jev"]],
        "C9": [test["pred_C4"]], "C10": [jev["pred_C4_jev"]],
    }
    return test, wide, systems


def doubtful_random_control(test: pd.DataFrame, wide: pd.DataFrame, systems: dict) -> None:
    doubtful = doubtful_mask(wide)
    routes_c6 = test["route_C2"]
    if not (routes_c6.ne("Logit") == doubtful).all():
        raise SystemExit("C6 doubtful set differs from the Logit confidence rule")
    share_mb = float(routes_c6[doubtful].eq("ModernBERT").mean())
    y, fam = test["y_true"], test["family"]
    rows, controls = [], []
    for seed in cacic.SEEDS:
        pick = np.random.default_rng(seed).random(len(wide)) < share_mb
        routes = pd.Series(np.where(~doubtful, "Logit", np.where(pick, "ModernBERT", "DomURLs-BERT")), index=wide.index, dtype=object)
        pred = cacic.apply_routes(wide, routes, probe_logit=True)[0]
        controls.append(pd.Series(pred.values, index=test.index))
        f1, fpr = f1_fpr(y, pred)
        wl = float(pred[test["dga_type"].eq("WL")].mean())
        for ref in ("C6", "C3"):
            r = paired_f1_bootstrap(y, fam, systems[ref][0], pred, n_boot=1000, seed=7)
            rows.append({"control_seed": seed, "share_modernbert_doubtful": share_mb, "control_f1": f1, "control_fpr": fpr,
                         "control_wl_recall": wl, "ref": ref, "diff_ref_minus_control": r["diff"], "low": r["low"],
                         "high": r["high"], "clear": bool(r["low"] > 0 or r["high"] < 0)})
    for ref in ("C6", "C3"):
        r = paired_f1_bootstrap_multi(y, fam, systems[ref], controls, n_boot=1000, seed=7)
        f1s = [f1_fpr(y, c) for c in controls]
        rows.append({"control_seed": "mean5", "share_modernbert_doubtful": share_mb, "control_f1": np.mean([a for a, _ in f1s]),
                     "control_fpr": np.mean([b for _, b in f1s]), "control_wl_recall": np.mean([c[test["dga_type"].eq("WL")].mean() for c in controls]),
                     "ref": ref, "diff_ref_minus_control": r["diff"], "low": r["low"], "high": r["high"],
                     "clear": bool(r["low"] > 0 or r["high"] < 0)})
    pd.DataFrame(rows).to_csv(OUT / "revision_doubtful_random_control.csv", index=False)
    print(f"[1] share of doubtful domains to ModernBERT {share_mb:.4f}; written revision_doubtful_random_control.csv")


def cue_auc() -> None:
    words = load_words(WORDS)
    smoke = set(pd.read_csv(SMOKE)["domain"].str.strip().str.lower())
    rows = []
    for split, label in (("final_test", "test"), ("gate1", "development")):
        d1 = pd.read_csv(PRED / f"{split}_doubtful_d1_decision_predictions.csv")
        jev = pd.read_csv(PRED / f"{split}_doubtful_jev_decision_predictions.csv")
        frame = d1[["domain", "dga_type", "p_wl"]].rename(columns={"p_wl": "d1"}).merge(
            jev[["domain", "p_wl"]].rename(columns={"p_wl": "jev"}), on="domain", how="inner", validate="one_to_one")
        if len(frame) != len(d1) or len(frame) != len(jev):
            raise SystemExit(f"{split}: d1 and Jev doubtful sets differ")
        frame = frame[frame["dga_type"].isin(["WL", "NW"])]
        frame = frame[~frame["domain"].str.strip().str.lower().isin(smoke)]
        frame["dictionary"] = [coverage(main_label(d), words) for d in frame["domain"]]
        is_wl = frame["dga_type"].eq("WL").astype(int)
        for cue in ("dictionary", "d1", "jev"):
            rows.append({"set": f"{label} doubtful DGA domains, selection sample removed", "cue": cue,
                         "n_wl": int(is_wl.sum()), "n_nw": int((1 - is_wl).sum()), "auc_wl_vs_nw": roc_auc_score(is_wl, frame[cue])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "revision_cue_auc.csv", index=False)
    print("[2]", table.to_string(index=False))


def seed_bootstrap(y, family, preds_a, preds_b, n_boot=1000, seed=7, metric="f1") -> dict[str, float]:
    """Paired bootstrap that resamples families and benign domains and, for a side with several routers, one router per replicate."""
    y = np.asarray(y, int)
    fam = np.asarray(family).astype(str)
    pos, neg = y == 1, y == 0
    _, code = np.unique(fam[pos], return_inverse=True)
    n_fam = code.max() + 1
    size = np.bincount(code, minlength=n_fam).astype(float)

    def counts(preds):
        rows = [np.asarray(p, int) for p in preds]
        return np.stack([np.bincount(code, weights=r[pos], minlength=n_fam) for r in rows]), np.stack([r[neg].astype(float) for r in rows])

    tp_a, fp_a = counts(preds_a)
    tp_b, fp_b = counts(preds_b)
    n_neg = fp_a.shape[1]

    def value(tp, fp, fam_w, neg_w):
        t, f, s = tp @ fam_w, fp @ neg_w, size @ fam_w
        return _f1(t, f, s - t) if metric == "f1" else f / neg_w.sum()

    point = float(np.mean(value(tp_a, fp_a, np.ones(n_fam), np.ones(n_neg))) - np.mean(value(tp_b, fp_b, np.ones(n_fam), np.ones(n_neg))))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        fam_w = np.bincount(rng.integers(0, n_fam, n_fam), minlength=n_fam).astype(float) if metric == "f1" else np.ones(n_fam)
        neg_w = np.bincount(rng.integers(0, n_neg, n_neg), minlength=n_neg).astype(float)
        ia, ib = rng.integers(0, len(preds_a)), rng.integers(0, len(preds_b))
        diffs[r] = float(value(tp_a[ia:ia + 1], fp_a[ia:ia + 1], fam_w, neg_w)[0] - value(tp_b[ib:ib + 1], fp_b[ib:ib + 1], fam_w, neg_w)[0])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "low": float(low), "high": float(high)}


def router_seed_intervals(test: pd.DataFrame, systems: dict) -> None:
    old = {"f1": pd.read_csv(OUT / "test_v2_all_pairs.csv"), "fpr": pd.read_csv(OUT / "test_v2_all_pairs_fpr.csv")}
    rows = []
    for a, b in itertools.combinations(systems, 2):
        if not {"C4", "C5"} & {a, b}:
            continue
        for metric in ("f1", "fpr"):
            r = seed_bootstrap(test["y_true"], test["family"], systems[a], systems[b], metric=metric)
            prev = old[metric].query("a == @a and b == @b").iloc[0]
            rows.append({"a": a, "b": b, "metric": metric, "diff": r["diff"], "low": r["low"], "high": r["high"],
                         "clear": bool(r["low"] > 0 or r["high"] < 0), "mean_router_low": prev["low"],
                         "mean_router_high": prev["high"], "mean_router_clear": bool(prev["clear"])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "revision_router_seed_intervals.csv", index=False)
    changed = table[table["clear"] != table["mean_router_clear"]]
    print(f"[3] {len(table)} intervals with C4 or C5; clear flag changed in {len(changed)}")
    if len(changed):
        print(changed.to_string(index=False))


def train_router_masked(train: pd.DataFrame, features: list[dict], drop: np.ndarray, seed: int, fp_weight: float):
    """cacic.train_router with some router-training rows left out."""
    target, resolved = cheapest_correct_targets(train)
    keep = np.flatnonzero(resolved.to_numpy() & ~drop)
    weights = sample_weights(train)
    if fp_weight:
        any_fp = pd.concat([train[f"{m}__pred"].astype(int).eq(1) for m in cacic.MODELS], axis=1).any(axis=1)
        weights = weights * np.where(train["y_true"].astype(int).eq(0) & any_fp, 1.0 + fp_weight, 1.0)
    model = make_router(seed)
    model.fit([features[i] for i in keep], target.iloc[keep], model__sample_weight=weights.iloc[keep].to_numpy())
    return model


def routers_without_overlap(test: pd.DataFrame, systems: dict, expert_training_csv: Path = EXPERT_TRAIN) -> None:
    train = cacic.load_split("router_train")
    features = cacic._features(train)
    expert = set(pd.read_csv(expert_training_csv, usecols=["domain"])["domain"].astype(str).str.strip().str.lower())
    overlap = train["domain"].astype(str).str.strip().str.lower().isin(expert).to_numpy()
    print(f"[4] router-training domains also in the expert training data: {int(overlap.sum())} of {len(train)}"
          f" ({int((overlap & train['y_true'].eq(1).to_numpy()).sum())} DGA)")
    dev = cacic.load_split("gate1")
    rows, new_test = [], {"C4": [], "C5": []}
    for name, w in (("C4", 0.0), ("C5", 2.0)):
        for seed in cacic.SEEDS:
            for variant, drop in (("original", np.zeros(len(train), bool)), ("no_overlap", overlap)):
                model = train_router_masked(train, features, drop, seed, w)
                for split, wide in (("development", dev), ("test", cacic.load_split("final_test"))):
                    pred = cacic.apply_routes(wide, cacic.router_routes(model, wide), probe_logit=True)[0]
                    f1, fpr = f1_fpr(wide["y_true"], pred)
                    rows.append({"system": name, "seed": seed, "variant": variant, "set": split, "f1": f1, "fpr": fpr})
                    if split == "test" and variant == "original":
                        saved = systems[name][cacic.SEEDS.index(seed)]
                        if not (np.asarray(pred) == np.asarray(saved)).all():
                            raise SystemExit(f"{name} seed {seed}: retrained router does not reproduce the saved test predictions")
                    if split == "test" and variant == "no_overlap":
                        new_test[name].append(pd.Series(np.asarray(pred), index=test.index))
    table = pd.DataFrame(rows)
    summary = table.groupby(["system", "variant", "set"])[["f1", "fpr"]].agg(["mean", "std"]).round(4)
    for name in ("C4", "C5"):
        r = paired_f1_bootstrap_multi(test["y_true"], test["family"], new_test[name], systems[name], n_boot=1000, seed=7)
        table = pd.concat([table, pd.DataFrame([{"system": name, "seed": "mean5", "variant": "no_overlap_minus_original",
                                                 "set": "test", "f1": r["diff"], "fpr": np.nan, "low": r["low"], "high": r["high"]}])])
    table.to_csv(OUT / "revision_router_no_overlap.csv", index=False)
    print(summary.to_string())
    print(table[table["seed"].eq("mean5")].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=["control", "auc", "intervals", "overlap"],
                        default=["control", "auc", "intervals", "overlap"])
    parser.add_argument("--expert-training-csv", type=Path, default=EXPERT_TRAIN,
                        help="Expert training domains, or the released exact overlap index (default).")
    args = parser.parse_args()
    if "overlap" in args.checks and not args.expert_training_csv.is_file():
        parser.error("Supply --expert-training-csv or use the released data/router_expert_overlap.csv")
    test, wide, systems = load_test()
    if "control" in args.checks:
        doubtful_random_control(test, wide, systems)
    if "auc" in args.checks:
        cue_auc()
    if "intervals" in args.checks:
        router_seed_intervals(test, systems)
    if "overlap" in args.checks:
        routers_without_overlap(test, systems, args.expert_training_csv)


if __name__ == "__main__":
    main()
