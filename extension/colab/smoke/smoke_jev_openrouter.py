#!/usr/bin/env python3
"""Smoke test of Jev (TypeSafe, via OpenRouter) on the same 400 development domains as the local models.

Same typed questions, same state format and both option orders as smoke_run.py.
Latency is the API round trip from this machine: it is not comparable with the T4 numbers.
The OpenRouter key is read from opencode's auth file and is never printed or saved.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MODEL = "typesafe/jev-1.14"
ENDPOINTS = ("https://openrouter.ai/api/v1/systemone", "https://openrouter.ai/api/alpha/decisions")
KEY = json.load(open(Path.home() / ".local/share/opencode/auth.json"))["openrouter"]["key"]

WL_A, WL_B = "Real English words joined together", "Random-looking letters or digits"
DGA_A, DGA_B = "Malware, using a domain generation algorithm", "A person or company, for a real website or service"


def questions(swapped: bool) -> dict:
    wl = {"a": WL_B, "b": WL_A} if swapped else {"a": WL_A, "b": WL_B}
    dga = {"a": DGA_B, "b": DGA_A} if swapped else {"a": DGA_A, "b": DGA_B}
    return {
        "wl": {"type": "choice", "instructions": "Look only at the domain name. How is its main part built?", "criteria": wl},
        "dga": {"type": "choice", "instructions": "Look only at the domain name. Who most likely created it?", "criteria": dga},
    }


def prob_of(answer: dict, key: str) -> float:
    if "probabilities" in answer:
        return float(answer["probabilities"][key])
    conf = float(answer["confidence"])
    return conf if answer["choice"] == key else 1.0 - conf


def call(endpoint: str, domain: str, q: dict) -> tuple[dict, float]:
    body = json.dumps({"model": MODEL, "state": f"Domain name: {domain}", "questions": q}).encode()
    req = urllib.request.Request(endpoint, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data, (time.perf_counter() - t0) * 1000.0
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from None
    raise RuntimeError("unreachable")


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    frame = pd.read_csv(HERE / "smoke_domains.csv")
    domains = frame["domain"].tolist()
    endpoint = None
    for candidate in ENDPOINTS:
        try:
            first, _ = call(candidate, domains[0], questions(False))
            endpoint = candidate
            break
        except RuntimeError as exc:
            print("endpoint", candidate, "failed:", str(exc)[:200], flush=True)
    if endpoint is None:
        raise SystemExit("no Jev endpoint worked")
    print("endpoint:", endpoint)
    print("example raw answer:", json.dumps(first.get("answers"), default=str)[:500], flush=True)

    rows, cost = [], 0.0
    for swapped in (False, True):
        q = questions(swapped)
        key = "b" if swapped else "a"
        for d in domains:
            data, ms = call(endpoint, d, q)
            ans = data["answers"]
            cost += float(data.get("usage", {}).get("cost", 0.0) or 0.0)
            rows.append({"domain": d, "swapped": swapped, "p_wl": prob_of(ans["wl"], key), "p_dga": prob_of(ans["dga"], key), "latency_ms": ms})
    res = pd.DataFrame(rows)
    wide = res[~res.swapped].set_index("domain")[["p_wl", "p_dga", "latency_ms"]].join(
        res[res.swapped].set_index("domain")[["p_wl", "p_dga"]].add_suffix("_swapped"))
    out = frame.merge(wide.reset_index(), on="domain")
    out["model"] = "jev"
    out.to_csv(OUT / "smoke_out_jev.csv", index=False)

    dga = out[out.dga_type.isin(["WL", "NW"])]
    wl, nw = dga.dga_type.eq("WL"), dga.dga_type.eq("NW")
    meta = {
        "model": MODEL,
        "endpoint": endpoint,
        "rows": int(len(out)),
        "latency_ms_api_mean": float(out["latency_ms"].mean()),
        "latency_ms_api_p50": float(out["latency_ms"].median()),
        "auc_wl_vs_nw_wl_option_first": auc(dga.loc[wl, "p_wl"], dga.loc[nw, "p_wl"]),
        "auc_wl_vs_nw_random_option_first": auc(dga.loc[wl, "p_wl_swapped"], dga.loc[nw, "p_wl_swapped"]),
        "auc_dga_vs_benign_wl_option_first": auc(out.loc[out.y_true == 1, "p_dga"], out.loc[out.y_true == 0, "p_dga"]),
        "auc_dga_vs_benign_random_option_first": auc(out.loc[out.y_true == 1, "p_dga_swapped"], out.loc[out.y_true == 0, "p_dga_swapped"]),
        "flip_rate_wl": float(((out.p_wl >= 0.5) != (out.p_wl_swapped >= 0.5)).mean()),
        "flip_rate_dga": float(((out.p_dga >= 0.5) != (out.p_dga_swapped >= 0.5)).mean()),
        "cost_usd": cost,
    }
    json.dump(meta, open(OUT / "smoke_meta_jev.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
