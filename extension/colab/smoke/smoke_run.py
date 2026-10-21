# Smoke test, step 2: run one decision model on 400 development domains on the T4.
# Two typed questions per domain (WL and DGA), each also with the option order swapped.
# Reads /content/smoke_cfg.json; writes /content/smoke_out_<model>.csv and smoke_meta_<model>.json.
import json
import os
import platform
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch

cfg = json.load(open("/content/smoke_cfg.json"))
MODEL = cfg["model"]
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    if "confidence" in answer and "choice" in answer:
        conf = float(answer["confidence"])
        return conf if answer["choice"] == key else 1.0 - conf
    raise KeyError(f"cannot read probabilities from {answer}")


def state_of(domain: str) -> str:
    return f"Domain name: {domain}"


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_used = "n/a"
batch_fn = None

if MODEL == "julia":
    from julia import load_model

    try:
        engine = load_model("/content/Julia-1", device=device, strict_encoding=True, max_length=8192, head_length=512)
    except Exception as exc:  # BF16 not available on T4: fall back to CPU
        print("julia on cuda failed:", repr(exc)[:300], flush=True)
        device = "cpu"
        engine = load_model("/content/Julia-1", device="cpu", strict_encoding=True, max_length=8192, head_length=512)

    def ask(domain, q):
        return engine.predict(state=state_of(domain), questions=q)["answers"]
elif MODEL == "laya":
    import laya

    try:
        agent = laya.load("convaiinnovations/laya", device=device)
    except TypeError:
        agent = laya.load("convaiinnovations/laya")

    def ask(domain, q):
        return agent.predict(state_of(domain), q)["answers"]
elif MODEL == "d1":
    from transformers import AutoModel

    for dt in (torch.float16, torch.float32):
        model = AutoModel.from_pretrained("LiquidAI/d1-3B", trust_remote_code=True, dtype=dt).to(device).eval()
        probe = model.system_one(state_of("example.com"), questions(False))["answers"]
        if np.isfinite(prob_of(probe["wl"], "a")):
            dtype_used = str(dt)
            break
        del model
        torch.cuda.empty_cache()

    def ask(domain, q):
        return model.system_one(state_of(domain), q)["answers"]

    def batch_fn(domains, q):
        return [r["answers"] for r in model.system_one_batch([(state_of(d), q) for d in domains])]
else:
    raise SystemExit(f"unknown model {MODEL}")

frame = pd.read_csv("/content/smoke_domains.csv")
domains = frame["domain"].tolist()
print("example raw answer:", json.dumps(ask(domains[0], questions(False)), default=str)[:600], flush=True)

for _ in range(2):  # two warm-up rounds, as in the expert runs
    for d in domains[:8]:
        ask(d, questions(False))

rows = []
for swapped in (False, True):
    q = questions(swapped)
    key_wl, key_dga = ("b", "b") if swapped else ("a", "a")
    for d in domains:
        t0 = time.perf_counter()
        ans = ask(d, q)
        ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"domain": d, "swapped": swapped, "p_wl": prob_of(ans["wl"], key_wl), "p_dga": prob_of(ans["dga"], key_dga), "latency_ms": ms})
res = pd.DataFrame(rows)
wide = res[~res.swapped].set_index("domain")[["p_wl", "p_dga", "latency_ms"]].join(
    res[res.swapped].set_index("domain")[["p_wl", "p_dga"]].add_suffix("_swapped")
)

batched_ms = np.nan
if batch_fn is not None:
    q = questions(False)
    t0 = time.perf_counter()
    for start in range(0, len(domains), 64):
        batch_fn(domains[start:start + 64], q)
    batched_ms = (time.perf_counter() - t0) * 1000.0 / len(domains)

out = frame.merge(wide.reset_index(), on="domain")
out["model"] = MODEL
out.to_csv(f"/content/smoke_out_{MODEL}.csv", index=False)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


dga = out[out.dga_type.isin(["WL", "NW"])]
meta = {
    "model": MODEL,
    "device": device,
    "dtype": dtype_used,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "rows": int(len(out)),
    "latency_ms_batch1_mean": float(out["latency_ms"].mean()),
    "latency_ms_batch1_p50": float(out["latency_ms"].median()),
    "latency_ms_batched_64": batched_ms,
    "auc_wl_vs_nw": auc(dga[dga.dga_type == "WL"].p_wl, dga[dga.dga_type == "NW"].p_wl),
    "auc_dga_vs_benign": auc(out[out.y_true == 1].p_dga, out[out.y_true == 0].p_dga),
    "flip_rate_wl": float(((out.p_wl >= 0.5) != (out.p_wl_swapped >= 0.5)).mean()),
    "flip_rate_dga": float(((out.p_dga >= 0.5) != (out.p_dga_swapped >= 0.5)).mean()),
}
json.dump(meta, open(f"/content/smoke_meta_{MODEL}.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
print("RUN_OK", MODEL)
