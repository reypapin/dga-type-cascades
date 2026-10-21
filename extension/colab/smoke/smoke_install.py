# Smoke test, step 1: install one decision model on the Colab VM.
# Reads /content/smoke_cfg.json -> {"model": "julia" | "laya" | "d1"}.
import json
import subprocess
import sys

cfg = json.load(open("/content/smoke_cfg.json"))
model = cfg["model"]


def pip(*args):
    print("pip", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


if model == "julia":
    pip("huggingface_hub")
    from huggingface_hub import snapshot_download

    snapshot_download("SupersonicLabs/Julia-1", local_dir="/content/Julia-1")
    pip("-e", "/content/Julia-1")
elif model == "laya":
    pip("laya")
elif model == "d1":
    pip("transformers>=5.14")
else:
    raise SystemExit(f"unknown model {model}")

out = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout
print("\n".join(line for line in out.splitlines() if line.split()[0].lower() in {"torch", "transformers", "laya", "julia", "huggingface-hub"}))
print("INSTALL_OK", model)
