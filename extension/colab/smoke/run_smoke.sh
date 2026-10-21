#!/usr/bin/env bash
# Usage: run_smoke.sh MODEL   (julia | laya | d1)
# One T4 session per model: upload -> install -> restart kernel -> run -> download -> stop.
set -u
MODEL="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
S="smoke-$MODEL"
OUT="$HERE/out"
mkdir -p "$OUT"
printf '{"model": "%s"}\n' "$MODEL" > "$OUT/cfg_$MODEL.json"

step() { echo "== $*"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "FAILED rc=$rc: $*"; colab stop -s "$S"; exit $rc; fi; }

step colab new -s "$S" --gpu T4
step colab upload -s "$S" "$HERE/smoke_domains.csv" /content/smoke_domains.csv
step colab upload -s "$S" "$OUT/cfg_$MODEL.json" /content/smoke_cfg.json
step colab exec -s "$S" -f "$HERE/smoke_install.py" --timeout 1500
step colab restart-kernel -s "$S"
step colab exec -s "$S" -f "$HERE/smoke_run.py" --timeout 2400
step colab download -s "$S" "/content/smoke_out_$MODEL.csv" "$OUT/smoke_out_$MODEL.csv"
step colab download -s "$S" "/content/smoke_meta_$MODEL.json" "$OUT/smoke_meta_$MODEL.json"
colab stop -s "$S"
echo "SMOKE_DONE $MODEL"
