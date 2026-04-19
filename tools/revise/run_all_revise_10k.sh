#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

mkdir -p logs/revise work_dirs/revise docs/revise

PY="/root/miniconda3/envs/tpseg/bin/python"

run_one() {
  local cfg="$1"
  local tag="$2"
  local log="logs/revise/${tag}.log"
  echo "[RUN] ${cfg} -> ${log}"
  PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}" \
    "$PY" -u tools/revise/train_revise_external.py --config "$cfg" 2>&1 | tee "$log"
}

run_one configs/tp_dataset/vmunet_tp_revise.py vmunet_tp_revise_10k
run_one configs/tp_dataset/emcad_tp_revise.py emcad_tp_revise_10k
run_one configs/tp_dataset/bevanet_s_tp_revise.py bevanet_s_tp_revise_10k

PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}" \
  "$PY" -u tools/revise/build_revise_tables.py

echo "[DONE] revised tables in docs/revise"
