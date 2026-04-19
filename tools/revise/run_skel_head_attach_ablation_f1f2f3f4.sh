#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="/root/miniconda3/envs/tpseg/bin/python"
mkdir -p logs/revise docs/revise work_dirs

run_one() {
  local cfg="$1"
  local tag="$2"
  local log="logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_${tag}.log"
  echo "[RUN] ${cfg} -> ${log}"
  PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}" \
    "$PY" -u third_party/mmsegmentation/tools/train.py "$cfg" --launcher none 2>&1 | tee "$log"
}

run_one configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F1.py F1
run_one configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4.py F2
run_one configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F3.py F3
run_one configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F4.py F4

PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}" \
  "$PY" -u tools/revise/build_fpn_attach_ablation_summary.py

echo "[DONE] logs in logs/revise"
echo "[DONE] summary in docs/revise"
