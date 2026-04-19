#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/tactile_topo_seg
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}"
mkdir -p logs/revise work_dirs/revise docs/revise

run_tpseg() {
  local cfg="$1"
  conda activate tpseg
  python -u tools/revise/train_revise_external.py --config "$cfg"
}

run_vmunet() {
  local cfg="$1"
  conda activate vmunet
  python -u tools/revise/train_revise_external.py --config "$cfg"
}

run_tpseg configs/tp_dataset/emcad_tp_revise_v2_pretrain.py 2>&1 | tee logs/revise/train_emcad_tp_revise_v2_pretrain.log
run_tpseg configs/tp_dataset/bevanet_s_tp_revise_v2_pretrain.py 2>&1 | tee logs/revise/train_bevanet_s_tp_revise_v2_pretrain.log
run_vmunet configs/tp_dataset/vmunet_tp_revise_v2_pretrain.py 2>&1 | tee logs/revise/train_vmunet_tp_revise_v2_pretrain.log
