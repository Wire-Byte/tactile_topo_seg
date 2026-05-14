#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/tactile_topo_seg
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}"
mkdir -p logs/yellowblock/revise work_dirs/yellowblock/revise docs/yellowblock/revise

run_tpseg() {
  local cfg="$1"
  conda activate tpseg
  python -u tools/yellowblock/train_revise_external_yellowblock.py --config "$cfg"
}

run_vmunet() {
  local cfg="$1"
  conda activate vmunet
  python -u tools/yellowblock/train_revise_external_yellowblock.py --config "$cfg"
}

run_tpseg configs/yellowblock/emcad_yellowblock_revise_v2_pretrain.py 2>&1 | tee logs/yellowblock/revise/train_emcad_yellowblock_revise_v2_pretrain.log
run_tpseg configs/yellowblock/bevanet_s_yellowblock_revise_v2_pretrain.py 2>&1 | tee logs/yellowblock/revise/train_bevanet_s_yellowblock_revise_v2_pretrain.log
run_vmunet configs/yellowblock/vmunet_yellowblock_revise_v2_pretrain.py 2>&1 | tee logs/yellowblock/revise/train_vmunet_yellowblock_revise_v2_pretrain.log
