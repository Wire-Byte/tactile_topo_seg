#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/tactile_topo_seg
mkdir -p work_dirs/table2_topology logs/table2_topology

run_one() {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local pred_dir="work_dirs/table2_topology/${name}_preds"
  local out_dir="work_dirs/table2_topology/${name}_topo"
  local log="logs/table2_topology/${name}.log"

  echo "[START] ${name}" | tee -a "$log"
  /root/miniconda3/envs/tpseg/bin/python -u tools/export_preds_unique_by_index.py \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --output-dir "$pred_dir" \
    >> "$log" 2>&1

  /root/miniconda3/envs/tpseg/bin/python -u tools/eval_tp_topology.py \
    --pred-dir "$pred_dir" \
    --gt-dir data/TP-Dataset/GroundTruth \
    --index-file data/TP-Dataset/Index/val.txt \
    --out-dir "$out_dir" \
    >> "$log" 2>&1

  echo "[DONE] ${name}" | tee -a "$log"
}

run_one core_innovation configs/tp_dataset/ablation_generated/core_both.py work_dirs/ablation/core_both/ckpt/core_both/best_mIoU_iter_9500.pth
run_one deeplabv3 configs/tp_dataset/deeplabv3plus_r50_tp_main_table1_B.py work_dirs/deeplabv3plus_r50_tp_main_table1_B/ckpt/deeplabv3plus_r50_tp_main_table1_B/best_mIoU_iter_9000.pth
run_one pspnet configs/tp_dataset/pspnet_r50_tp_main_table1_B.py work_dirs/pspnet_r50_tp_main_table1_B/ckpt/pspnet_r50_tp_main_table1_B/best_mIoU_iter_9000.pth
run_one upernet configs/tp_dataset/upernet_r50_tp_main_table1_B.py work_dirs/upernet_r50_tp_main_table1_B/ckpt/upernet_r50_tp_main_table1_B/best_mIoU_iter_6000.pth

echo "ALL_DONE"
