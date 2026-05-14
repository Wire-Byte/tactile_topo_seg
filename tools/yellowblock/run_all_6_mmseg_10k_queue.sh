#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/tactile_topo_seg
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="$ROOT:$ROOT/third_party/mmsegmentation:${PYTHONPATH:-}"
mkdir -p logs/yellowblock/mmseg work_dirs/yellowblock/mmseg

START_INDEX=${START_INDEX:-1}
MAX_RETRIES=${MAX_RETRIES:-3}
TIMEOUT_SECS=${TIMEOUT_SECS:-28800}
GPU_WAIT_SECS=${GPU_WAIT_SECS:-180}
GPU_MEM_THRESHOLD_MB=${GPU_MEM_THRESHOLD_MB:-1024}
GPU_FREE_MIN_MB=${GPU_FREE_MIN_MB:-4096}
NO_LOG_PROGRESS_SECS=${NO_LOG_PROGRESS_SECS:-900}
TRAIN_ITERS=${TRAIN_ITERS:-10000}
VAL_INTERVAL=${VAL_INTERVAL:-1000}
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
LOGGER_INTERVAL=${LOGGER_INTERVAL:-10}

cleanup_residual_by_cfg() {
  local cfg="$1"
  local pids
  pids=$(pgrep -f "$cfg" || true)
  if [[ -n "$pids" ]]; then
    echo "[guard] Killing residual PIDs for $cfg: $pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids=$(pgrep -f "$cfg" || true)
    if [[ -n "$pids" ]]; then
      echo "[guard] Force killing residual PIDs for $cfg: $pids"
      kill -KILL $pids 2>/dev/null || true
    fi
  fi
}

wait_gpu_ready() {
  local waited=0
  local used
  local total
  local free
  local app_lines
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')

    if ! [[ "$used" =~ ^[0-9]+$ ]] || ! [[ "$total" =~ ^[0-9]+$ ]] || (( total <= 0 )); then
      echo "[guard] failed to query GPU memory reliably (used='${used:-NA}' total='${total:-NA}'), waiting..."
      sleep 5
      waited=$((waited + 5))
      if (( waited >= GPU_WAIT_SECS )); then
        echo "[guard] GPU query still invalid after ${GPU_WAIT_SECS}s"
        return 1
      fi
      continue
    fi

    free=$((total - used))
    app_lines=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    if (( free >= GPU_FREE_MIN_MB )); then
      if (( used <= GPU_MEM_THRESHOLD_MB )); then
        break
      fi
      if [[ -n "$app_lines" ]] && echo "$app_lines" | grep -qi '\[Not Found\]'; then
        echo "[guard] free GPU memory is sufficient and only ghost contexts are visible; proceeding with launch."
        break
      fi
    fi

    echo "[guard] GPU used=${used}MiB total=${total}MiB free=${free}MiB (need used<=${GPU_MEM_THRESHOLD_MB} and free>=${GPU_FREE_MIN_MB}), waiting..."
    if [[ -n "$app_lines" ]]; then
      echo "$app_lines"
    elif (( used > GPU_MEM_THRESHOLD_MB )); then
      echo "[guard] WARNING: GPU memory is busy but no compute-app process is visible (possible leaked/ghost context)."
    fi
    sleep 5
    waited=$((waited + 5))
    if (( waited >= GPU_WAIT_SECS )); then
      echo "[guard] GPU still busy after ${GPU_WAIT_SECS}s"
      return 1
    fi
  done
}

run_with_guard() {
  local name="$1"
  local env_name="$2"
  local cfg="$3"
  local log_file="$4"
  local train_cmd="$5"

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "============================================================" | tee -a "$log_file"
    echo "[queue] model=${name} attempt=${attempt}/${MAX_RETRIES}" | tee -a "$log_file"
    echo "[queue] cfg=${cfg}" | tee -a "$log_file"
    cleanup_residual_by_cfg "$cfg"

    if ! wait_gpu_ready; then
      echo "[queue] GPU not ready before launch, aborting ${name}" | tee -a "$log_file"
      return 1
    fi

    conda activate "$env_name"
    local rc=0
    local start_ts
    local last_progress_ts
    local now_ts
    local mtime
    start_ts=$(date +%s)
    last_progress_ts=$start_ts

    set +e
    setsid bash -lc "$train_cmd" >>"$log_file" 2>&1 &
    local cmd_pid=$!
    local pgid
    pgid=$(ps -o pgid= "$cmd_pid" 2>/dev/null | tr -d '[:space:]' || true)

    while kill -0 "$cmd_pid" 2>/dev/null; do
      now_ts=$(date +%s)
      mtime=$(stat -c %Y "$log_file" 2>/dev/null || echo 0)

      if [[ "$mtime" =~ ^[0-9]+$ ]] && (( mtime > 0 )) && (( mtime >= last_progress_ts )); then
        last_progress_ts=$mtime
      fi

      if (( now_ts - start_ts > TIMEOUT_SECS )); then
        echo "[queue] ${name} hit wall-time timeout ${TIMEOUT_SECS}s; killing..." | tee -a "$log_file"
        if [[ -n "$pgid" ]]; then
          kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
        kill -TERM "$cmd_pid" 2>/dev/null || true
        sleep 3
        if [[ -n "$pgid" ]]; then
          kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
        kill -KILL "$cmd_pid" 2>/dev/null || true
        rc=124
        break
      fi

      if (( now_ts - last_progress_ts > NO_LOG_PROGRESS_SECS )); then
        echo "[queue] ${name} no log progress for ${NO_LOG_PROGRESS_SECS}s; killing for restart..." | tee -a "$log_file"
        if [[ -n "$pgid" ]]; then
          kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
        kill -TERM "$cmd_pid" 2>/dev/null || true
        sleep 3
        if [[ -n "$pgid" ]]; then
          kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
        kill -KILL "$cmd_pid" 2>/dev/null || true
        rc=125
        break
      fi

      sleep 10
    done

    if [[ $rc -eq 0 ]]; then
      wait "$cmd_pid"
      rc=$?
    else
      wait "$cmd_pid" 2>/dev/null || true
    fi
    set -e

    if [[ $rc -eq 0 ]]; then
      echo "[queue] ${name} finished successfully" | tee -a "$log_file"
      return 0
    fi

    if grep -Eiq 'cuda out of memory|outofmemoryerror|cudnn_status_alloc_failed|resourceexhaustederror' "$log_file"; then
      echo "[queue] ${name} hit OOM/resource exhaustion; stop retrying to avoid useless loop." | tee -a "$log_file"
      return 1
    fi

    if [[ $rc -eq 124 ]]; then
      echo "[queue] ${name} timed out after ${TIMEOUT_SECS}s; restarting..." | tee -a "$log_file"
    elif [[ $rc -eq 125 ]]; then
      echo "[queue] ${name} restarted because log stalled for ${NO_LOG_PROGRESS_SECS}s..." | tee -a "$log_file"
    else
      echo "[queue] ${name} failed with code=${rc}; restarting..." | tee -a "$log_file"
    fi

    cleanup_residual_by_cfg "$cfg"
    attempt=$((attempt + 1))
  done

  echo "[queue] ${name} failed after ${MAX_RETRIES} attempts" | tee -a "$log_file"
  return 1
}

run_mmseg_tpseg() {
  local name="$1"
  local cfg="$2"
  local log_file="$3"
  local run_cfg="$cfg"
  local extra_opts=""

  if [[ "$name" == "segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable" ]]; then
    run_cfg="configs/yellowblock/segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable_nocrop.py"
    extra_opts=" train_dataloader.batch_size=1 train_dataloader.num_workers=0 train_dataloader.persistent_workers=False val_dataloader.num_workers=0 val_dataloader.persistent_workers=False test_dataloader.num_workers=0 test_dataloader.persistent_workers=False"
  elif [[ "$name" == "segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4" ]]; then
    run_cfg="configs/yellowblock/segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4_nocrop.py"
    extra_opts=" train_dataloader.batch_size=1 train_dataloader.num_workers=0 train_dataloader.persistent_workers=False val_dataloader.num_workers=0 val_dataloader.persistent_workers=False test_dataloader.num_workers=0 test_dataloader.persistent_workers=False"
  fi

  local cmd="cd '$ROOT' && source /root/miniconda3/etc/profile.d/conda.sh && conda activate tpseg && export PYTHONPATH='$ROOT:$ROOT/third_party/mmsegmentation:\${PYTHONPATH:-}' && python -u third_party/mmsegmentation/tools/train.py '$run_cfg' --launcher none --cfg-options train_cfg.max_iters=${TRAIN_ITERS} train_cfg.val_interval=${VAL_INTERVAL} default_hooks.checkpoint.interval=${CKPT_INTERVAL} default_hooks.logger.interval=${LOGGER_INTERVAL}${extra_opts}"
  run_with_guard "$name" tpseg "$run_cfg" "$log_file" "$cmd"
}

declare -a MODEL_NAMES=(
  "segformer_b2_yellowblock"
  "segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable"
  "segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4"
  "pspnet_r50_yellowblock_main_table1_B"
  "upernet_r50_yellowblock_main_table1_B"
  "deeplabv3plus_r50_yellowblock_main_table1_B"
)

declare -a MODEL_CFGS=(
  "configs/yellowblock/segformer_b2_yellowblock.py"
  "configs/yellowblock/segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable.py"
  "configs/yellowblock/segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4.py"
  "configs/yellowblock/pspnet_r50_yellowblock_main_table1_B.py"
  "configs/yellowblock/upernet_r50_yellowblock_main_table1_B.py"
  "configs/yellowblock/deeplabv3plus_r50_yellowblock_main_table1_B.py"
)

declare -a MODEL_LOGS=(
  "logs/yellowblock/mmseg/train_segformer_b2_yellowblock_10k.log"
  "logs/yellowblock/mmseg/train_segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable_10k.log"
  "logs/yellowblock/mmseg/train_segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4_10k.log"
  "logs/yellowblock/mmseg/train_pspnet_r50_yellowblock_main_table1_B_10k.log"
  "logs/yellowblock/mmseg/train_upernet_r50_yellowblock_main_table1_B_10k.log"
  "logs/yellowblock/mmseg/train_deeplabv3plus_r50_yellowblock_main_table1_B_10k.log"
)

if (( START_INDEX < 1 || START_INDEX > ${#MODEL_NAMES[@]} )); then
  echo "START_INDEX out of range: ${START_INDEX}"
  exit 1
fi

echo "[queue] Starting 6-model mmseg queue from model index ${START_INDEX}"
echo "[queue] TRAIN_ITERS=${TRAIN_ITERS} VAL_INTERVAL=${VAL_INTERVAL} CKPT_INTERVAL=${CKPT_INTERVAL} LOGGER_INTERVAL=${LOGGER_INTERVAL}"
echo "[queue] MAX_RETRIES=${MAX_RETRIES} TIMEOUT_SECS=${TIMEOUT_SECS} NO_LOG_PROGRESS_SECS=${NO_LOG_PROGRESS_SECS} GPU_WAIT_SECS=${GPU_WAIT_SECS} GPU_MEM_THRESHOLD_MB=${GPU_MEM_THRESHOLD_MB} GPU_FREE_MIN_MB=${GPU_FREE_MIN_MB}"

for ((i = START_INDEX - 1; i < ${#MODEL_NAMES[@]}; i++)); do
  name="${MODEL_NAMES[$i]}"
  cfg="${MODEL_CFGS[$i]}"
  log_file="${MODEL_LOGS[$i]}"

  echo "[queue] >>> ($((i + 1))/${#MODEL_NAMES[@]}) ${name}"
  run_mmseg_tpseg "$name" "$cfg" "$log_file"
done

echo "[queue] All 6 mmseg models finished"
