#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Monitor clDice training milestones and auto-run final tri-model comparison.

Features:
1) Poll training log and capture milestone snapshots at 1000/5000/10000 iters
   including train loss and validation summary metrics.
2) After training reaches 10000, run:
   - official MMSeg test for baseline/skel/skel+clDice
   - unique prediction export for all three models
   - topology evaluation for all three models
3) Write merged summary CSV for quick comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


MILESTONES = (1000, 5000, 10000)


def run_cmd(cmd, cwd: Path, env: Dict[str, str], log_file: Optional[Path] = None) -> str:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    writer = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        writer = log_file.open("w", encoding="utf-8")

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if writer is not None:
            writer.write(line)

    if writer is not None:
        writer.close()

    ret = proc.wait()
    out = "".join(lines)
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}\n{out[-4000:]}")
    return out


def parse_train_line_for_iter(log_text: str, it: int) -> Optional[Dict[str, float]]:
    pat = re.compile(
        rf"Iter\(train\)\s*\[\s*{it}/10000\].*?loss:\s*([0-9.]+).*?"
        r"decode\.loss_ce:\s*([0-9.]+).*?decode\.loss_dice:\s*([0-9.]+).*?"
        r"decode\.loss_cldice:\s*([0-9.]+).*?aux\.loss_skel_bce:\s*([0-9.]+).*?"
        r"aux\.loss_skel_dice:\s*([0-9.]+).*?aux\.loss_skel_cldice:\s*([0-9.]+)",
        flags=re.S,
    )
    m = pat.search(log_text)
    if not m:
        return None
    return {
        "loss": float(m.group(1)),
        "decode.loss_ce": float(m.group(2)),
        "decode.loss_dice": float(m.group(3)),
        "decode.loss_cldice": float(m.group(4)),
        "aux.loss_skel_bce": float(m.group(5)),
        "aux.loss_skel_dice": float(m.group(6)),
        "aux.loss_skel_cldice": float(m.group(7)),
    }


def parse_val_after_iter(log_text: str, it: int) -> Optional[Dict[str, float]]:
    marker = re.search(rf"Saving checkpoint at\s+{it}\s+iterations", log_text)
    if not marker:
        return None

    tail = log_text[marker.end():]
    m = re.search(
        r"Iter\(val\)\s*\[281/281\].*?aAcc:\s*([0-9.]+).*?mIoU:\s*([0-9.]+).*?"
        r"mAcc:\s*([0-9.]+).*?mDice:\s*([0-9.]+).*?mFscore:\s*([0-9.]+).*?"
        r"mPrecision:\s*([0-9.]+).*?mRecall:\s*([0-9.]+)",
        tail,
        flags=re.S,
    )
    if not m:
        return None

    return {
        "aAcc": float(m.group(1)),
        "mIoU": float(m.group(2)),
        "mAcc": float(m.group(3)),
        "mDice": float(m.group(4)),
        "mFscore": float(m.group(5)),
        "mPrecision": float(m.group(6)),
        "mRecall": float(m.group(7)),
    }


def parse_tactile_row(log_text: str) -> Dict[str, float]:
    m = re.search(
        r"\|\s*tactile_paving\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*"
        r"([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        log_text,
    )
    if not m:
        raise RuntimeError("Cannot parse tactile_paving row from test log output.")
    return {
        "tactile_iou": float(m.group(1)),
        "tactile_acc": float(m.group(2)),
        "tactile_dice": float(m.group(3)),
        "tactile_fscore": float(m.group(4)),
        "tactile_precision": float(m.group(5)),
        "tactile_recall": float(m.group(6)),
    }


def parse_val_summary(log_text: str) -> Dict[str, float]:
    m = re.search(
        r"\[(?:281|\d+)/\d+\].*?aAcc:\s*([0-9.]+).*?mIoU:\s*([0-9.]+).*?mAcc:\s*([0-9.]+).*?"
        r"mDice:\s*([0-9.]+).*?mFscore:\s*([0-9.]+).*?mPrecision:\s*([0-9.]+).*?mRecall:\s*([0-9.]+)",
        log_text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Cannot parse summary mIoU/mDice metrics from test output.")
    return {
        "aAcc": float(m.group(1)),
        "mIoU": float(m.group(2)),
        "mAcc": float(m.group(3)),
        "mDice": float(m.group(4)),
        "mFscore": float(m.group(5)),
        "mPrecision": float(m.group(6)),
        "mRecall": float(m.group(7)),
    }


def parse_pair_official_csv(csv_path: Path, variant: str) -> Dict[str, float]:
    if variant not in {"baseline", "skel"}:
        raise ValueError("variant must be baseline or skel")
    col = "baseline" if variant == "baseline" else "skel"
    out: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["metric"]] = float(row[col])
    return out


def parse_val_metrics_after_checkpoint(log_text: str, it: int) -> Dict[str, float]:
    marker = re.search(rf"Saving checkpoint at\s+{it}\s+iterations", log_text)
    if not marker:
        raise RuntimeError(f"Cannot find checkpoint marker for iter={it} in training log")

    tail = log_text[marker.end():]
    summary = parse_val_summary(tail)
    tactile = parse_tactile_row(tail)
    summary.update(tactile)
    return summary


def find_latest_best(ckpt_dir: Path) -> Path:
    cands = sorted(ckpt_dir.glob("best_mIoU_iter_*.pth"), key=lambda p: int(re.search(r"iter_(\d+)", p.name).group(1)))
    if not cands:
        raise FileNotFoundError(f"No best_mIoU_iter_*.pth under {ckpt_dir}")
    return cands[-1]


def run_official_test(
    root: Path,
    env: Dict[str, str],
    config: Path,
    checkpoint: Path,
    out_log: Path,
) -> Dict[str, float]:
    cmd = [
        sys.executable,
        str(root / "third_party/mmsegmentation/tools/test.py"),
        str(config),
        str(checkpoint),
        "--launcher",
        "none",
    ]
    out = run_cmd(cmd, cwd=root, env=env, log_file=out_log)
    metrics = parse_val_summary(out)
    metrics.update(parse_tactile_row(out))
    return metrics


def run_topology(
    root: Path,
    env: Dict[str, str],
    config: Path,
    checkpoint: Path,
    pred_dir: Path,
    mapping_csv: Path,
    topo_csv: Path,
    topo_summary_csv: Path,
) -> Dict[str, float]:
    export_cmd = [
        sys.executable,
        str(root / "tools/export_preds_unique_by_index.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(pred_dir),
        "--mapping-csv",
        str(mapping_csv),
        "--device",
        "cuda:0",
    ]
    run_cmd(export_cmd, cwd=root, env=env)

    topo_cmd = [
        sys.executable,
        str(root / "tools/eval_tp_topology.py"),
        "--pred-dir",
        str(pred_dir),
        "--out-csv",
        str(topo_csv),
        "--summary-csv",
        str(topo_summary_csv),
    ]
    run_cmd(topo_cmd, cwd=root, env=env)

    with topo_summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    return {
        "topo_samples": float(row["samples"]),
        "topo_mean_dice": float(row["mean_dice"]),
        "topo_mean_iou": float(row["mean_iou"]),
        "topo_mean_clDice": float(row["mean_clDice"]),
        "topo_mean_cc_pred": float(row["mean_cc_pred"]),
        "topo_mean_cc_gt": float(row["mean_cc_gt"]),
        "topo_mean_lcr_pred": float(row["mean_lcr_pred"]),
        "topo_mean_lcr_gt": float(row["mean_lcr_gt"]),
    }


def dump_stage_csv(stage_json: Path, stage_csv: Path) -> None:
    data = json.loads(stage_json.read_text(encoding="utf-8"))
    rows = []
    for k in sorted(data.keys(), key=int):
        item = data[k]
        rows.append({
            "iter": int(k),
            "loss": item.get("train", {}).get("loss"),
            "decode.loss_ce": item.get("train", {}).get("decode.loss_ce"),
            "decode.loss_dice": item.get("train", {}).get("decode.loss_dice"),
            "decode.loss_cldice": item.get("train", {}).get("decode.loss_cldice"),
            "aux.loss_skel_bce": item.get("train", {}).get("aux.loss_skel_bce"),
            "aux.loss_skel_dice": item.get("train", {}).get("aux.loss_skel_dice"),
            "aux.loss_skel_cldice": item.get("train", {}).get("aux.loss_skel_cldice"),
            "mIoU": item.get("val", {}).get("mIoU"),
            "mDice": item.get("val", {}).get("mDice"),
            "mFscore": item.get("val", {}).get("mFscore"),
            "mPrecision": item.get("val", {}).get("mPrecision"),
            "mRecall": item.get("val", {}).get("mRecall"),
        })

    stage_csv.parent.mkdir(parents=True, exist_ok=True)
    with stage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["iter"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor clDice training and auto-run final comparison.")
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stage-json", default="work_dirs/segformer_b2_tp_skel_cldice/stage_monitor.json")
    parser.add_argument("--stage-csv", default="work_dirs/segformer_b2_tp_skel_cldice/stage_monitor.csv")
    parser.add_argument("--final-dir", default="work_dirs/final_compare_b2_tp")
    parser.add_argument("--baseline-config", default="configs/tp_dataset/segformer_b2_tp.py")
    parser.add_argument("--baseline-ckpt", default="work_dirs/segformer_b2_tp/iter_8500.pth")
    parser.add_argument("--skel-config", default="configs/tp_dataset/segformer_b2_tp_skel.py")
    parser.add_argument("--skel-ckpt", default="work_dirs/segformer_b2_tp_skel/ckpt/segformer_b2_tp_skel/best_mIoU_iter_8500.pth")
    parser.add_argument("--cldice-config", default="configs/tp_dataset/segformer_b2_tp_skel_clDice.py")
    parser.add_argument("--cldice-ckpt-dir", default="work_dirs/segformer_b2_tp_skel_cldice/ckpt/segformer_b2_tp_skel_cldice")
    parser.add_argument("--final-only", action="store_true", help="Skip waiting and run final comparison immediately.")
    parser.add_argument(
        "--official-pair-csv",
        default="work_dirs/compare_official_seg_metrics_baseline_vs_skel_8500_A.csv",
        help="CSV with baseline/skel official metrics.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_log = (root / args.train_log).resolve() if not Path(args.train_log).is_absolute() else Path(args.train_log)
    stage_json = (root / args.stage_json).resolve() if not Path(args.stage_json).is_absolute() else Path(args.stage_json)
    stage_csv = (root / args.stage_csv).resolve() if not Path(args.stage_csv).is_absolute() else Path(args.stage_csv)
    final_dir = (root / args.final_dir).resolve() if not Path(args.final_dir).is_absolute() else Path(args.final_dir)

    baseline_cfg = (root / args.baseline_config).resolve()
    baseline_ckpt = (root / args.baseline_ckpt).resolve()
    skel_cfg = (root / args.skel_config).resolve()
    skel_ckpt = (root / args.skel_ckpt).resolve()
    cldice_cfg = (root / args.cldice_config).resolve()
    cldice_ckpt_dir = (root / args.cldice_ckpt_dir).resolve()
    official_pair_csv = (root / args.official_pair_csv).resolve()

    stage_json.parent.mkdir(parents=True, exist_ok=True)
    if stage_json.exists():
        state = json.loads(stage_json.read_text(encoding="utf-8"))
    else:
        state = {}

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    required = f"{root}:{root / 'third_party/mmsegmentation'}"
    env["PYTHONPATH"] = required if not py_path else f"{required}:{py_path}"

    print(f"[MONITOR] train_log={train_log}")

    def run_final_compare() -> None:
        print("[MONITOR] starting final comparison...")
        final_dir.mkdir(parents=True, exist_ok=True)

        cldice_best = find_latest_best(cldice_ckpt_dir)
        best_iter_match = re.search(r"iter_(\d+)", cldice_best.name)
        if not best_iter_match:
            raise RuntimeError(f"Cannot parse best iter from checkpoint name: {cldice_best.name}")
        cldice_best_iter = int(best_iter_match.group(1))
        print(f"[MONITOR] clDice checkpoint: {cldice_best}")

        result = {}

        # Baseline/Skel official metrics from validated historical official csv.
        if not official_pair_csv.exists():
            raise FileNotFoundError(f"Missing official pair csv: {official_pair_csv}")
        result["baseline"] = parse_pair_official_csv(official_pair_csv, "baseline")
        result["skel"] = parse_pair_official_csv(official_pair_csv, "skel")

        # clDice official metrics from this run's validation log at best checkpoint iter.
        train_text = train_log.read_text(encoding="utf-8", errors="ignore")
        result["skel_cldice"] = parse_val_metrics_after_checkpoint(train_text, cldice_best_iter)

        variants = {
            "baseline": (baseline_cfg, baseline_ckpt),
            "skel": (skel_cfg, skel_ckpt),
            "skel_cldice": (cldice_cfg, cldice_best),
        }
        for name, (cfg, ckpt) in variants.items():
            pred_dir = final_dir / f"preds_{name}_unique"
            mapping_csv = final_dir / f"preds_{name}_mapping.csv"
            topo_csv = final_dir / f"topo_{name}.csv"
            topo_summary = final_dir / f"topo_{name}_summary.csv"
            print(f"[MONITOR] topology eval: {name}")
            topo = run_topology(root, env, cfg, ckpt, pred_dir, mapping_csv, topo_csv, topo_summary)
            result[name].update(topo)

        summary_csv = final_dir / "summary_baseline_skel_skel_cldice.csv"
        fields = [
            "variant",
            "aAcc",
            "mIoU",
            "mDice",
            "mFscore",
            "mPrecision",
            "mRecall",
            "tactile_iou",
            "tactile_dice",
            "tactile_precision",
            "tactile_recall",
            "topo_samples",
            "topo_mean_dice",
            "topo_mean_iou",
            "topo_mean_clDice",
            "topo_mean_cc_pred",
            "topo_mean_cc_gt",
            "topo_mean_lcr_pred",
            "topo_mean_lcr_gt",
        ]

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for v in ["baseline", "skel", "skel_cldice"]:
                row = {"variant": v}
                row.update(result[v])
                writer.writerow(row)

        with (final_dir / "summary_baseline_skel_skel_cldice.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        (final_dir / "done.flag").write_text("ok\n", encoding="utf-8")
        print(f"[MONITOR] final summary ready: {summary_csv}")

    if args.final_only:
        run_final_compare()
        return

    final_done_flag = final_dir / "done.flag"
    while True:
        if not train_log.exists():
            time.sleep(args.poll_seconds)
            continue

        text = train_log.read_text(encoding="utf-8", errors="ignore")

        changed = False
        for it in MILESTONES:
            key = str(it)
            if key in state and state[key].get("complete"):
                continue

            train_vals = parse_train_line_for_iter(text, it)
            val_vals = parse_val_after_iter(text, it)
            if train_vals is None or val_vals is None:
                continue

            state[key] = {
                "complete": True,
                "train": train_vals,
                "val": val_vals,
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            changed = True
            print(f"[MONITOR] captured iter={it} mIoU={val_vals['mIoU']:.2f} mDice={val_vals['mDice']:.2f}")

        if changed:
            stage_json.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            dump_stage_csv(stage_json, stage_csv)

        is_train_done = "Iter(train) [10000/10000]" in text and "Saving checkpoint at 10000 iterations" in text
        if is_train_done and not final_done_flag.exists():
            run_final_compare()
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
