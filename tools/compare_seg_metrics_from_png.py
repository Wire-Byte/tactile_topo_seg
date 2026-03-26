#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare segmentation metrics from existing PNG predictions.

Supports two matching modes:
1) mapping mode (recommended): use per-sample CSV with pred_file/gt_file
2) index mode: use split-file index and pred names like 0000.png

No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def load_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def confusion_for_class(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred_fg = pred == 1
    gt_fg = gt == 1
    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    fp = int(np.logical_and(pred_fg, np.logical_not(gt_fg)).sum())
    fn = int(np.logical_and(np.logical_not(pred_fg), gt_fg).sum())
    tn = int(np.logical_and(np.logical_not(pred_fg), np.logical_not(gt_fg)).sum())
    return tp, fp, fn, tn


def metrics_from_conf(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    iou_fg = tp / (tp + fp + fn + 1e-12)
    iou_bg = tn / (tn + fp + fn + 1e-12)
    dice_fg = 2 * tp / (2 * tp + fp + fn + 1e-12)
    dice_bg = 2 * tn / (2 * tn + fp + fn + 1e-12)
    prec_fg = tp / (tp + fp + 1e-12)
    rec_fg = tp / (tp + fn + 1e-12)
    fscore_fg = 2 * prec_fg * rec_fg / (prec_fg + rec_fg + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "aAcc": acc,
        "IoU_bg": iou_bg,
        "IoU_fg": iou_fg,
        "mIoU": (iou_bg + iou_fg) / 2.0,
        "Dice_bg": dice_bg,
        "Dice_fg": dice_fg,
        "mDice": (dice_bg + dice_fg) / 2.0,
        "Precision_fg": prec_fg,
        "Recall_fg": rec_fg,
        "Fscore_fg": fscore_fg,
        "mFscore": ((2 * (tn / (tn + fn + 1e-12)) * (tn / (tn + fp + 1e-12)) / ((tn / (tn + fn + 1e-12)) + (tn / (tn + fp + 1e-12)) + 1e-12)) + fscore_fg) / 2.0,
    }


def evaluate_by_index(pred_dir: Path, split_file: Path, gt_root: Path) -> Dict[str, float]:
    split_ids = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    total_tp = total_fp = total_fn = total_tn = 0
    used = 0

    for pred_path in sorted(pred_dir.glob("*.png")):
        try:
            idx = int(pred_path.stem)
        except ValueError:
            continue

        if idx < 0 or idx >= len(split_ids):
            continue

        part_id = split_ids[idx]

        gt_path = gt_root / f"{part_id}.png"
        if not gt_path.exists():
            continue

        pred = load_binary(pred_path)
        gt = load_binary(gt_path)
        if pred.shape != gt.shape:
            # Nearest resize via PIL to avoid extra dependencies.
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred = (np.array(pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0).astype(np.uint8)

        tp, fp, fn, tn = confusion_for_class(pred, gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        used += 1

    if used == 0:
        raise RuntimeError(f"No matched prediction/gt pairs under: {pred_dir}")

    out = metrics_from_conf(total_tp, total_fp, total_fn, total_tn)
    out["num_images"] = float(used)
    return out


def evaluate_by_mapping(mapping_csv: Path, pred_dir: Path) -> Dict[str, float]:
    total_tp = total_fp = total_fn = total_tn = 0
    used = 0

    with mapping_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_name = row.get("pred_file", "").strip()
            gt_name = row.get("gt_file", "").strip()
            if not pred_name or not gt_name:
                continue

            pred_path = pred_dir / pred_name
            gt_path = Path(gt_name)
            if not pred_path.exists() or not gt_path.exists():
                continue

            pred = load_binary(pred_path)
            gt = load_binary(gt_path)
            if pred.shape != gt.shape:
                pred_img = Image.fromarray((pred * 255).astype(np.uint8))
                pred = (np.array(pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0).astype(np.uint8)

            tp, fp, fn, tn = confusion_for_class(pred, gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            used += 1

    if used == 0:
        raise RuntimeError(f"No matched rows found from mapping file: {mapping_csv}")

    out = metrics_from_conf(total_tp, total_fp, total_fn, total_tn)
    out["num_images"] = float(used)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline/skel segmentation metrics from existing png predictions.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--skel-dir", required=True)
    parser.add_argument("--baseline-mapping-csv", default=None, help="Optional baseline mapping CSV with columns pred_file,gt_file.")
    parser.add_argument("--skel-mapping-csv", default=None, help="Optional skel mapping CSV with columns pred_file,gt_file.")
    parser.add_argument("--split-file", default="data/TP-Dataset/Index/val.txt")
    parser.add_argument("--gt-root", default="data/TP-Dataset/GroundTruth")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    skel_dir = Path(args.skel_dir)
    split_file = Path(args.split_file)
    gt_root = Path(args.gt_root)
    out_csv = Path(args.out_csv)

    if args.baseline_mapping_csv and args.skel_mapping_csv:
        b = evaluate_by_mapping(Path(args.baseline_mapping_csv), baseline_dir)
        s = evaluate_by_mapping(Path(args.skel_mapping_csv), skel_dir)
    else:
        b = evaluate_by_index(baseline_dir, split_file, gt_root)
        s = evaluate_by_index(skel_dir, split_file, gt_root)

    keys = [
        "num_images",
        "aAcc",
        "IoU_bg",
        "IoU_fg",
        "mIoU",
        "Dice_bg",
        "Dice_fg",
        "mDice",
        "Precision_fg",
        "Recall_fg",
        "Fscore_fg",
        "mFscore",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "skel", "delta_skel_minus_base"])
        for k in keys:
            bv = b[k]
            sv = s[k]
            w.writerow([k, f"{bv:.12f}", f"{sv:.12f}", f"{(sv - bv):.12f}"])

    print(f"[OK] Wrote segmentation comparison csv: {out_csv}")
    print("[SUMMARY] mIoU baseline={:.4f} skel={:.4f} delta={:+.4f}".format(b["mIoU"], s["mIoU"], s["mIoU"] - b["mIoU"]))
    print("[SUMMARY] mDice baseline={:.4f} skel={:.4f} delta={:+.4f}".format(b["mDice"], s["mDice"], s["mDice"] - b["mDice"]))
    print("[SUMMARY] mFscore baseline={:.4f} skel={:.4f} delta={:+.4f}".format(b["mFscore"], s["mFscore"], s["mFscore"] - b["mFscore"]))


if __name__ == "__main__":
    main()
