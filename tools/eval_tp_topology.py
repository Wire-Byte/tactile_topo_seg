#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate TP topology-style metrics from predicted PNG masks.

Expected prediction directory layout:
- one PNG per sample, file name stem is dataset index, e.g. 0000.png

Outputs:
- per-sample CSV with columns similar to historical topo csv files
- summary CSV with mean metrics and aggregate counts
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "opencv-python is required for topology evaluation. "
            "Please install opencv-python or opencv-python-headless."
        ) from e
    return cv2


def read_mask_as_binary(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_fg = pred.astype(bool)
    gt_fg = gt.astype(bool)
    inter = np.logical_and(pred_fg, gt_fg).sum(dtype=np.float64)
    pred_sum = pred_fg.sum(dtype=np.float64)
    gt_sum = gt_fg.sum(dtype=np.float64)
    union = np.logical_or(pred_fg, gt_fg).sum(dtype=np.float64)

    if pred_sum + gt_sum == 0:
        dice = 1.0
    else:
        dice = float((2.0 * inter) / (pred_sum + gt_sum + 1e-12))

    if union == 0:
        iou = 1.0
    else:
        iou = float(inter / (union + 1e-12))

    return dice, iou


def morph_skeleton(binary01: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    img = (binary01 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return (skel > 0).astype(np.uint8)


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sk = morph_skeleton(pred)
    gt_sk = morph_skeleton(gt)

    pred_sk_sum = pred_sk.sum(dtype=np.float64)
    gt_sk_sum = gt_sk.sum(dtype=np.float64)

    if pred_sk_sum == 0 and gt_sk_sum == 0:
        return 1.0

    tprec = float((pred_sk * gt).sum(dtype=np.float64) / (pred_sk_sum + 1e-12)) if pred_sk_sum > 0 else 0.0
    tsens = float((gt_sk * pred).sum(dtype=np.float64) / (gt_sk_sum + 1e-12)) if gt_sk_sum > 0 else 0.0

    if tprec + tsens == 0:
        return 0.0
    return float((2.0 * tprec * tsens) / (tprec + tsens + 1e-12))


def count_connected_components(binary01: np.ndarray) -> int:
    cv2 = _import_cv2()
    num_labels, _ = cv2.connectedComponents((binary01 > 0).astype(np.uint8), connectivity=8)
    return int(max(0, num_labels - 1))


def largest_component_ratio(binary01: np.ndarray) -> float:
    cv2 = _import_cv2()
    bin8 = (binary01 > 0).astype(np.uint8)
    total_fg = int(bin8.sum())
    if total_fg == 0:
        return 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
    if num_labels <= 1:
        return 0.0

    # stats[1:, cv2.CC_STAT_AREA] are foreground component areas
    max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(max_area / (total_fg + 1e-12))


def count_holes(binary01: np.ndarray) -> int:
    cv2 = _import_cv2()
    # Count holes via contour hierarchy: child contours correspond to holes.
    mask = (binary01 > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return 0

    h = hierarchy[0]
    holes = 0
    for i in range(len(contours)):
        parent_idx = h[i][3]
        if parent_idx != -1:
            holes += 1
    return int(holes)


@dataclass
class Row:
    idx: int
    part_id: str
    pred_file: str
    gt_file: str
    dice: float
    iou: float
    clDice: float
    cc_pred: int
    cc_gt: int
    lcr_pred: float
    lcr_gt: float
    holes_pred: int
    holes_gt: int
    fgpix_pred: int
    fgpix_gt: int


def iter_prediction_files(pred_dir: Path) -> Iterable[Path]:
    return sorted(p for p in pred_dir.glob("*.png") if p.is_file())


def load_split_lines(split_file: Path) -> List[str]:
    lines = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Split file is empty: {split_file}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate topology-style metrics for TP masks.")
    parser.add_argument("--pred-dir", required=True, help="Directory with predicted PNG masks (index-based filenames).")
    parser.add_argument("--split-file", default="data/TP-Dataset/Index/val.txt", help="Split txt with sample IDs.")
    parser.add_argument("--gt-root", default="data/TP-Dataset/GroundTruth", help="Root directory containing GT png files.")
    parser.add_argument("--out-csv", required=True, help="Output per-sample CSV path.")
    parser.add_argument("--summary-csv", default=None, help="Optional summary CSV path.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    split_file = Path(args.split_file)
    gt_root = Path(args.gt_root)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv) if args.summary_csv else out_csv.with_name(out_csv.stem + "_summary.csv")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    split_lines = load_split_lines(split_file)

    rows: List[Row] = []
    missing = 0

    for pred_path in iter_prediction_files(pred_dir):
        try:
            idx = int(pred_path.stem)
        except ValueError:
            # Skip non index-like png files.
            continue

        if idx < 0 or idx >= len(split_lines):
            continue

        part_id = split_lines[idx]
        gt_path = gt_root / f"{part_id}.png"
        if not gt_path.exists():
            missing += 1
            continue

        pred = read_mask_as_binary(pred_path)
        gt = read_mask_as_binary(gt_path)

        if pred.shape != gt.shape:
            cv2 = _import_cv2()
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        d, j = dice_iou(pred, gt)
        cd = cldice(pred, gt)

        row = Row(
            idx=idx,
            part_id=part_id,
            pred_file=pred_path.name,
            gt_file=str(gt_path),
            dice=d,
            iou=j,
            clDice=cd,
            cc_pred=count_connected_components(pred),
            cc_gt=count_connected_components(gt),
            lcr_pred=largest_component_ratio(pred),
            lcr_gt=largest_component_ratio(gt),
            holes_pred=count_holes(pred),
            holes_gt=count_holes(gt),
            fgpix_pred=int(pred.sum()),
            fgpix_gt=int(gt.sum()),
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid prediction/gt pairs found in: {pred_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx",
            "part_id",
            "pred_file",
            "gt_file",
            "dice",
            "iou",
            "clDice",
            "cc_pred",
            "cc_gt",
            "lcr_pred",
            "lcr_gt",
            "holes_pred",
            "holes_gt",
            "fgpix_pred",
            "fgpix_gt",
        ])
        for r in sorted(rows, key=lambda x: x.idx):
            writer.writerow([
                r.idx,
                r.part_id,
                r.pred_file,
                r.gt_file,
                f"{r.dice:.12f}",
                f"{r.iou:.12f}",
                f"{r.clDice:.12f}",
                r.cc_pred,
                r.cc_gt,
                f"{r.lcr_pred:.12f}",
                f"{r.lcr_gt:.12f}",
                r.holes_pred,
                r.holes_gt,
                r.fgpix_pred,
                r.fgpix_gt,
            ])

    metrics = {
        "samples": len(rows),
        "missing_gt": missing,
        "mean_dice": float(np.mean([r.dice for r in rows])),
        "mean_iou": float(np.mean([r.iou for r in rows])),
        "mean_clDice": float(np.mean([r.clDice for r in rows])),
        "mean_cc_pred": float(np.mean([r.cc_pred for r in rows])),
        "mean_cc_gt": float(np.mean([r.cc_gt for r in rows])),
        "mean_lcr_pred": float(np.mean([r.lcr_pred for r in rows])),
        "mean_lcr_gt": float(np.mean([r.lcr_gt for r in rows])),
        "mean_holes_pred": float(np.mean([r.holes_pred for r in rows])),
        "mean_holes_gt": float(np.mean([r.holes_gt for r in rows])),
    }

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow([
            metrics["samples"],
            metrics["missing_gt"],
            f"{metrics['mean_dice']:.12f}",
            f"{metrics['mean_iou']:.12f}",
            f"{metrics['mean_clDice']:.12f}",
            f"{metrics['mean_cc_pred']:.12f}",
            f"{metrics['mean_cc_gt']:.12f}",
            f"{metrics['mean_lcr_pred']:.12f}",
            f"{metrics['mean_lcr_gt']:.12f}",
            f"{metrics['mean_holes_pred']:.12f}",
            f"{metrics['mean_holes_gt']:.12f}",
        ])

    print(f"[OK] per-sample csv: {out_csv}")
    print(f"[OK] summary csv   : {summary_csv}")
    print(
        "[SUMMARY] "
        f"N={metrics['samples']} "
        f"dice={metrics['mean_dice']:.4f} "
        f"iou={metrics['mean_iou']:.4f} "
        f"clDice={metrics['mean_clDice']:.4f}"
    )


if __name__ == "__main__":
    main()
