#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare official MMSeg eval metrics from log/json artifacts.

This script is intended for paper/report numbers, using the exact values
reported by MMSeg test logs/json instead of recomputing from exported PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict


def load_json_metrics(json_path: Path) -> Dict[str, float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = {}
    for k in ["aAcc", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall"]:
        if k in data:
            out[k] = float(data[k])
    return out


def load_tactile_row(log_path: Path) -> Dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    # Example row:
    # | tactile_paving | 67.63 | 89.36 | 80.69 | 80.69  |   73.55   | 89.36  |
    m = re.search(
        r"\|\s*tactile_paving\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        text,
    )
    if not m:
        raise RuntimeError(f"Cannot find tactile_paving row in log: {log_path}")

    return {
        "tactile_iou": float(m.group(1)),
        "tactile_acc": float(m.group(2)),
        "tactile_dice": float(m.group(3)),
        "tactile_fscore": float(m.group(4)),
        "tactile_precision": float(m.group(5)),
        "tactile_recall": float(m.group(6)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare official baseline/skel metrics from eval json+log.")
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--skel-json", required=True)
    parser.add_argument("--skel-log", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    b = load_json_metrics(Path(args.baseline_json))
    b.update(load_tactile_row(Path(args.baseline_log)))

    s = load_json_metrics(Path(args.skel_json))
    s.update(load_tactile_row(Path(args.skel_log)))

    order = [
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
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "skel", "delta_skel_minus_base"])
        for k in order:
            bv = b[k]
            sv = s[k]
            w.writerow([k, f"{bv:.4f}", f"{sv:.4f}", f"{(sv - bv):+.4f}"])

    print(f"[OK] wrote official comparison csv: {out_csv}")


if __name__ == "__main__":
    main()
