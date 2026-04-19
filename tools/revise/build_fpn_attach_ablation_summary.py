#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = [
    {
        "name": "F1",
        "config": "configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F1.py",
        "work_dir": "work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F1",
        "log": "logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F1.log",
    },
    {
        "name": "F2",
        "config": "configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4.py",
        "work_dir": "work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4",
        "log": "logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F2.log",
    },
    {
        "name": "F3",
        "config": "configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F3.py",
        "work_dir": "work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F3",
        "log": "logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F3.log",
    },
    {
        "name": "F4",
        "config": "configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F4.py",
        "work_dir": "work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F4",
        "log": "logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F4.log",
    },
]

SAVE_ITER_RE = re.compile(r"Saving checkpoint at\s+(\d+)\s+iterations")
VAL_RE = re.compile(
    r"Iter\(val\)\s+\[281/281\]\s+"
    r"aAcc:\s*([0-9.]+)\s+"
    r"mIoU:\s*([0-9.]+)\s+"
    r"mAcc:\s*([0-9.]+)\s+"
    r"mDice:\s*([0-9.]+)\s+"
    r"mFscore:\s*([0-9.]+)\s+"
    r"mPrecision:\s*([0-9.]+)\s+"
    r"mRecall:\s*([0-9.]+)"
)
BEST_RE = re.compile(r"best checkpoint with\s+([0-9.]+)\s+mIoU\s+at\s+(\d+)\s+iter", re.IGNORECASE)


def parse_log(log_path: Path) -> Dict[str, Optional[float]]:
    if not log_path.exists():
        return {
            "status": "missing_log",
            "best_iter": None,
            "best_mIoU": None,
            "aAcc": None,
            "mIoU": None,
            "mDice": None,
            "mFscore": None,
            "mPrecision": None,
            "mRecall": None,
        }

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_iter: Optional[int] = None
    metrics_by_iter: Dict[int, Dict[str, float]] = {}
    best_iter: Optional[int] = None
    best_miou: Optional[float] = None

    for line in lines:
        m = SAVE_ITER_RE.search(line)
        if m:
            current_iter = int(m.group(1))
            continue

        m = VAL_RE.search(line)
        if m and current_iter is not None:
            metrics_by_iter[current_iter] = {
                "aAcc": float(m.group(1)),
                "mIoU": float(m.group(2)),
                "mDice": float(m.group(4)),
                "mFscore": float(m.group(5)),
                "mPrecision": float(m.group(6)),
                "mRecall": float(m.group(7)),
            }
            continue

        m = BEST_RE.search(line)
        if m:
            best_miou = float(m.group(1))
            best_iter = int(m.group(2))

    if best_iter is None:
        return {
            "status": "running_or_no_best",
            "best_iter": None,
            "best_mIoU": None,
            "aAcc": None,
            "mIoU": None,
            "mDice": None,
            "mFscore": None,
            "mPrecision": None,
            "mRecall": None,
        }

    row = metrics_by_iter.get(best_iter)
    if row is None:
        return {
            "status": "best_found_metrics_missing",
            "best_iter": best_iter,
            "best_mIoU": best_miou,
            "aAcc": None,
            "mIoU": best_miou,
            "mDice": None,
            "mFscore": None,
            "mPrecision": None,
            "mRecall": None,
        }

    return {
        "status": "ok",
        "best_iter": best_iter,
        "best_mIoU": best_miou,
        **row,
    }


def main() -> None:
    out_dir = ROOT / "docs/revise"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "fpn_attach_ablation_table.csv"
    out_md = out_dir / "fpn_attach_ablation_table.md"

    rows: List[Dict[str, object]] = []
    for exp in EXPERIMENTS:
        parsed = parse_log(ROOT / exp["log"])
        rows.append(
            {
                "Attach": exp["name"],
                "status": parsed["status"],
                "best_iter": parsed["best_iter"],
                "aAcc": parsed["aAcc"],
                "mIoU": parsed["mIoU"],
                "mDice": parsed["mDice"],
                "mFscore": parsed["mFscore"],
                "mPrecision": parsed["mPrecision"],
                "mRecall": parsed["mRecall"],
                "config": exp["config"],
                "work_dir": exp["work_dir"],
                "log": exp["log"],
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Attach",
                "status",
                "best_iter",
                "aAcc",
                "mIoU",
                "mDice",
                "mFscore",
                "mPrecision",
                "mRecall",
                "config",
                "work_dir",
                "log",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# Skeleton Head Attach Ablation (F1/F2/F3/F4)",
        "",
        "| Attach | status | best_iter | aAcc | mIoU | mDice | mFscore | mPrecision | mRecall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v: object) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(r["Attach"]),
                    fmt(r["status"]),
                    fmt(r["best_iter"]),
                    fmt(r["aAcc"]),
                    fmt(r["mIoU"]),
                    fmt(r["mDice"]),
                    fmt(r["mFscore"]),
                    fmt(r["mPrecision"]),
                    fmt(r["mRecall"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## File Mapping",
        "",
        "| Attach | config | work_dir | log |",
        "|---|---|---|---|",
    ]

    for r in rows:
        lines.append(
            f"| {r['Attach']} | {r['config']} | {r['work_dir']} | {r['log']} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] wrote {out_csv}")
    print(f"[DONE] wrote {out_md}")


if __name__ == "__main__":
    main()
