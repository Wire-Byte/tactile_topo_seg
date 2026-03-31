#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Table 2 topology comparison artifacts for TP segmentation models."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "tools" / "export_preds_unique_by_index.py"
TOPO_SCRIPT = ROOT / "tools" / "eval_tp_topology.py"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    config: Path
    checkpoint: Path


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        key="segformer_b2_baseline",
        display_name="SegFormer-B2 (baseline)",
        config=ROOT / "configs" / "tp_dataset" / "segformer_b2_tp.py",
        checkpoint=ROOT / "work_dirs" / "segformer_b2_tp" / "iter_8500.pth",
    ),
    ModelSpec(
        key="segformer_b2_skeleton",
        display_name="SegFormer-B2 + Skeleton",
        config=ROOT / "configs" / "tp_dataset" / "segformer_b2_tp_skel.py",
        checkpoint=ROOT / "work_dirs" / "segformer_b2_tp_skel" / "ckpt" / "segformer_b2_tp_skel" / "best_mIoU_iter_8500.pth",
    ),
    ModelSpec(
        key="segformer_b2_skeleton_cldice_v2",
        display_name="SegFormer-B2 + Skeleton + clDice-v2",
        config=ROOT / "configs" / "tp_dataset" / "segformer_b2_tp_skel_clDice_v2_4090_bs4.py",
        checkpoint=ROOT / "work_dirs" / "segformer_b2_tp_skel_cldice_v2_4090_bs4" / "ckpt" / "segformer_b2_tp_skel_cldice_v2_4090_bs4" / "best_mIoU_iter_8000.pth",
    ),
    ModelSpec(
        key="deeplabv3plus_r50",
        display_name="DeepLabV3+ (R50)",
        config=ROOT / "configs" / "tp_dataset" / "deeplabv3plus_r50_tp_main_table1_B.py",
        checkpoint=ROOT / "work_dirs" / "deeplabv3plus_r50_tp_main_table1_B" / "ckpt" / "deeplabv3plus_r50_tp_main_table1_B" / "best_mIoU_iter_9000.pth",
    ),
    ModelSpec(
        key="pspnet_r50",
        display_name="PSPNet (R50)",
        config=ROOT / "configs" / "tp_dataset" / "pspnet_r50_tp_main_table1_B.py",
        checkpoint=ROOT / "work_dirs" / "pspnet_r50_tp_main_table1_B" / "ckpt" / "pspnet_r50_tp_main_table1_B" / "best_mIoU_iter_9000.pth",
    ),
    ModelSpec(
        key="upernet_r50",
        display_name="UPerNet (R50)",
        config=ROOT / "configs" / "tp_dataset" / "upernet_r50_tp_main_table1_B.py",
        checkpoint=ROOT / "work_dirs" / "upernet_r50_tp_main_table1_B" / "ckpt" / "upernet_r50_tp_main_table1_B" / "best_mIoU_iter_6000.pth",
    ),
]


def _run(cmd: List[str], env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(x) for x in cmd)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== CMD =====\n{printable}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {printable}")


def _summary_row(path: Path) -> Dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty summary csv: {path}")
    return rows[0]


def _mean_from_per_sample(path: Path, key: str) -> float:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty per-sample csv: {path}")
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals)


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "config",
        "checkpoint",
        "samples",
        "missing_gt",
        "mean_dice",
        "mean_iou",
        "mean_clDice",
        "mean_cc_pred",
        "mean_cc_gt",
        "mean_lcr_pred",
        "mean_lcr_gt",
        "mean_holes_pred",
        "mean_holes_gt",
        "mean_fgpix_pred",
        "mean_fgpix_gt",
        "notes",
        "per_sample_csv",
        "summary_csv",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(text: str, digits: int = 4) -> str:
    return f"{float(text):.{digits}f}"


def _write_markdown(path: Path, rows: List[Dict[str, str]], split_file: Path, gt_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Table 2 Topology Comparison")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Evaluation split: `{split_file.relative_to(ROOT)}`")
    lines.append(f"- Ground-truth root: `{gt_root.relative_to(ROOT)}`")
    lines.append("- Prediction export: `tools/export_preds_unique_by_index.py`")
    lines.append("- Topology evaluation: `tools/eval_tp_topology.py`")
    lines.append("")
    lines.append("## Results: Overlap And Connectivity Scores")
    lines.append("")
    lines.append("| Model | Dice | IoU | clDice |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {_format_float(row['mean_dice'])} | {_format_float(row['mean_iou'])} | {_format_float(row['mean_clDice'])} |"
        )
    lines.append("")
    lines.append("## Results: Structural Statistics")
    lines.append("")
    lines.append("| Model | CC Pred | CC GT | LCR Pred | LCR GT | Holes Pred | Holes GT | FG Pred | FG GT |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {_format_float(row['mean_cc_pred'])} | {_format_float(row['mean_cc_gt'])} | "
            f"{_format_float(row['mean_lcr_pred'])} | {_format_float(row['mean_lcr_gt'])} | "
            f"{_format_float(row['mean_holes_pred'])} | {_format_float(row['mean_holes_gt'])} | "
            f"{_format_float(row['mean_fgpix_pred'])} | {_format_float(row['mean_fgpix_gt'])} |"
        )
    lines.append("")
    lines.append("## Metric Meanings")
    lines.append("- `Dice`: prediction mask and GT mask overlap score. Higher is better.")
    lines.append("- `IoU`: intersection-over-union between prediction mask and GT mask. Higher is better.")
    lines.append("- `clDice`: connectivity-aware overlap score computed from prediction and GT skeletons. Higher is better for thin connected structures like tactile paving.")
    lines.append("- `CC Pred`: average number of connected components in predicted masks. Lower usually means less fragmentation.")
    lines.append("- `CC GT`: average number of connected components in GT masks. This is the structural reference of the dataset and should be identical across models.")
    lines.append("- `LCR Pred`: average largest-component ratio in predicted masks. Higher usually means the foreground is concentrated in one dominant connected structure.")
    lines.append("- `LCR GT`: average largest-component ratio in GT masks. This is the structural reference of the dataset and should be identical across models.")
    lines.append("- `Holes Pred`: average number of holes inside predicted foreground regions. Lower usually means cleaner masks.")
    lines.append("- `Holes GT`: average number of holes inside GT masks. This is the structural reference of the dataset and should be identical across models.")
    lines.append("- `FG Pred`: average foreground-pixel count in predicted masks. It reflects the predicted object scale or area.")
    lines.append("- `FG GT`: average foreground-pixel count in GT masks. It is the area reference of the dataset and should be identical across models.")
    lines.append("")
    lines.append("## Model Checkpoints")
    for row in rows:
        lines.append(f"- `{row['model']}`")
        lines.append(f"  config: `{row['config']}`")
        lines.append(f"  checkpoint: `{row['checkpoint']}`")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append(f"- Consolidated CSV: `{(ROOT / 'docs' / 'tables' / 'table2_topology_metrics.csv').relative_to(ROOT)}`")
    lines.append(f"- Detailed markdown: `{(ROOT / 'docs' / 'tables' / 'table2_topology_metrics.md').relative_to(ROOT)}`")
    lines.append(f"- Per-sample CSVs: `{(ROOT / 'work_dirs' / 'table2_topology' / 'csv').relative_to(ROOT)}`")
    lines.append(f"- Prediction masks: `{(ROOT / 'work_dirs' / 'table2_topology' / 'preds').relative_to(ROOT)}`")
    lines.append(f"- Logs: `{(ROOT / 'logs' / 'table2_topology').relative_to(ROOT)}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Table 2 topology comparison artifacts.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run child tools.")
    parser.add_argument("--device", default="cuda:0", help="Inference device for prediction export.")
    parser.add_argument("--split-file", default="data/TP-Dataset/Index/val.txt", help="Split file used for export and evaluation.")
    parser.add_argument("--gt-root", default="data/TP-Dataset/GroundTruth", help="GT root used for topology evaluation.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing prediction and csv artifacts when present.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    split_file = (ROOT / args.split_file).resolve()
    gt_root = (ROOT / args.gt_root).resolve()

    pred_root = ROOT / "work_dirs" / "table2_topology" / "preds"
    csv_root = ROOT / "work_dirs" / "table2_topology" / "csv"
    logs_root = ROOT / "logs" / "table2_topology"
    docs_csv = ROOT / "docs" / "tables" / "table2_topology_metrics.csv"
    docs_md = ROOT / "docs" / "tables" / "table2_topology_metrics.md"

    pred_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    required = f"{ROOT}:{ROOT / 'third_party' / 'mmsegmentation'}"
    env["PYTHONPATH"] = required if not py_path else f"{required}:{py_path}"

    rows: List[Dict[str, str]] = []

    for spec in MODEL_SPECS:
        pred_dir = pred_root / spec.key
        mapping_csv = pred_dir / "pred_mapping.csv"
        per_csv = csv_root / f"{spec.key}_topology_per_sample.csv"
        summary_csv = csv_root / f"{spec.key}_topology_summary.csv"
        log_path = logs_root / f"{spec.key}.log"

        if not args.skip_existing or not pred_dir.exists() or not any(pred_dir.glob("*.png")):
            export_cmd = [
                args.python,
                str(EXPORT_SCRIPT),
                "--config",
                str(spec.config),
                "--checkpoint",
                str(spec.checkpoint),
                "--split-file",
                str(split_file),
                "--out-dir",
                str(pred_dir),
                "--mapping-csv",
                str(mapping_csv),
                "--device",
                args.device,
            ]
            _run(export_cmd, env, log_path)

        if not args.skip_existing or not per_csv.exists() or not summary_csv.exists():
            topo_cmd = [
                args.python,
                str(TOPO_SCRIPT),
                "--pred-dir",
                str(pred_dir),
                "--split-file",
                str(split_file),
                "--gt-root",
                str(gt_root),
                "--out-csv",
                str(per_csv),
                "--summary-csv",
                str(summary_csv),
            ]
            _run(topo_cmd, env, log_path)

        summary = _summary_row(summary_csv)
        rows.append(
            {
                "model": spec.display_name,
                "config": str(spec.config.relative_to(ROOT)),
                "checkpoint": str(spec.checkpoint.relative_to(ROOT)),
                "samples": summary["samples"],
                "missing_gt": summary["missing_gt"],
                "mean_dice": summary["mean_dice"],
                "mean_iou": summary["mean_iou"],
                "mean_clDice": summary["mean_clDice"],
                "mean_cc_pred": summary["mean_cc_pred"],
                "mean_cc_gt": summary["mean_cc_gt"],
                "mean_lcr_pred": summary["mean_lcr_pred"],
                "mean_lcr_gt": summary["mean_lcr_gt"],
                "mean_holes_pred": summary["mean_holes_pred"],
                "mean_holes_gt": summary["mean_holes_gt"],
                "mean_fgpix_pred": f"{_mean_from_per_sample(per_csv, 'fgpix_pred'):.12f}",
                "mean_fgpix_gt": f"{_mean_from_per_sample(per_csv, 'fgpix_gt'):.12f}",
                "notes": "Topology metrics on val split via eval_tp_topology.py",
                "per_sample_csv": str(per_csv.relative_to(ROOT)),
                "summary_csv": str(summary_csv.relative_to(ROOT)),
                "log": str(log_path.relative_to(ROOT)),
            }
        )

    _write_csv(docs_csv, rows)
    _write_markdown(docs_md, rows, split_file, gt_root)
    print(f"[OK] wrote {docs_csv}")
    print(f"[OK] wrote {docs_md}")


if __name__ == "__main__":
    main()
