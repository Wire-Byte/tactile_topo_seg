#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


def _import_cv2():
    import cv2  # type: ignore
    return cv2


def read_mask_as_binary(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


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


def count_connected_components(binary01: np.ndarray) -> int:
    cv2 = _import_cv2()
    num_labels, _ = cv2.connectedComponents((binary01 > 0).astype(np.uint8), connectivity=8)
    return int(max(0, num_labels - 1))


def count_holes(binary01: np.ndarray) -> int:
    cv2 = _import_cv2()
    mask = (binary01 > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return 0
    h = hierarchy[0]
    holes = 0
    for i in range(len(contours)):
        if h[i][3] != -1:
            holes += 1
    return int(holes)


def tprec_tsens_cldice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12):
    pred_sk = morph_skeleton(pred)
    gt_sk = morph_skeleton(gt)

    pred_sk_sum = float(pred_sk.sum())
    gt_sk_sum = float(gt_sk.sum())

    if pred_sk_sum == 0 and gt_sk_sum == 0:
        return 1.0, 1.0, 1.0

    t_prec = float((pred_sk * gt).sum() / (pred_sk_sum + eps)) if pred_sk_sum > 0 else 0.0
    t_sens = float((gt_sk * pred).sum() / (gt_sk_sum + eps)) if gt_sk_sum > 0 else 0.0

    if t_prec + t_sens <= 0:
        cldice = 0.0
    else:
        cldice = float(2.0 * t_prec * t_sens / (t_prec + t_sens + eps))

    return t_prec, t_sens, cldice


@dataclass
class SampleRow:
    model: str
    idx: int
    part_id: str
    pred_file: str
    gt_file: str
    t_prec: float
    t_sens: float
    cldice: float
    cc_pred: int
    cc_gt: int
    holes_pred: int
    holes_gt: int
    abs_dbeta0: int
    abs_dbeta1: int
    betti_error: int


def _sample_header() -> List[str]:
    return [
        "model",
        "idx",
        "part_id",
        "pred_file",
        "gt_file",
        "T_prec",
        "T_sens",
        "clDice",
        "CC_pred",
        "CC_gt",
        "Holes_pred",
        "Holes_gt",
        "abs_dBeta0",
        "abs_dBeta1",
        "Betti_Error",
    ]


def _row_to_csv(r: SampleRow) -> List[str]:
    return [
        r.model,
        str(r.idx),
        r.part_id,
        r.pred_file,
        r.gt_file,
        f"{r.t_prec:.12f}",
        f"{r.t_sens:.12f}",
        f"{r.cldice:.12f}",
        str(r.cc_pred),
        str(r.cc_gt),
        str(r.holes_pred),
        str(r.holes_gt),
        str(r.abs_dbeta0),
        str(r.abs_dbeta1),
        str(r.betti_error),
    ]


def _ensure_csv_header(path: Path, header: List[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)


def _load_done_indices(model_csv: Path) -> Set[int]:
    done: Set[int] = set()
    if not model_csv.exists() or model_csv.stat().st_size == 0:
        return done
    with model_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                done.add(int(r["idx"]))
            except Exception:
                continue
    return done


def _read_model_rows_from_csv(model_csv: Path) -> List[SampleRow]:
    out: List[SampleRow] = []
    if not model_csv.exists() or model_csv.stat().st_size == 0:
        return out
    with model_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                SampleRow(
                    model=r["model"],
                    idx=int(r["idx"]),
                    part_id=r["part_id"],
                    pred_file=r["pred_file"],
                    gt_file=r["gt_file"],
                    t_prec=float(r["T_prec"]),
                    t_sens=float(r["T_sens"]),
                    cldice=float(r["clDice"]),
                    cc_pred=int(r["CC_pred"]),
                    cc_gt=int(r["CC_gt"]),
                    holes_pred=int(r["Holes_pred"]),
                    holes_gt=int(r["Holes_gt"]),
                    abs_dbeta0=int(r["abs_dBeta0"]),
                    abs_dbeta1=int(r["abs_dBeta1"]),
                    betti_error=int(r["Betti_Error"]),
                )
            )
    return out


def _write_summary_files(
    out_dir: Path,
    model_order: List[str],
    summary_rows: Dict[str, Dict[str, float]],
) -> None:
    summary_csv = out_dir / "topology_v2_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "samples",
            "mean_T_prec",
            "mean_T_sens",
            "mean_clDice",
            "mean_Betti_Error",
            "mean_abs_dBeta0",
            "mean_abs_dBeta1",
            "ref_mean_clDice_table2",
            "delta_clDice_vs_table2",
        ])
        for model in model_order:
            if model not in summary_rows:
                continue
            s = summary_rows[model]
            w.writerow([
                model,
                int(s["samples"]),
                f"{s['mean_t_prec']:.12f}",
                f"{s['mean_t_sens']:.12f}",
                f"{s['mean_cldice']:.12f}",
                f"{s['mean_betti']:.12f}",
                f"{s['mean_db0']:.12f}",
                f"{s['mean_db1']:.12f}",
                f"{s['ref_cldice']:.12f}",
                f"{s['delta_cldice']:.12f}",
            ])

    summary_md = out_dir / "topology_v2_summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Topology V2 Metrics\n\n")
        f.write("Computed with per-sample-first averaging on the existing Table2 predictions.\n\n")
        f.write("- T_sens is reported as requested.\n")
        f.write("- Betti Error uses: |CC_pred-CC_gt| + |Holes_pred-Holes_gt| (per sample, then averaged).\n\n")
        f.write("| Model | Samples | T_prec | T_sens | clDice | Betti Error | |dBeta0| | |dBeta1| |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for model in model_order:
            if model not in summary_rows:
                continue
            s = summary_rows[model]
            f.write(
                f"| {model} | {int(s['samples'])} | {s['mean_t_prec']:.4f} | {s['mean_t_sens']:.4f} | "
                f"{s['mean_cldice']:.4f} | {s['mean_betti']:.4f} | {s['mean_db0']:.4f} | {s['mean_db1']:.4f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build topology-v2 metrics: T_prec, T_sens, clDice, Betti Error.")
    parser.add_argument("--pred-root", default="work_dirs/table2_topology/preds")
    parser.add_argument("--mapping-csv", default="work_dirs/table2_topology/preds/segformer_b2_baseline/pred_mapping.csv")
    parser.add_argument("--summary-src", default="docs/tables/table2_topology_metrics.csv")
    parser.add_argument("--out-dir", default="docs/tables/topo_v2")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N newly processed samples.")
    parser.add_argument("--resume", action="store_true", help="Resume from per-model CSV files if they already exist.")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    mapping_csv = Path(args.mapping_csv)
    summary_src = Path(args.summary_src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if not summary_src.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_src}")

    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        mapping_rows = list(csv.DictReader(f))
    if not mapping_rows:
        raise RuntimeError("Empty mapping CSV")

    with summary_src.open("r", newline="", encoding="utf-8") as f:
        model_rows = list(csv.DictReader(f))
    if not model_rows:
        raise RuntimeError("Empty summary source CSV")

    per_sample_csv = out_dir / "topology_v2_per_sample.csv"
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    _ensure_csv_header(per_sample_csv, _sample_header())

    summary_rows: Dict[str, Dict[str, float]] = {}
    model_order = [m["model"] for m in model_rows]

    for m in model_rows:
        model_name = m["model"]
        per_csv_path = Path(m["per_sample_csv"])
        model_key = per_csv_path.name.replace("_topology_per_sample.csv", "")
        pred_dir = pred_root / model_key
        model_csv = per_model_dir / f"{model_key}_topology_v2_per_sample.csv"
        _ensure_csv_header(model_csv, _sample_header())

        done_indices: Set[int] = _load_done_indices(model_csv) if args.resume else set()
        total = len(mapping_rows)
        todo = max(0, total - len(done_indices))
        print(f"[MODEL] {model_name} ({model_key}) total={total} done={len(done_indices)} todo={todo}")

        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing model prediction dir: {pred_dir}")

        processed_new = 0
        skipped_done = 0

        with model_csv.open("a", newline="", encoding="utf-8") as fm, per_sample_csv.open("a", newline="", encoding="utf-8") as fall:
            wm = csv.writer(fm)
            wall = csv.writer(fall)

            for r in mapping_rows:
                idx = int(r["idx"])
                if idx in done_indices:
                    skipped_done += 1
                    continue

                pred_file = r["pred_file"]
                gt_file = r["gt_file"]
                part_id = r["part_id"]

                pred_path = pred_dir / pred_file
                gt_path = Path(gt_file)
                if not pred_path.exists() or not gt_path.exists():
                    continue

                pred = read_mask_as_binary(pred_path)
                gt = read_mask_as_binary(gt_path)
                if pred.shape != gt.shape:
                    cv2 = _import_cv2()
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                t_prec, t_sens, cldice = tprec_tsens_cldice(pred, gt)

                cc_pred = count_connected_components(pred)
                cc_gt = count_connected_components(gt)
                holes_pred = count_holes(pred)
                holes_gt = count_holes(gt)
                abs_dbeta0 = abs(cc_pred - cc_gt)
                abs_dbeta1 = abs(holes_pred - holes_gt)
                betti_error = abs_dbeta0 + abs_dbeta1

                sample = SampleRow(
                    model=model_name,
                    idx=idx,
                    part_id=part_id,
                    pred_file=pred_file,
                    gt_file=str(gt_path),
                    t_prec=t_prec,
                    t_sens=t_sens,
                    cldice=cldice,
                    cc_pred=cc_pred,
                    cc_gt=cc_gt,
                    holes_pred=holes_pred,
                    holes_gt=holes_gt,
                    abs_dbeta0=abs_dbeta0,
                    abs_dbeta1=abs_dbeta1,
                    betti_error=betti_error,
                )
                wm.writerow(_row_to_csv(sample))
                wall.writerow(_row_to_csv(sample))

                processed_new += 1
                if processed_new % max(1, args.progress_every) == 0:
                    print(f"[PROGRESS] {model_key}: new={processed_new}, done_total={len(done_indices) + processed_new}/{total}")

        rows = _read_model_rows_from_csv(model_csv)
        if rows:
            mean_t_prec = float(np.mean([x.t_prec for x in rows]))
            mean_t_sens = float(np.mean([x.t_sens for x in rows]))
            mean_cldice = float(np.mean([x.cldice for x in rows]))
            mean_betti = float(np.mean([x.betti_error for x in rows]))
            mean_db0 = float(np.mean([x.abs_dbeta0 for x in rows]))
            mean_db1 = float(np.mean([x.abs_dbeta1 for x in rows]))

            ref_row = next((x for x in model_rows if x["model"] == model_name), None)
            ref_cl = float(ref_row["mean_clDice"]) if ref_row is not None else np.nan
            delta = mean_cldice - ref_cl

            summary_rows[model_name] = {
                "samples": float(len(rows)),
                "mean_t_prec": mean_t_prec,
                "mean_t_sens": mean_t_sens,
                "mean_cldice": mean_cldice,
                "mean_betti": mean_betti,
                "mean_db0": mean_db0,
                "mean_db1": mean_db1,
                "ref_cldice": ref_cl,
                "delta_cldice": delta,
            }
            _write_summary_files(out_dir, model_order, summary_rows)

        print(
            f"[MODEL_DONE] {model_key}: new={processed_new}, skipped_done={skipped_done}, total_written={len(rows)}"
        )

    if not summary_rows:
        raise RuntimeError("No per-sample metrics were computed.")

    print(f"[OK] per-sample: {per_sample_csv}")
    print(f"[OK] summary   : {out_dir / 'topology_v2_summary.csv'}")
    print(f"[OK] markdown  : {out_dir / 'topology_v2_summary.md'}")


if __name__ == "__main__":
    main()
