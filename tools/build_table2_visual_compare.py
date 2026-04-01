#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER: Sequence[Tuple[str, str]] = (
    ("segformer_b2_baseline", "SegFormer-B2"),
    ("segformer_b2_skeleton", "SegFormer-B2+Skeleton"),
    ("segformer_b2_skeleton_cldice_v2", "SegFormer-B2+Skeleton+clDice-v2"),
    ("deeplabv3plus_r50", "DeepLabV3+ (R50)"),
    ("pspnet_r50", "PSPNet (R50)"),
    ("upernet_r50", "UPerNet (R50)"),
)


def _load_rows(mapping_csv: Path) -> List[Dict[str, str]]:
    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty mapping csv: {mapping_csv}")
    return rows


def _ensure_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., 0]
    raise ValueError(f"Unexpected image array ndim={arr.ndim}")


def _to_binary_bw(mask_arr: np.ndarray) -> np.ndarray:
    mask = _ensure_gray(mask_arr)
    return (mask > 0).astype(np.uint8) * 255


def _open_binary_mask(path: Path, size_hw: Tuple[int, int]) -> Image.Image:
    # size_hw: (H, W)
    arr = np.array(Image.open(path))
    bw = _to_binary_bw(arr)
    img = Image.fromarray(bw, mode="L")
    if img.size != (size_hw[1], size_hw[0]):
        img = img.resize((size_hw[1], size_hw[0]), resample=Image.NEAREST)
    return img.convert("RGB")


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # textbbox exists in modern Pillow and gives robust width for centering.
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return (x1 - x0, y1 - y0)


def build_one_panel(
    row: Dict[str, str],
    pred_root: Path,
    out_dir: Path,
    index_width: int,
    labels_font: ImageFont.ImageFont,
) -> Path:
    idx = int(row["idx"])
    pred_file = row["pred_file"]
    img_path = Path(row["img_file"])
    gt_path = Path(row["gt_file"])

    if not img_path.exists():
        raise FileNotFoundError(f"Missing original image: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT image: {gt_path}")

    original = Image.open(img_path).convert("RGB")
    h, w = original.height, original.width

    tiles: List[Tuple[str, Image.Image]] = []
    tiles.append(("Original", original))
    tiles.append(("Ground Truth", _open_binary_mask(gt_path, (h, w))))

    for model_key, display_name in MODEL_ORDER:
        pred_path = pred_root / model_key / pred_file
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        tiles.append((display_name, _open_binary_mask(pred_path, (h, w))))

    ncols = len(tiles)
    margin = 10
    label_h = 30
    canvas_w = margin * (ncols + 1) + w * ncols
    canvas_h = margin * 2 + label_h + h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    y_img = margin + label_h
    for col, (name, tile) in enumerate(tiles):
        x = margin + col * (w + margin)
        canvas.paste(tile, (x, y_img))
        tw, th = _text_size(draw, name, labels_font)
        tx = x + max(0, (w - tw) // 2)
        ty = margin + max(0, (label_h - th) // 2)
        draw.text((tx, ty), name, fill=(20, 20, 20), font=labels_font)

    out_path = out_dir / f"{idx:0{index_width}d}_compare.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-sample visual comparison panels for Table2 models.")
    parser.add_argument(
        "--mapping-csv",
        default="work_dirs/table2_topology/preds/segformer_b2_baseline/pred_mapping.csv",
        help="CSV with idx/part_id/pred_file/img_file/gt_file.",
    )
    parser.add_argument(
        "--pred-root",
        default="work_dirs/table2_topology/preds",
        help="Root directory containing prediction folders for all models.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/table2_topology/vis_compare",
        help="Output directory for per-sample comparison images.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of sample indices to export. Example: --indices 0 12 57",
    )
    args = parser.parse_args()

    mapping_csv = Path(args.mapping_csv)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)

    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(mapping_csv)

    selected = None if not args.indices else set(args.indices)
    if selected is not None:
        rows = [r for r in rows if int(r["idx"]) in selected]
        if not rows:
            raise RuntimeError("No rows matched --indices.")

    font = ImageFont.load_default()
    index_width = max(4, len(str(max(int(r["idx"]) for r in rows))))

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        out_path = build_one_panel(row, pred_root, out_dir, index_width, font)
        if i % 20 == 0 or i == total:
            print(f"[INFO] exported {i}/{total}: {out_path.name}")

    print(f"[OK] done: {out_dir}")


if __name__ == "__main__":
    main()
