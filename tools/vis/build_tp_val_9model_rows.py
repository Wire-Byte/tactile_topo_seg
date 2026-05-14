#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build per-sample qualitative rows for TP-Dataset val, 9-model comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER: Sequence[Tuple[str, str, str]] = (
    ("segformer_b2", "SegFormer-B2", "work_dirs/table2_topology/preds/segformer_b2_baseline"),
    ("ours_skeleton", "Ours(+Skeleton)", "work_dirs/vis/tp_val_9models/preds/core_none"),
    ("ours_full", "Ours(Full)", "work_dirs/vis/tp_val_9models/preds/core_both"),
    ("deeplabv3plus", "DeepLabV3+", "work_dirs/table2_topology/preds/deeplabv3plus_r50"),
    ("pspnet", "PSPNet", "work_dirs/table2_topology/preds/pspnet_r50"),
    ("upernet", "UPerNet", "work_dirs/table2_topology/preds/upernet_r50"),
    ("vmunet", "VM-UNet", "work_dirs/revise/vmunet_tp_revise/preds_best"),
    ("emcad", "EMCAD", "work_dirs/revise/emcad_tp_revise/preds_best"),
    ("bevanet_s", "BEVANet-S", "work_dirs/revise/bevanet_s_tp_revise/preds_best"),
)


def _load_rows(mapping_csv: Path) -> List[Dict[str, str]]:
    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty mapping csv: {mapping_csv}")
    return rows


def _resolve(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _binary_mask(path: Path) -> Image.Image:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    bw = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(bw, mode="L")


def _tile_rgb(img: Image.Image, tile_size: int, resample: int) -> Image.Image:
    return img.convert("RGB").resize((tile_size, tile_size), resample=resample)


def _open_mask_tile(path: Path, tile_size: int) -> Image.Image:
    return _tile_rgb(_binary_mask(path), tile_size, Image.Resampling.NEAREST)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def _paste_label(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, text: str, font: ImageFont.ImageFont) -> None:
    tw, th = _text_size(draw, text, font)
    draw.text((x + max(0, (width - tw) // 2), y + max(0, (24 - th) // 2)), text, fill=(20, 20, 20), font=font)


def build_row(
    row: Dict[str, str],
    root: Path,
    out_path: Path,
    tile_size: int,
    gap: int,
    labeled: bool,
    font: ImageFont.ImageFont,
) -> None:
    idx = int(row["idx"])
    pred_file = row["pred_file"]
    img_path = _resolve(row["img_file"], root)
    gt_path = _resolve(row["gt_file"], root)

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT: {gt_path}")

    tiles: List[Tuple[str, Image.Image]] = [
        ("Original", _tile_rgb(Image.open(img_path), tile_size, Image.Resampling.BICUBIC)),
        ("GroundTruth", _open_mask_tile(gt_path, tile_size)),
    ]

    for _, label, pred_dir_text in MODEL_ORDER:
        pred_path = root / pred_dir_text / pred_file
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for idx={idx}: {pred_path}")
        tiles.append((label, _open_mask_tile(pred_path, tile_size)))

    cols = len(tiles)
    label_h = 28 if labeled else 0
    canvas_w = cols * tile_size + (cols + 1) * gap
    canvas_h = tile_size + (2 * gap) + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y_img = gap + label_h
    for col, (label, tile) in enumerate(tiles):
        x = gap + col * (tile_size + gap)
        if labeled:
            _paste_label(draw, x, gap, tile_size, label, font)
        canvas.paste(tile, (x, y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TP val 9-model visual rows.")
    parser.add_argument("--mapping-csv", default="work_dirs/table2_topology/preds/segformer_b2_baseline/pred_mapping.csv")
    parser.add_argument("--out-root", default="work_dirs/vis/tp_val_9models")
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    mapping_csv = _resolve(args.mapping_csv, root)
    out_root = _resolve(args.out_root, root)
    rows = _load_rows(mapping_csv)
    if args.indices:
        selected = set(args.indices)
        rows = [r for r in rows if int(r["idx"]) in selected]
        if not rows:
            raise RuntimeError("No rows matched --indices.")

    font = ImageFont.load_default()
    rows_labeled = out_root / "rows_labeled"
    rows_clean = out_root / "rows_clean"
    index_csv = out_root / "row_index.csv"
    index_csv.parent.mkdir(parents=True, exist_ok=True)

    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "part_id", "labeled_row", "clean_row", "img_file", "gt_file"])
        for n, row in enumerate(rows, start=1):
            idx = int(row["idx"])
            safe_part = row["part_id"].replace("/", "_")
            filename = f"{idx:04d}_{safe_part}.png"
            labeled_path = rows_labeled / filename
            clean_path = rows_clean / filename
            build_row(row, root, labeled_path, args.tile_size, args.gap, True, font)
            build_row(row, root, clean_path, args.tile_size, args.gap, False, font)
            writer.writerow([idx, row["part_id"], str(labeled_path), str(clean_path), row["img_file"], row["gt_file"]])
            if n % 25 == 0 or n == len(rows):
                print(f"[INFO] built {n}/{len(rows)}")

    print(f"[OK] labeled rows: {rows_labeled}")
    print(f"[OK] clean rows  : {rows_clean}")
    print(f"[OK] index csv   : {index_csv}")


if __name__ == "__main__":
    main()
