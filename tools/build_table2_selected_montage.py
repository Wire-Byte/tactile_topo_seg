#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLS: Sequence[Tuple[str, str]] = (
    ("original", "Original"),
    ("gt", "GT"),
    ("segformer_b2_baseline", "SegFormer-B2"),
    ("segformer_b2_skeleton", "SegFormer-B2+Skeleton"),
    ("segformer_b2_skeleton_cldice_v2", "SegFormer-B2+Skeleton+clDice"),
    ("deeplabv3plus_r50", "DeepLabV3+"),
    ("pspnet_r50", "PSPNet"),
    ("upernet_r50", "UPerNet"),
)


def _load_rows(mapping_csv: Path) -> Dict[int, Dict[str, str]]:
    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty mapping CSV: {mapping_csv}")
    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        out[int(r["idx"])] = r
    return out


def _ensure_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., 0]
    raise ValueError(f"Unexpected ndim={arr.ndim}")


def _mask_to_bw_rgb(path: Path, size_wh: Tuple[int, int]) -> Image.Image:
    arr = np.array(Image.open(path))
    bw = (_ensure_gray(arr) > 0).astype(np.uint8) * 255
    img = Image.fromarray(bw, mode="L").convert("RGB")
    if img.size != size_wh:
        img = img.resize(size_wh, resample=Image.NEAREST)
    return img


def _fit_rgb(path: Path, size_wh: Tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize(size_wh, resample=Image.BILINEAR)


def _build_rotated_label(text: str, font: ImageFont.ImageFont, angle: int = 45) -> Image.Image:
    tmp = Image.new("RGBA", (2000, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    txt = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
    dt = ImageDraw.Draw(txt)
    dt.text((4, 4), text, fill=(20, 20, 20, 255), font=font)
    return txt.rotate(angle, expand=True, resample=Image.BICUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selected multi-row comparison montage with rotated column labels.")
    parser.add_argument("--indices", nargs="+", type=int, required=True, help="Sample indices, e.g. 1 6 11 19")
    parser.add_argument(
        "--mapping-csv",
        default="work_dirs/table2_topology/preds/segformer_b2_baseline/pred_mapping.csv",
    )
    parser.add_argument("--pred-root", default="work_dirs/table2_topology/preds")
    parser.add_argument("--out", default="work_dirs/table2_topology/vis_compare/montage_selected.png")
    parser.add_argument("--tile-w", type=int, default=260)
    parser.add_argument("--tile-h", type=int, default=200)
    args = parser.parse_args()

    mapping_csv = Path(args.mapping_csv)
    pred_root = Path(args.pred_root)
    out_path = Path(args.out)

    rows = _load_rows(mapping_csv)
    missing = [i for i in args.indices if i not in rows]
    if missing:
        raise RuntimeError(f"Indices not found in mapping CSV: {missing}")

    tile_w = args.tile_w
    tile_h = args.tile_h
    gap = 10
    margin = 16

    font = ImageFont.load_default()
    labels = [_build_rotated_label(name, font, angle=45) for _, name in COLS]
    labels_h = max(lb.height for lb in labels) + 8

    nrows = len(args.indices)
    ncols = len(COLS)
    canvas_w = margin * 2 + ncols * tile_w + (ncols - 1) * gap
    canvas_h = margin * 2 + nrows * tile_h + (nrows - 1) * gap + labels_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 248))

    for r_i, idx in enumerate(args.indices):
        meta = rows[idx]
        pred_file = meta["pred_file"]
        img_path = Path(meta["img_file"])
        gt_path = Path(meta["gt_file"])

        y = margin + r_i * (tile_h + gap)
        for c_i, (key, _) in enumerate(COLS):
            x = margin + c_i * (tile_w + gap)
            if key == "original":
                tile = _fit_rgb(img_path, (tile_w, tile_h))
            elif key == "gt":
                tile = _mask_to_bw_rgb(gt_path, (tile_w, tile_h))
            else:
                pred_path = pred_root / key / pred_file
                if not pred_path.exists():
                    raise FileNotFoundError(f"Missing prediction: {pred_path}")
                tile = _mask_to_bw_rgb(pred_path, (tile_w, tile_h))
            canvas.paste(tile, (x, y))

    y_label = margin + nrows * tile_h + (nrows - 1) * gap + 4
    for c_i, lb in enumerate(labels):
        x_center = margin + c_i * (tile_w + gap) + tile_w // 2
        x = int(x_center - lb.width // 2)
        canvas.paste(lb, (x, y_label), lb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
