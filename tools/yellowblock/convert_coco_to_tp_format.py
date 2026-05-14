#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert Roboflow COCO-segmentation export into TP-style folder layout.

Input layout:
  data/YellowBlock.v1i.coco-segmentation/{train,valid,test}/
    - *.jpg
    - _annotations.coco.json

Output layout (example):
  data/YellowBlock-TP/
    JPEGImages/
    GroundTruth/
    Index/
      train.txt
      val.txt
      test.txt

Index lines are image stems, compatible with TPIndexDataset/TPDataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert YellowBlock COCO to TP-style dataset")
    p.add_argument(
        "--input-root",
        default="data/YellowBlock.v1i.coco-segmentation",
        help="Roboflow COCO export root",
    )
    p.add_argument(
        "--output-root",
        default="data/YellowBlock-TP",
        help="Output TP-style dataset root",
    )
    p.add_argument(
        "--mapping",
        choices=["merge", "yellow-only"],
        default="merge",
        help="merge: yellow_block + yellow_block_damaged -> foreground; yellow-only: only yellow_block",
    )
    p.add_argument(
        "--link-images",
        action="store_true",
        help="Use hard links for images when possible (fallback to copy)",
    )
    return p.parse_args()


def _category_name_map(coco: Dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in coco.get("categories", []):
        out[int(c["id"])] = str(c["name"])
    return out


def _valid_category_ids(name_map: Dict[int, str], mapping: str) -> set[int]:
    if mapping == "merge":
        keep = {"yellow_block", "yellow_block_damaged"}
    else:
        keep = {"yellow_block"}
    return {cid for cid, name in name_map.items() if name in keep}


def _ann_by_image(coco: Dict) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        out.setdefault(img_id, []).append(ann)
    return out


def _draw_polygons(mask: np.ndarray, segmentation) -> None:
    if not isinstance(segmentation, list):
        return
    pil = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil)
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
        draw.polygon(pts, outline=1, fill=1)
    mask[:, :] = np.array(pil, dtype=np.uint8)


def _build_mask(width: int, height: int, anns: List[Dict], keep_ids: set[int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        if int(ann.get("category_id", -1)) not in keep_ids:
            continue
        seg = ann.get("segmentation", None)
        _draw_polygons(mask, seg)
    return mask


def _safe_link_or_copy(src: Path, dst: Path, link_images: bool) -> None:
    if dst.exists():
        return
    if link_images:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def convert_split(
    input_root: Path,
    split_name: str,
    out_img_dir: Path,
    out_gt_dir: Path,
    keep_ids: set[int],
    link_images: bool,
) -> Tuple[List[str], int]:
    split_dir = input_root / split_name
    ann_path = split_dir / "_annotations.coco.json"
    coco = json.loads(ann_path.read_text(encoding="utf-8"))

    images = coco.get("images", [])
    by_img = _ann_by_image(coco)

    index_lines: List[str] = []
    non_empty = 0

    for item in images:
        img_id = int(item["id"])
        file_name = str(item["file_name"])
        width = int(item["width"])
        height = int(item["height"])

        src_img = split_dir / file_name
        stem = Path(file_name).stem
        out_img = out_img_dir / f"{stem}.jpg"
        out_gt = out_gt_dir / f"{stem}.png"

        anns = by_img.get(img_id, [])
        mask = _build_mask(width, height, anns, keep_ids)
        if int(mask.sum()) > 0:
            non_empty += 1

        _safe_link_or_copy(src_img, out_img, link_images=link_images)
        Image.fromarray(mask, mode="L").save(out_gt)
        index_lines.append(stem)

    return index_lines, non_empty


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    out_img_dir = output_root / "JPEGImages"
    out_gt_dir = output_root / "GroundTruth"
    out_idx_dir = output_root / "Index"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_idx_dir.mkdir(parents=True, exist_ok=True)

    sample_coco = json.loads((input_root / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    name_map = _category_name_map(sample_coco)
    keep_ids = _valid_category_ids(name_map, mapping=args.mapping)

    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    summary = {}
    for raw_split, out_split in split_map.items():
        lines, non_empty = convert_split(
            input_root=input_root,
            split_name=raw_split,
            out_img_dir=out_img_dir,
            out_gt_dir=out_gt_dir,
            keep_ids=keep_ids,
            link_images=args.link_images,
        )
        (out_idx_dir / f"{out_split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary[out_split] = {
            "samples": len(lines),
            "non_empty_masks": non_empty,
            "non_empty_ratio": (float(non_empty) / max(1, len(lines))),
        }

    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "output_root": str(output_root),
            "mapping": args.mapping,
            "kept_category_ids": sorted(list(keep_ids)),
            "summary": summary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
