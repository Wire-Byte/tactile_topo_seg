#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export segmentation predictions with unique index-based filenames.

This avoids filename collisions from nested dataset paths when exporting
predictions for topology evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from mmengine.config import Config
from mmseg.apis import inference_model, init_model


def load_split(split_file: Path):
    lines = [ln.strip() for ln in split_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f'Empty split file: {split_file}')
    return lines


def main():
    parser = argparse.ArgumentParser(description='Export unique-named predictions for a split.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split-file', default='data/TP-Dataset/Index/val.txt')
    parser.add_argument('--data-root', default='data/TP-Dataset')
    parser.add_argument('--img-subdir', default='JPEGImages')
    parser.add_argument('--gt-subdir', default='GroundTruth')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--mapping-csv', default=None)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    split_file = Path(args.split_file)
    data_root = Path(args.data_root)
    img_root = data_root / args.img_subdir
    gt_root = data_root / args.gt_subdir
    out_dir = Path(args.out_dir)
    map_csv = Path(args.mapping_csv) if args.mapping_csv else (out_dir / 'pred_mapping.csv')

    out_dir.mkdir(parents=True, exist_ok=True)
    map_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    # Inference-only pipeline: avoid annotation-dependent transforms.
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='PackSegInputs'),
    ]
    model = init_model(cfg, str(ckpt_path), device=args.device)

    split_ids = load_split(split_file)
    width = max(4, len(str(len(split_ids) - 1)))

    with map_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'part_id', 'pred_file', 'img_file', 'gt_file'])

        for idx, part_id in enumerate(split_ids):
            img_path = img_root / f'{part_id}.jpg'
            gt_path = gt_root / f'{part_id}.png'
            if not img_path.exists():
                raise FileNotFoundError(f'Missing image file: {img_path}')

            result = inference_model(model, str(img_path))
            pred = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)

            pred_file = f'{idx:0{width}d}.png'
            Image.fromarray(pred).save(out_dir / pred_file)
            writer.writerow([idx, part_id, pred_file, str(img_path), str(gt_path)])

            if (idx + 1) % 50 == 0:
                print(f'[INFO] exported {idx + 1}/{len(split_ids)}')

    print(f'[OK] prediction dir: {out_dir}')
    print(f'[OK] mapping csv   : {map_csv}')


if __name__ == '__main__':
    main()
