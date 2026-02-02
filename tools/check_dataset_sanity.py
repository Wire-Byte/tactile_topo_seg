#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset sanity checker for TP dataset + skeleton.

Usage example:
  PYTHONPATH=.:third_party/mmsegmentation \
  python tools/check_dataset_sanity.py \
    --config configs/tp_dataset/segformer_b2_tp.py \
    --split train \
    --num 8 \
    --require-positive \
    --out work_dirs/segformer_b2_tp/sanity

This script:
  - builds dataset from config (real pipeline)
  - samples items and checks gt_sem_seg / gt_skeleton presence, shapes, unique values
  - optionally checks skeleton is inside mask
  - saves visualization PNGs (no GUI needed)
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np

# Force headless backend (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _build_dataset(cfg, split: str):
    # Lazy imports so the script can run with correct PYTHONPATH
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import DATASETS

    if isinstance(cfg, (str, Path)):
        cfg = Config.fromfile(str(cfg))

    init_default_scope("mmseg")

    # cfg.train_dataloader / val_dataloader / test_dataloader
    key = f"{split}_dataloader"
    if not hasattr(cfg, key):
        raise KeyError(f"Config has no '{key}'. Available: train_dataloader/val_dataloader/test_dataloader")

    dataloader_cfg = getattr(cfg, key)
    if "dataset" not in dataloader_cfg:
        raise KeyError(f"'{key}' has no dataset field")

    dataset_cfg = dataloader_cfg["dataset"]
    ds = DATASETS.build(dataset_cfg)
    return ds


def _extract_maps(sample):
    """
    sample is usually a dict with keys: inputs, data_samples
    data_samples holds gt_sem_seg / gt_skeleton etc.
    """
    if isinstance(sample, dict) and "data_samples" in sample:
        dsample = sample["data_samples"]
    else:
        dsample = sample  # fallback

    def get_attr(name):
        return hasattr(dsample, name)

    # semantic mask
    if not get_attr("gt_sem_seg"):
        raise AttributeError("Missing gt_sem_seg in data_samples. Check your pipeline PackSegInputs.")

    gt_sem = dsample.gt_sem_seg.data
    # mmengine PixelData usually: torch Tensor [1,H,W]
    try:
        gt_sem = gt_sem.detach().cpu().numpy()
    except Exception:
        gt_sem = np.array(gt_sem)

    # skeleton
    if not get_attr("gt_skeleton"):
        raise AttributeError("Missing gt_skeleton in data_samples. Check your pipeline GenerateTPSkeleton + PackSegInputsWithSkeleton.")

    gt_skel = dsample.gt_skeleton.data
    try:
        gt_skel = gt_skel.detach().cpu().numpy()
    except Exception:
        gt_skel = np.array(gt_skel)

    # image meta / path (best-effort)
    img_path = None
    try:
        img_path = dsample.img_path
    except Exception:
        try:
            img_path = dsample.metainfo.get("img_path", None)
        except Exception:
            img_path = None

    return gt_sem, gt_skel, img_path


def _save_viz(out_path: Path, gt_sem, gt_skel, img_path=None, title_prefix=""):
    """
    Save side-by-side visualization:
      left: gt_sem_seg
      right: gt_skeleton
      overlay: skeleton on sem (optional third panel)
    """
    # squeeze to [H,W]
    sem = gt_sem[0] if gt_sem.ndim == 3 and gt_sem.shape[0] == 1 else gt_sem
    sk  = gt_skel[0] if gt_skel.ndim == 3 and gt_skel.shape[0] == 1 else gt_skel

    fig = plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title(f"{title_prefix}gt_sem_seg")
    ax1.imshow(sem, cmap="gray")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title(f"{title_prefix}gt_skeleton")
    ax2.imshow(sk, cmap="gray")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title(f"{title_prefix}overlay(skel on sem)")
    ax3.imshow(sem, cmap="gray")
    # overlay skeleton in red
    sk_mask = (sk > 0).astype(np.float32)
    ax3.imshow(np.dstack([sk_mask, np.zeros_like(sk_mask), np.zeros_like(sk_mask)]), alpha=0.6)
    ax3.axis("off")

    if img_path:
        fig.suptitle(str(img_path), fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to mmseg config, e.g. configs/tp_dataset/segformer_b2_tp.py")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Which dataloader dataset to use")
    parser.add_argument("--num", type=int, default=8, help="How many samples to save")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 means deterministic random)")
    parser.add_argument("--require-positive", action="store_true",
                        help="Only keep samples where gt_sem_seg contains class=1 (TP present)")
    parser.add_argument("--check-skel-inside", action="store_true",
                        help="Check skeleton pixels are inside sem mask (class==1).")
    parser.add_argument("--max-scan", type=int, default=2000,
                        help="Max items to scan when require-positive is on")
    parser.add_argument("--out", default="work_dirs/_sanity", help="Output directory to save PNGs")
    parser.add_argument("--indices", default=None,
                        help="Comma-separated indices to inspect (overrides random). e.g. 0,5,123")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _build_dataset(args.config, args.split)
    n = len(ds)
    print(f"[OK] Built dataset split='{args.split}', length={n}")

    # choose indices
    if args.indices is not None:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        # random pick, then filter if required-positive
        indices = list(range(n))
        random.shuffle(indices)

    saved = 0
    scanned = 0

    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        scanned += 1
        if args.require_positive and scanned > args.max_scan:
            print(f"[WARN] Reached max_scan={args.max_scan} while searching positive samples.")
            break

        try:
            sample = ds[idx]
            gt_sem, gt_skel, img_path = _extract_maps(sample)
        except Exception as e:
            print(f"[ERR] idx={idx} failed to load/extract: {e}")
            continue

        # squeeze to [H,W] for checks
        sem = gt_sem[0] if gt_sem.ndim == 3 and gt_sem.shape[0] == 1 else gt_sem
        sk  = gt_skel[0] if gt_skel.ndim == 3 and gt_skel.shape[0] == 1 else gt_skel

        sem_u = np.unique(sem)
        sk_u  = np.unique(sk)

        if args.require_positive and (1 not in sem_u):
            continue

        # basic shape check
        if sem.shape != sk.shape:
            print(f"[BAD] idx={idx} shape mismatch: sem={sem.shape}, sk={sk.shape} (expected equal)")
            # still save for debugging
        else:
            print(f"[OK] idx={idx} sem_u={sem_u} sk_u={sk_u} shape={sem.shape}")

        # skeleton-inside check
        if args.check_skel_inside:
            sk_pos = (sk > 0)
            sem_pos = (sem == 1)
            outside = np.logical_and(sk_pos, np.logical_not(sem_pos)).sum()
            total = sk_pos.sum()
            if total == 0:
                print(f"[WARN] idx={idx} skeleton is empty (total sk pixels = 0)")
            elif outside > 0:
                ratio = outside / (total + 1e-6)
                print(f"[BAD] idx={idx} skeleton outside mask: outside={outside}/{total} ({ratio:.3%})")
            else:
                print(f"[OK] idx={idx} skeleton-inside check passed (sk pixels={total})")

        # save visualization
        fname = f"{args.split}_{idx:05d}.png"
        out_path = out_dir / fname
        _save_viz(out_path, gt_sem, gt_skel, img_path=img_path, title_prefix=f"idx={idx} ")
        print(f"  -> saved: {out_path}")

        saved += 1
        if saved >= args.num:
            break

    print(f"\nDone. Saved {saved} samples to: {out_dir.resolve()}")
    if saved == 0:
        print("[HINT] If require-positive is on, try increasing --max-scan or turn it off.")


if __name__ == "__main__":
    main()
