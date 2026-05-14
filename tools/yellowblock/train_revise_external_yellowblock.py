#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]

# Match MMSeg ImageNet-style preprocessing for fairer comparison.
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_py_config(path: Path) -> Dict:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    for k in [
        "experiment_name",
        "model_name",
        "repo_root",
        "model_kwargs",
        "data",
        "train",
        "optimizer",
    ]:
        if not hasattr(mod, k):
            raise RuntimeError(f"Missing '{k}' in config: {path}")
        out[k] = getattr(mod, k)
    return out


def _read_split_ids(split_file: Path) -> List[str]:
    lines = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Empty split file: {split_file}")
    return lines


class TPIndexDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_file: Path,
        image_size: Tuple[int, int],
        train: bool,
        img_subdir: str,
        gt_subdir: str,
        normalize: bool = True,
        val_keep_original_size: bool = True,
    ) -> None:
        self.data_root = data_root
        self.img_root = data_root / img_subdir
        self.gt_root = data_root / gt_subdir
        self.ids = _read_split_ids(split_file)
        self.image_size = image_size
        self.train = train
        self.normalize = normalize
        self.val_keep_original_size = val_keep_original_size

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _resize(img: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        w, h = size[1], size[0]
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)
        img_r = np.array(img_pil.resize((w, h), Image.BILINEAR))
        mask_r = np.array(mask_pil.resize((w, h), Image.NEAREST))
        return img_r, mask_r

    @staticmethod
    def _random_rescale(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ratio = random.uniform(0.5, 2.0)
        h, w = img.shape[:2]
        nh = max(64, int(h * ratio))
        nw = max(64, int(w * ratio))
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)
        img_s = np.array(img_pil.resize((nw, nh), Image.BILINEAR))
        mask_s = np.array(mask_pil.resize((nw, nh), Image.NEAREST))
        return img_s, mask_s

    @staticmethod
    def _random_crop_or_pad(img: np.ndarray, mask: np.ndarray, crop_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        ch, cw = crop_hw
        h, w = img.shape[:2]
        if h < ch or w < cw:
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w = img.shape[:2]

        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        return img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]

    def __getitem__(self, idx: int):
        part_id = self.ids[idx]
        img_path = self.img_root / f"{part_id}.jpg"
        gt_path = self.gt_root / f"{part_id}.png"

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_raw = np.array(Image.open(gt_path), dtype=np.uint8)
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[..., 0]
        mask = (mask_raw > 0).astype(np.uint8)

        if self.train:
            img, mask = self._random_rescale(img, mask)
            img, mask = self._random_crop_or_pad(img, mask, self.image_size)
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
        else:
            if not self.val_keep_original_size:
                img, mask = self._resize(img, mask, self.image_size)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        if self.normalize:
            img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        else:
            img_t = img_t / 255.0
        mask_t = torch.from_numpy(mask).long()

        meta = {
            "idx": idx,
            "part_id": part_id,
            "img_file": str(img_path),
            "gt_file": str(gt_path),
        }
        return img_t, mask_t, meta


def _fast_hist(pred: np.ndarray, gt: np.ndarray, n: int = 2) -> np.ndarray:
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(int) + pred[k].astype(int), minlength=n * n).reshape(n, n)


def _seg_metrics_from_hist(hist: np.ndarray) -> Dict[str, float]:
    tp = np.diag(hist)
    sum_gt = hist.sum(axis=1)
    sum_pred = hist.sum(axis=0)
    total = hist.sum()

    iou = tp / (sum_gt + sum_pred - tp + 1e-12)
    dice = 2 * tp / (sum_gt + sum_pred + 1e-12)
    acc_cls = tp / (sum_gt + 1e-12)
    prec = tp / (sum_pred + 1e-12)
    rec = acc_cls
    fscore = 2 * prec * rec / (prec + rec + 1e-12)

    return {
        "aAcc": float(tp.sum() / (total + 1e-12)),
        "mIoU": float(np.mean(iou)),
        "mAcc": float(np.mean(acc_cls)),
        "mDice": float(np.mean(dice)),
        "mFscore": float(np.mean(fscore)),
        "mPrecision": float(np.mean(prec)),
        "mRecall": float(np.mean(rec)),
    }


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _resize_logits(logits: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_hw:
        return logits
    return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)


def _extract_main_aux_outputs(model_name: str, out) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if model_name == "vmunet":
        if isinstance(out, list):
            main = out[-1]
            aux = out[:-1]
        else:
            main = out
            aux = []
        return main, aux

    if model_name == "emcad":
        if not isinstance(out, list):
            return out, []
        return out[-1], out[:-1]

    if model_name == "bevanet_s":
        if isinstance(out, list) and len(out) >= 2:
            # BEVANet train-time outputs are [semantic_aux, semantic_main, boundary_aux].
            # boundary_aux is 1-channel and must not be used with CrossEntropyLoss.
            main = out[1]
            aux = [out[0]]
            return main, aux
        return out, []

    raise RuntimeError(f"Unknown model_name: {model_name}")


def build_model(model_name: str, repo_root: Path, model_kwargs: Dict) -> nn.Module:
    repo = str(repo_root.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    if model_name == "vmunet":
        from models.vmunet.vmunet import VMUNet  # type: ignore

        kwargs = dict(model_kwargs)
        ckpt = kwargs.get("load_ckpt_path")
        if ckpt:
            ckpt_path = Path(str(ckpt))
            if not ckpt_path.is_absolute():
                ckpt_path = ROOT / ckpt_path
            if ckpt_path.exists():
                kwargs["load_ckpt_path"] = str(ckpt_path)
            else:
                logging.warning("[PRETRAIN] vmunet checkpoint not found: %s; fallback to scratch", ckpt_path)
                kwargs["load_ckpt_path"] = None

        model = VMUNet(**kwargs)
        if hasattr(model, "load_from"):
            try:
                model.load_from()
            except Exception:
                # from-scratch settings may not provide pretrained ckpt
                pass
        return model

    if model_name == "emcad":
        from lib.networks import EMCADNet  # type: ignore

        kwargs = dict(model_kwargs)
        if bool(kwargs.get("pretrain", False)):
            pretrained_dir = Path(str(kwargs.get("pretrained_dir", "")))
            if not pretrained_dir.is_absolute():
                pretrained_dir = ROOT / pretrained_dir
            encoder = str(kwargs.get("encoder", "pvt_v2_b0"))
            expected_file = pretrained_dir / f"{encoder}.pth"
            if not expected_file.exists():
                logging.warning(
                    "[PRETRAIN] emcad pretrained file missing: %s; fallback to scratch",
                    expected_file,
                )
                kwargs["pretrain"] = False
            kwargs["pretrained_dir"] = str(pretrained_dir)

        return EMCADNet(**kwargs)

    if model_name == "bevanet_s":
        from configs import config as beva_cfg  # type: ignore
        from models.BEVANet import get_model  # type: ignore

        beva_cfg.defrost()
        beva_cfg.MODEL.BRANCHES = int(model_kwargs.get("branches", 2))
        beva_cfg.MODEL.PLANES = int(model_kwargs.get("planes", 32))
        beva_cfg.MODEL.NUM_BLOCKS = list(model_kwargs.get("num_blocks", [2, 2, 3, 3, 2]))
        beva_cfg.MODEL.SEMANTIC_KERNEL_SIZE = list(model_kwargs.get("semantic_kernel_size", [0, 35, 35]))
        beva_cfg.MODEL.DETAIL_KERNEL_SIZE = list(model_kwargs.get("detail_kernel_size", [0, 23]))
        beva_cfg.MODEL.MLP_EXPAND = list(model_kwargs.get("mlp_expand", [4, 4, 4]))
        beva_cfg.MODEL.PPM_PLANES = int(model_kwargs.get("ppm_planes", 96))
        beva_cfg.MODEL.HEAD_PLANES = int(model_kwargs.get("head_planes", 128))
        beva_cfg.MODEL.PRETRAINED = str(model_kwargs.get("pretrained", ""))
        beva_cfg.DATASET.NUM_CLASSES = int(model_kwargs.get("num_classes", 2))
        beva_cfg.freeze()

        return get_model(
            beva_cfg,
            pretrain_path=beva_cfg.MODEL.PRETRAINED,
            task="seg",
            num_classes=beva_cfg.DATASET.NUM_CLASSES,
            is_trainning=True,
        )

    raise RuntimeError(f"Unsupported model_name: {model_name}")


def evaluate_model(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    hist = np.zeros((2, 2), dtype=np.float64)

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = model(images)
            main_logits, _ = _extract_main_aux_outputs(model_name, out)
            main_logits = _resize_logits(main_logits, (masks.shape[-2], masks.shape[-1]))
            pred = torch.argmax(main_logits, dim=1).detach().cpu().numpy()
            gt = masks.detach().cpu().numpy()
            for p, g in zip(pred, gt):
                hist += _fast_hist(p, g, n=2)

    return _seg_metrics_from_hist(hist)


def export_predictions(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> Path:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv = out_dir / "pred_mapping.csv"

    with mapping_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "part_id", "pred_file", "img_file", "gt_file"])

        with torch.no_grad():
            for images, masks, metas in loader:
                images = images.to(device, non_blocking=True)
                out = model(images)
                main_logits, _ = _extract_main_aux_outputs(model_name, out)
                main_logits = _resize_logits(main_logits, (masks.shape[-2], masks.shape[-1]))
                pred = torch.argmax(main_logits, dim=1).detach().cpu().numpy().astype(np.uint8)

                bs = pred.shape[0]
                for i in range(bs):
                    idx = int(metas["idx"][i])
                    part_id = str(metas["part_id"][i])
                    img_file = str(metas["img_file"][i])
                    gt_file = str(metas["gt_file"][i])
                    pred_file = f"{idx:04d}.png"
                    Image.fromarray(pred[i]).save(out_dir / pred_file)
                    w.writerow([idx, part_id, pred_file, img_file, gt_file])

    return mapping_csv


def run_topology_eval(
    pred_dir: Path,
    mapping_csv: Path,
    metrics_dir: Path,
    topology_split_file: Path,
    topology_gt_root: Path,
) -> Tuple[Path, Path, Path, Path]:
    per_sample = metrics_dir / "topology_per_sample.csv"
    summary = metrics_dir / "topology_summary.csv"

    cmd = [
        sys.executable,
        str(ROOT / "tools/eval_tp_topology.py"),
        "--pred-dir",
        str(pred_dir),
        "--split-file",
        str(topology_split_file),
        "--gt-root",
        str(topology_gt_root),
        "--out-csv",
        str(per_sample),
        "--summary-csv",
        str(summary),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    topo_v2_per = metrics_dir / "topology_v2_per_sample.csv"
    topo_v2_sum = metrics_dir / "topology_v2_summary.csv"

    from tools.build_topo_v2_metrics import (
        count_connected_components,
        count_holes,
        read_mask_as_binary,
        tprec_tsens_cldice,
    )

    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tprec_list: List[float] = []
    tsens_list: List[float] = []
    cldice_list: List[float] = []
    db0_list: List[int] = []
    db1_list: List[int] = []
    betti_list: List[int] = []

    with topo_v2_per.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx",
            "part_id",
            "pred_file",
            "gt_file",
            "T_prec",
            "T_sens",
            "clDice",
            "abs_dBeta0",
            "abs_dBeta1",
            "Betti_Error",
        ])

        for r in rows:
            pred_path = pred_dir / r["pred_file"]
            gt_path = Path(r["gt_file"])
            pred = read_mask_as_binary(pred_path)
            gt = read_mask_as_binary(gt_path)
            if pred.shape != gt.shape:
                pred_img = Image.fromarray((pred * 255).astype(np.uint8))
                pred = (np.array(pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0).astype(np.uint8)

            t_prec, t_sens, cldice = tprec_tsens_cldice(pred, gt)
            cc_pred = count_connected_components(pred)
            cc_gt = count_connected_components(gt)
            holes_pred = count_holes(pred)
            holes_gt = count_holes(gt)
            db0 = abs(cc_pred - cc_gt)
            db1 = abs(holes_pred - holes_gt)
            betti = db0 + db1

            tprec_list.append(t_prec)
            tsens_list.append(t_sens)
            cldice_list.append(cldice)
            db0_list.append(db0)
            db1_list.append(db1)
            betti_list.append(betti)

            w.writerow([
                r["idx"],
                r["part_id"],
                r["pred_file"],
                r["gt_file"],
                f"{t_prec:.12f}",
                f"{t_sens:.12f}",
                f"{cldice:.12f}",
                db0,
                db1,
                betti,
            ])

    with topo_v2_sum.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "samples",
            "mean_T_prec",
            "mean_T_sens",
            "mean_clDice",
            "mean_Betti_Error",
            "mean_abs_dBeta0",
            "mean_abs_dBeta1",
        ])
        w.writerow([
            len(tprec_list),
            f"{float(np.mean(tprec_list)):.12f}",
            f"{float(np.mean(tsens_list)):.12f}",
            f"{float(np.mean(cldice_list)):.12f}",
            f"{float(np.mean(betti_list)):.12f}",
            f"{float(np.mean(db0_list)):.12f}",
            f"{float(np.mean(db1_list)):.12f}",
        ])

    return per_sample, summary, topo_v2_per, topo_v2_sum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train external official models on yellowblock-style TP format with unified evaluation.")
    p.add_argument("--config", required=True, help="Path to config in configs/yellowblock")
    p.add_argument("--work-dir", default=None, help="Override work dir")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    cfg = load_py_config(cfg_path)

    exp_name = cfg["experiment_name"]
    model_name = cfg["model_name"]
    repo_root = (ROOT / cfg["repo_root"]).resolve()
    model_kwargs = dict(cfg["model_kwargs"])

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    optim_cfg = cfg["optimizer"]

    default_work_subdir = str(train_cfg.get("work_subdir", "yellowblock/revise"))
    work_dir = Path(args.work_dir) if args.work_dir else (ROOT / "work_dirs" / default_work_subdir / exp_name)
    work_dir.mkdir(parents=True, exist_ok=True)

    log_subdir = str(train_cfg.get("log_subdir", "yellowblock/revise"))
    logs_dir = ROOT / "logs" / log_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"train_{exp_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    set_seed(int(train_cfg.get("seed", 3407)))

    data_root = ROOT / data_cfg["data_root"]
    normalize = bool(data_cfg.get("normalize", True))
    val_keep_original_size = bool(data_cfg.get("val_keep_original_size", True))
    if model_name in {"vmunet", "emcad"} and val_keep_original_size:
        # VM-UNet/EMCAD require aligned spatial sizes for skip merges.
        val_keep_original_size = False
        logging.warning(
            "[CALIBRATION] %s forces val_keep_original_size=False to avoid odd-size skip mismatch",
            model_name,
        )

    train_set = TPIndexDataset(
        data_root=data_root,
        split_file=data_root / data_cfg["train_index"],
        image_size=tuple(data_cfg["image_size"]),
        train=True,
        img_subdir=data_cfg["img_subdir"],
        gt_subdir=data_cfg["gt_subdir"],
        normalize=normalize,
        val_keep_original_size=False,
    )
    val_set = TPIndexDataset(
        data_root=data_root,
        split_file=data_root / data_cfg["val_index"],
        image_size=tuple(data_cfg["image_size"]),
        train=False,
        img_subdir=data_cfg["img_subdir"],
        gt_subdir=data_cfg["gt_subdir"],
        normalize=normalize,
        val_keep_original_size=val_keep_original_size,
    )

    logging.info(
        "[CALIBRATION] normalize=%s val_keep_original_size=%s train_size=%s",
        normalize,
        val_keep_original_size,
        tuple(data_cfg["image_size"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(train_cfg["num_workers"]) // 2),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, repo_root, model_kwargs).to(device)

    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if optim_cfg["type"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=wd,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    max_iters = int(train_cfg["max_iters"])
    warmup_iters = int(train_cfg.get("warmup_iters", 500))
    warmup_iters = max(0, min(warmup_iters, max_iters - 1))
    start_factor = 1e-6

    def lr_lambda(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            warmup_progress = float(step) / float(max(1, warmup_iters))
            return start_factor + (1.0 - start_factor) * warmup_progress

        decay_iters = max(1, max_iters - warmup_iters)
        decay_step = min(max(0, step - warmup_iters), decay_iters)
        return max(0.0, 1.0 - float(decay_step) / float(decay_iters))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    ema_momentum = float(train_cfg.get("ema_momentum", 0.0))
    use_ema = ema_momentum > 0.0
    eval_with_ema = bool(train_cfg.get("eval_with_ema", True))
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logging.info("[EMA] enabled momentum=%.6f eval_with_ema=%s", ema_momentum, eval_with_ema)

    ckpt_dir = work_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = work_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_interval = int(train_cfg["val_interval"])
    log_interval = int(train_cfg["log_interval"])

    best_miou = -1.0
    best_iter = 0
    best_ckpt = ckpt_dir / "best_mIoU.pth"
    train_iter = _infinite(train_loader)

    for it in range(1, max_iters + 1):
        model.train()
        images, masks, _ = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(images)
            main_logits, aux_logits = _extract_main_aux_outputs(model_name, out)
            main_logits = _resize_logits(main_logits, (masks.shape[-2], masks.shape[-1]))
            loss = ce(main_logits, masks)

            if aux_logits:
                aux_losses = []
                for aux in aux_logits:
                    aux = _resize_logits(aux, (masks.shape[-2], masks.shape[-1]))
                    aux_losses.append(ce(aux, masks))
                loss = loss + 0.4 * sum(aux_losses) / len(aux_losses)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if use_ema and ema_model is not None:
            with torch.no_grad():
                model_state = model.state_dict()
                ema_state = ema_model.state_dict()
                for k, v in ema_state.items():
                    if not torch.is_floating_point(v):
                        v.copy_(model_state[k])
                    else:
                        v.mul_(ema_momentum).add_(model_state[k], alpha=1.0 - ema_momentum)

        if it % log_interval == 0:
            logging.info(
                "iter=%d/%d lr=%.6g loss=%.6f",
                it,
                max_iters,
                optimizer.param_groups[0]["lr"],
                float(loss.detach().item()),
            )

        if it % val_interval == 0 or it == max_iters:
            eval_model = ema_model if (use_ema and eval_with_ema and ema_model is not None) else model
            val_metrics = evaluate_model(eval_model, model_name, val_loader, device)
            logging.info(
                "[VAL] iter=%d aAcc=%.4f mIoU=%.4f mDice=%.4f mFscore=%.4f mPrecision=%.4f mRecall=%.4f",
                it,
                val_metrics["aAcc"] * 100,
                val_metrics["mIoU"] * 100,
                val_metrics["mDice"] * 100,
                val_metrics["mFscore"] * 100,
                val_metrics["mPrecision"] * 100,
                val_metrics["mRecall"] * 100,
            )

            iter_ckpt = ckpt_dir / f"iter_{it}.pth"
            torch.save(
                {
                    "iter": it,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict() if use_ema and ema_model is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                },
                iter_ckpt,
            )

            if val_metrics["mIoU"] > best_miou:
                best_miou = val_metrics["mIoU"]
                best_iter = it
                torch.save(
                    {
                        "iter": it,
                        "model": eval_model.state_dict(),
                        "ema_model": ema_model.state_dict() if use_ema and ema_model is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    best_ckpt,
                )
                best_ckpt_named = ckpt_dir / f"best_mIoU_iter_{it}.pth"
                torch.save(
                    {
                        "iter": it,
                        "model": eval_model.state_dict(),
                        "ema_model": ema_model.state_dict() if use_ema and ema_model is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    best_ckpt_named,
                )
                logging.info("[BEST] iter=%d mIoU=%.4f saved=%s", it, best_miou * 100, best_ckpt_named)

    if not best_ckpt.exists():
        raise RuntimeError("Best checkpoint was not created.")

    state = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)

    best_val_metrics = evaluate_model(model, model_name, val_loader, device)
    preds_dir = work_dir / "preds_best"
    mapping_csv = export_predictions(model, model_name, val_loader, device, preds_dir)

    topology_split = data_root / str(data_cfg.get("topology_split", data_cfg["val_index"]))
    topology_gt_root = data_root / str(data_cfg.get("topology_gt_subdir", data_cfg["gt_subdir"]))
    topo_per, topo_summary, topo2_per, topo2_sum = run_topology_eval(
        preds_dir,
        mapping_csv,
        metrics_dir,
        topology_split_file=topology_split,
        topology_gt_root=topology_gt_root,
    )

    topo2_row = list(csv.DictReader(topo2_sum.open("r", encoding="utf-8", newline="")))[0]

    final = {
        "experiment_name": exp_name,
        "model_name": model_name,
        "config": str(cfg_path.relative_to(ROOT)),
        "work_dir": str(work_dir.relative_to(ROOT)),
        "log_file": str(log_file.relative_to(ROOT)),
        "best_iter": int(best_iter),
        "best_checkpoint": str(best_ckpt.relative_to(ROOT)),
        "best_val_metrics": {k: float(v) for k, v in best_val_metrics.items()},
        "pred_dir": str(preds_dir.relative_to(ROOT)),
        "mapping_csv": str(mapping_csv.relative_to(ROOT)),
        "topology_csv": str(topo_per.relative_to(ROOT)),
        "topology_summary_csv": str(topo_summary.relative_to(ROOT)),
        "topology_v2_csv": str(topo2_per.relative_to(ROOT)),
        "topology_v2_summary_csv": str(topo2_sum.relative_to(ROOT)),
        "topology_v2_summary": {
            "mean_T_prec": float(topo2_row["mean_T_prec"]),
            "mean_T_sens": float(topo2_row["mean_T_sens"]),
            "mean_clDice": float(topo2_row["mean_clDice"]),
            "mean_Betti_Error": float(topo2_row["mean_Betti_Error"]),
            "mean_abs_dBeta0": float(topo2_row["mean_abs_dBeta0"]),
            "mean_abs_dBeta1": float(topo2_row["mean_abs_dBeta1"]),
            "samples": int(topo2_row["samples"]),
        },
    }

    out_json = metrics_dir / "final_summary.json"
    out_json.write_text(json.dumps(final, indent=2), encoding="utf-8")

    logging.info("[DONE] summary=%s", out_json)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
