#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Eval-only TP -> YellowBlock runner for external revise models."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.revise.train_revise_external import (  # noqa: E402
    TPIndexDataset,
    build_model,
    evaluate_model,
    export_predictions,
    load_py_config,
    set_seed,
)
from tools.eval_tp_topology import (  # noqa: E402
    cldice,
    count_connected_components,
    count_holes,
    dice_iou,
    largest_component_ratio,
    read_mask_as_binary,
)


def _resize_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    import cv2

    return cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)


def _run_topology_eval(pred_dir: Path, mapping_csv: Path, metrics_dir: Path) -> Tuple[Path, Path, Path, Path]:
    per_sample = metrics_dir / 'topology_per_sample.csv'
    summary = metrics_dir / 'topology_summary.csv'

    rows: List[Dict[str, object]] = []
    missing_gt = 0
    with mapping_csv.open('r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pred_path = pred_dir / r['pred_file']
            gt_path = Path(r['gt_file'])
            if not gt_path.exists():
                missing_gt += 1
                continue

            pred = read_mask_as_binary(pred_path)
            gt = read_mask_as_binary(gt_path)
            pred = _resize_to_gt(pred, gt)
            d, j = dice_iou(pred, gt)
            cd = cldice(pred, gt)

            rows.append({
                'idx': int(r['idx']),
                'part_id': r['part_id'],
                'pred_file': r['pred_file'],
                'gt_file': str(gt_path),
                'dice': d,
                'iou': j,
                'clDice': cd,
                'cc_pred': count_connected_components(pred),
                'cc_gt': count_connected_components(gt),
                'lcr_pred': largest_component_ratio(pred),
                'lcr_gt': largest_component_ratio(gt),
                'holes_pred': count_holes(pred),
                'holes_gt': count_holes(gt),
                'fgpix_pred': int(pred.sum()),
                'fgpix_gt': int(gt.sum()),
            })

    if not rows:
        raise RuntimeError(f'No valid prediction/gt pairs found in: {pred_dir}')

    fields = [
        'idx', 'part_id', 'pred_file', 'gt_file', 'dice', 'iou', 'clDice',
        'cc_pred', 'cc_gt', 'lcr_pred', 'lcr_gt', 'holes_pred', 'holes_gt',
        'fgpix_pred', 'fgpix_gt',
    ]
    with per_sample.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: int(x['idx'])):
            writer.writerow({k: (f'{r[k]:.12f}' if isinstance(r[k], float) else r[k]) for k in fields})

    metrics = {
        'samples': len(rows),
        'missing_gt': missing_gt,
        'mean_dice': float(np.mean([float(r['dice']) for r in rows])),
        'mean_iou': float(np.mean([float(r['iou']) for r in rows])),
        'mean_clDice': float(np.mean([float(r['clDice']) for r in rows])),
        'mean_cc_pred': float(np.mean([float(r['cc_pred']) for r in rows])),
        'mean_cc_gt': float(np.mean([float(r['cc_gt']) for r in rows])),
        'mean_lcr_pred': float(np.mean([float(r['lcr_pred']) for r in rows])),
        'mean_lcr_gt': float(np.mean([float(r['lcr_gt']) for r in rows])),
        'mean_holes_pred': float(np.mean([float(r['holes_pred']) for r in rows])),
        'mean_holes_gt': float(np.mean([float(r['holes_gt']) for r in rows])),
    }
    with summary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow({k: (f'{v:.12f}' if isinstance(v, float) else v) for k, v in metrics.items()})

    topo_v2_per = metrics_dir / 'topology_v2_per_sample.csv'
    topo_v2_sum = metrics_dir / 'topology_v2_summary.csv'
    subprocess.run([
        sys.executable,
        str(ROOT / 'tools/TP2Yellow/build_topology_v2_single.py'),
        '--pred-dir',
        str(pred_dir),
        '--mapping-csv',
        str(mapping_csv),
        '--out-csv',
        str(topo_v2_per),
        '--summary-csv',
        str(topo_v2_sum),
    ], cwd=str(ROOT), check=True)

    print(f'[OK] per-sample csv: {per_sample}')
    print(f'[OK] summary csv   : {summary}')
    print('[SUMMARY] N={samples} dice={mean_dice:.4f} iou={mean_iou:.4f} clDice={mean_clDice:.4f}'.format(**metrics))
    return per_sample, summary, topo_v2_per, topo_v2_sum


def _load_model_state(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> Dict:
    state = torch.load(checkpoint, map_location='cpu')
    model_state = state.get('model', state)
    model.load_state_dict(model_state)
    model.to(device)
    return state if isinstance(state, dict) else {}


def _augment_v2_with_lcr(work_dir: Path) -> None:
    topo_csv = work_dir / 'metrics' / 'topology_per_sample.csv'
    v2_csv = work_dir / 'metrics' / 'topology_v2_per_sample.csv'
    v2_summary = work_dir / 'metrics' / 'topology_v2_summary.csv'
    if not topo_csv.exists() or not v2_csv.exists():
        return

    with topo_csv.open('r', newline='', encoding='utf-8') as f:
        topo_rows = {r['idx']: r for r in csv.DictReader(f)}
    with v2_csv.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fields = [
        'idx', 'part_id', 'pred_file', 'gt_file', 'T_prec', 'T_sens', 'clDice',
        'CC_pred', 'CC_gt', 'LCR_pred', 'LCR_gt', 'abs_dLCR', 'Holes_pred',
        'Holes_gt', 'abs_dBeta0', 'abs_dBeta1', 'Betti_Error',
    ]
    out_rows: List[Dict[str, str]] = []
    for r in rows:
        t = topo_rows[r['idx']]
        lcr_pred = float(t['lcr_pred'])
        lcr_gt = float(t['lcr_gt'])
        out = dict(r)
        out['CC_pred'] = t['cc_pred']
        out['CC_gt'] = t['cc_gt']
        out['Holes_pred'] = t['holes_pred']
        out['Holes_gt'] = t['holes_gt']
        out['LCR_pred'] = f'{lcr_pred:.12f}'
        out['LCR_gt'] = f'{lcr_gt:.12f}'
        out['abs_dLCR'] = f'{abs(lcr_pred - lcr_gt):.12f}'
        out_rows.append(out)

    with v2_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r[k] for k in fields})

    summary = {
        'samples': len(out_rows),
        'mean_T_prec': float(np.mean([float(r['T_prec']) for r in out_rows])),
        'mean_T_sens': float(np.mean([float(r['T_sens']) for r in out_rows])),
        'mean_clDice': float(np.mean([float(r['clDice']) for r in out_rows])),
        'mean_LCR_pred': float(np.mean([float(r['LCR_pred']) for r in out_rows])),
        'mean_LCR_gt': float(np.mean([float(r['LCR_gt']) for r in out_rows])),
        'mean_abs_dLCR': float(np.mean([float(r['abs_dLCR']) for r in out_rows])),
        'mean_Betti_Error': float(np.mean([float(r['Betti_Error']) for r in out_rows])),
        'mean_abs_dBeta0': float(np.mean([float(r['abs_dBeta0']) for r in out_rows])),
        'mean_abs_dBeta1': float(np.mean([float(r['abs_dBeta1']) for r in out_rows])),
    }
    with v2_summary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow({k: (f'{v:.12f}' if isinstance(v, float) else v) for k, v in summary.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval external revise model on YellowBlock test split.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-root', default='data/YellowBlock-TP')
    parser.add_argument('--split-file', default='Index/test.txt')
    parser.add_argument('--img-subdir', default='JPEGImages')
    parser.add_argument('--gt-subdir', default='GroundTruth')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = work_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_py_config(cfg_path)
    model_name = cfg['model_name']
    repo_root = (ROOT / cfg['repo_root']).resolve()
    train_cfg = cfg['train']
    data_cfg = dict(cfg['data'])
    data_cfg['data_root'] = args.data_root
    data_cfg['val_index'] = args.split_file
    data_cfg['img_subdir'] = args.img_subdir
    data_cfg['gt_subdir'] = args.gt_subdir
    data_cfg['val_keep_original_size'] = bool(data_cfg.get('val_keep_original_size', False))
    if model_name in {'vmunet', 'emcad'}:
        data_cfg['val_keep_original_size'] = False

    set_seed(int(train_cfg.get('seed', 3407)))
    data_root = ROOT / data_cfg['data_root']
    test_set = TPIndexDataset(
        data_root=data_root,
        split_file=data_root / data_cfg['val_index'],
        image_size=tuple(data_cfg['image_size']),
        train=False,
        img_subdir=data_cfg['img_subdir'],
        gt_subdir=data_cfg['gt_subdir'],
        normalize=bool(data_cfg.get('normalize', True)),
        val_keep_original_size=bool(data_cfg.get('val_keep_original_size', False)),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(train_cfg.get('num_workers', 4)) // 2),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model(model_name, repo_root, dict(cfg['model_kwargs']))
    state = _load_model_state(model, checkpoint, device)

    seg_metrics = evaluate_model(model, model_name, test_loader, device)
    preds_dir = work_dir / 'preds_test'
    mapping_csv = export_predictions(model, model_name, test_loader, device, preds_dir)
    topo_per, topo_summary, topo2_per, topo2_sum = _run_topology_eval(preds_dir, mapping_csv, metrics_dir)
    _augment_v2_with_lcr(work_dir)

    with (metrics_dir / 'topology_summary.csv').open('r', newline='', encoding='utf-8') as f:
        topo_row = list(csv.DictReader(f))[0]
    with (metrics_dir / 'topology_v2_summary.csv').open('r', newline='', encoding='utf-8') as f:
        topo2_row = list(csv.DictReader(f))[0]

    final = {
        'experiment_name': cfg['experiment_name'],
        'model_name': model_name,
        'config': str(cfg_path.relative_to(ROOT)),
        'checkpoint': str(checkpoint.relative_to(ROOT)),
        'checkpoint_iter': state.get('iter', None),
        'work_dir': str(work_dir.relative_to(ROOT)),
        'split_file': str((data_root / data_cfg['val_index']).relative_to(ROOT)),
        'seg_metrics': {k: float(v) for k, v in seg_metrics.items()},
        'pred_dir': str(preds_dir.relative_to(ROOT)),
        'mapping_csv': str(mapping_csv.relative_to(ROOT)),
        'topology_csv': str(topo_per.relative_to(ROOT)),
        'topology_summary_csv': str(topo_summary.relative_to(ROOT)),
        'topology_v2_csv': str(topo2_per.relative_to(ROOT)),
        'topology_v2_summary_csv': str(topo2_sum.relative_to(ROOT)),
        'topology_summary': {k: float(v) if k not in {'samples', 'missing_gt'} else int(float(v)) for k, v in topo_row.items()},
        'topology_v2_summary': {k: float(v) if k != 'samples' else int(float(v)) for k, v in topo2_row.items()},
    }
    out_json = metrics_dir / 'final_summary.json'
    out_json.write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
