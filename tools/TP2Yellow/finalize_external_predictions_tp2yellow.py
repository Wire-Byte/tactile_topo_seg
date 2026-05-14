#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Finalize TP -> YellowBlock metrics from exported external-model predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.TP2Yellow.eval_revise_external_tp2yellow import _run_topology_eval  # noqa: E402
from tools.eval_tp_topology import read_mask_as_binary  # noqa: E402
from tools.revise.train_revise_external import _fast_hist, _seg_metrics_from_hist  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build final metrics JSON from pred_mapping.csv.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-iter', type=int, default=None)
    parser.add_argument('--split-file', default='data/YellowBlock-TP/Index/test.txt')
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_first_row(path: Path) -> Dict[str, str]:
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'Empty CSV: {path}')
    return rows[0]


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    pred_dir = work_dir / 'preds_test'
    mapping_csv = pred_dir / 'pred_mapping.csv'
    metrics_dir = work_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    hist = np.zeros((2, 2), dtype=np.float64)
    with mapping_csv.open('r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pred = read_mask_as_binary(pred_dir / r['pred_file'])
            gt = read_mask_as_binary(Path(r['gt_file']))
            if pred.shape != gt.shape:
                import cv2

                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            hist += _fast_hist(pred, gt, n=2)

    seg_metrics = _seg_metrics_from_hist(hist)
    topo_per, topo_summary, topo2_per, topo2_sum = _run_topology_eval(pred_dir, mapping_csv, metrics_dir)

    topo_row = _read_first_row(topo_summary)
    topo2_row = _read_first_row(topo2_sum)
    final = {
        'experiment_name': args.experiment_name,
        'model_name': args.model_name,
        'config': _relative(ROOT / args.config),
        'checkpoint': _relative(ROOT / args.checkpoint),
        'checkpoint_iter': args.checkpoint_iter,
        'work_dir': _relative(work_dir),
        'split_file': args.split_file,
        'seg_metrics': {k: float(v) for k, v in seg_metrics.items()},
        'pred_dir': _relative(pred_dir),
        'mapping_csv': _relative(mapping_csv),
        'topology_csv': _relative(topo_per),
        'topology_summary_csv': _relative(topo_summary),
        'topology_v2_csv': _relative(topo2_per),
        'topology_v2_summary_csv': _relative(topo2_sum),
        'topology_summary': {k: float(v) if k not in {'samples', 'missing_gt'} else int(float(v)) for k, v in topo_row.items()},
        'topology_v2_summary': {k: float(v) if k != 'samples' else int(float(v)) for k, v in topo2_row.items()},
    }
    out_json = metrics_dir / 'final_summary.json'
    out_json.write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
