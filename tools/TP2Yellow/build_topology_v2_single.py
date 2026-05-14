#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build single-model topology-v2 metrics with LCR columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tools.build_topo_v2_metrics import (
    count_connected_components,
    count_holes,
    read_mask_as_binary,
    tprec_tsens_cldice,
)
from tools.eval_tp_topology import largest_component_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build topology-v2 metrics for one prediction folder.')
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--mapping-csv', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--summary-csv', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    mapping_csv = Path(args.mapping_csv)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with mapping_csv.open('r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pred_path = pred_dir / r['pred_file']
            gt_path = Path(r['gt_file'])
            pred = read_mask_as_binary(pred_path)
            gt = read_mask_as_binary(gt_path)
            if pred.shape != gt.shape:
                import cv2
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            t_prec, t_sens, cldice = tprec_tsens_cldice(pred, gt)
            cc_pred = count_connected_components(pred)
            cc_gt = count_connected_components(gt)
            holes_pred = count_holes(pred)
            holes_gt = count_holes(gt)
            lcr_pred = largest_component_ratio(pred)
            lcr_gt = largest_component_ratio(gt)
            db0 = abs(cc_pred - cc_gt)
            db1 = abs(holes_pred - holes_gt)

            rows.append({
                'idx': r['idx'],
                'part_id': r['part_id'],
                'pred_file': r['pred_file'],
                'gt_file': r['gt_file'],
                'T_prec': t_prec,
                'T_sens': t_sens,
                'clDice': cldice,
                'CC_pred': cc_pred,
                'CC_gt': cc_gt,
                'LCR_pred': lcr_pred,
                'LCR_gt': lcr_gt,
                'abs_dLCR': abs(lcr_pred - lcr_gt),
                'Holes_pred': holes_pred,
                'Holes_gt': holes_gt,
                'abs_dBeta0': db0,
                'abs_dBeta1': db1,
                'Betti_Error': db0 + db1,
            })

    fields = [
        'idx', 'part_id', 'pred_file', 'gt_file', 'T_prec', 'T_sens', 'clDice',
        'CC_pred', 'CC_gt', 'LCR_pred', 'LCR_gt', 'abs_dLCR', 'Holes_pred',
        'Holes_gt', 'abs_dBeta0', 'abs_dBeta1', 'Betti_Error',
    ]
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f'{r[k]:.12f}' if isinstance(r[k], float) else r[k]) for k in fields})

    summary = {
        'samples': len(rows),
        'mean_T_prec': float(np.mean([r['T_prec'] for r in rows])),
        'mean_T_sens': float(np.mean([r['T_sens'] for r in rows])),
        'mean_clDice': float(np.mean([r['clDice'] for r in rows])),
        'mean_LCR_pred': float(np.mean([r['LCR_pred'] for r in rows])),
        'mean_LCR_gt': float(np.mean([r['LCR_gt'] for r in rows])),
        'mean_abs_dLCR': float(np.mean([r['abs_dLCR'] for r in rows])),
        'mean_Betti_Error': float(np.mean([r['Betti_Error'] for r in rows])),
        'mean_abs_dBeta0': float(np.mean([r['abs_dBeta0'] for r in rows])),
        'mean_abs_dBeta1': float(np.mean([r['abs_dBeta1'] for r in rows])),
    }
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow({k: (f'{v:.12f}' if isinstance(v, float) else v) for k, v in summary.items()})

    print(f'[OK] per-sample csv: {out_csv}')
    print(f'[OK] summary csv   : {summary_csv}')
    print('[SUMMARY] N={samples} T_prec={mean_T_prec:.4f} T_sens={mean_T_sens:.4f} clDice={mean_clDice:.4f} LCR={mean_LCR_pred:.4f}'.format(**summary))


if __name__ == '__main__':
    main()
