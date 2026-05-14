#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the TP -> YellowBlock generalization summary table."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


ROWS = [
    {
        'method': 'PSPNet',
        'backbone': 'ResNet-50',
        'key': 'pspnet_r50_tp9000_yellowblock',
        'config': 'configs/TP2Yellow/pspnet_r50_tp9000_yellowblock_test.py',
        'checkpoint': 'work_dirs/pspnet_r50_tp_main_table1_B/ckpt/pspnet_r50_tp_main_table1_B/best_mIoU_iter_9000.pth',
        'log': 'logs/TP2Yellow/test_pspnet_r50_tp9000_yellowblock.log',
    },
    {
        'method': 'DeepLabV3+',
        'backbone': 'ResNet-50',
        'key': 'deeplabv3plus_r50_tp9000_yellowblock',
        'config': 'configs/TP2Yellow/deeplabv3plus_r50_tp9000_yellowblock_test.py',
        'checkpoint': 'work_dirs/deeplabv3plus_r50_tp_main_table1_B/ckpt/deeplabv3plus_r50_tp_main_table1_B/best_mIoU_iter_9000.pth',
        'log': 'logs/TP2Yellow/test_deeplabv3plus_r50_tp9000_yellowblock.log',
    },
    {
        'method': 'UPerNet',
        'backbone': 'ResNet-50',
        'key': 'upernet_r50_tp6000_yellowblock',
        'config': 'configs/TP2Yellow/upernet_r50_tp6000_yellowblock_test.py',
        'checkpoint': 'work_dirs/upernet_r50_tp_main_table1_B/ckpt/upernet_r50_tp_main_table1_B/best_mIoU_iter_6000.pth',
        'log': 'logs/TP2Yellow/test_upernet_r50_tp6000_yellowblock.log',
    },
    {
        'method': 'VM-UNet',
        'backbone': 'VMamba',
        'key': 'vmunet_tp_revise_yellowblock',
        'config': 'configs/tp_dataset/vmunet_tp_revise.py',
        'checkpoint': 'work_dirs/revise/vmunet_tp_revise/ckpt/best_mIoU.pth',
        'external': True,
    },
    {
        'method': 'EMCAD',
        'backbone': 'PVTv2-B0',
        'key': 'emcad_tp_revise_yellowblock',
        'config': 'configs/tp_dataset/emcad_tp_revise.py',
        'checkpoint': 'work_dirs/revise/emcad_tp_revise/ckpt/best_mIoU.pth',
        'external': True,
    },
    {
        'method': 'BEVANet-S',
        'backbone': 'BEVANet-S',
        'key': 'bevanet_s_tp_revise_yellowblock',
        'config': 'configs/tp_dataset/bevanet_s_tp_revise.py',
        'checkpoint': 'work_dirs/revise/bevanet_s_tp_revise/ckpt/best_mIoU.pth',
        'external': True,
    },
    {
        'method': 'SegFormer-B2',
        'backbone': 'MiT-B2',
        'key': 'segformer_b2_tp8500_yellowblock',
        'config': 'configs/TP2Yellow/segformer_b2_tp8500_yellowblock_test.py',
        'checkpoint': 'work_dirs/segformer_b2_tp/iter_8500.pth',
        'log': 'logs/TP2Yellow/test_segformer_b2_tp8500_yellowblock.log',
    },
    {
        'method': 'Ours (+Skeleton)',
        'backbone': 'MiT-B2',
        'key': 'core_none_tp6500_yellowblock',
        'config': 'configs/TP2Yellow/core_none_tp6500_yellowblock_test.py',
        'checkpoint': 'work_dirs/ablation/core_none/ckpt/core_none/iter_6500.pth',
        'log': 'logs/TP2Yellow/test_core_none_tp6500_yellowblock.log',
    },
    {
        'method': 'Ours (+Skeleton+clDice)',
        'backbone': 'MiT-B2',
        'key': 'core_both_tp9500_yellowblock',
        'config': 'configs/TP2Yellow/core_both_tp9500_yellowblock_test.py',
        'checkpoint': 'work_dirs/ablation/core_both/ckpt/core_both/best_mIoU_iter_9500.pth',
        'log': 'logs/TP2Yellow/test_core_both_tp9500_yellowblock.log',
    },
]

SEG_KEYS = ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall']
PERCENT_KEYS = SEG_KEYS + [
    'Topo_Dice', 'Topo_IoU', 'T_prec', 'T_sens', 'clDice',
    'LCR', 'LCR_pred', 'LCR_gt', 'abs_dLCR',
]
COUNT_KEYS = [
    'Betti_Error', 'abs_dBeta0', 'abs_dBeta1',
    'CC_pred', 'CC_gt', 'Holes_pred', 'Holes_gt',
]


def read_csv_row(path: Path) -> Dict[str, str]:
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'Empty CSV: {path}')
    return rows[0]


def parse_mmseg_log(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding='utf-8', errors='ignore')
    matches = re.findall(
        r'Iter\(test\).*?aAcc:\s*([0-9.]+)\s+'
        r'mIoU:\s*([0-9.]+)\s+'
        r'mAcc:\s*([0-9.]+)\s+'
        r'mDice:\s*([0-9.]+)\s+'
        r'mFscore:\s*([0-9.]+)\s+'
        r'mPrecision:\s*([0-9.]+)\s+'
        r'mRecall:\s*([0-9.]+)',
        text,
    )
    if not matches:
        raise RuntimeError(f'Cannot parse mmseg metrics: {path}')
    vals = [float(v) for v in matches[-1]]
    return dict(zip(SEG_KEYS, vals))


def fmt_pct(raw01: str) -> str:
    return f'{float(raw01) * 100.0:.2f}'


def fmt_count(value: str) -> str:
    return f'{float(value):.2f}'


def build_rows() -> List[Dict[str, str]]:
    out_rows = []
    for spec in ROWS:
        key = spec['key']
        work_dir = ROOT / 'work_dirs/TP2Yellow' / key

        if spec.get('external'):
            final = json.loads((work_dir / 'metrics/final_summary.json').read_text(encoding='utf-8'))
            seg = {k: float(final['seg_metrics'][k]) * 100.0 for k in SEG_KEYS}
            topo = final['topology_summary']
            topo2 = final['topology_v2_summary']
        else:
            seg = parse_mmseg_log(ROOT / spec['log'])
            topo = read_csv_row(work_dir / 'topology_test_summary.csv')
            topo2 = read_csv_row(work_dir / 'topology_v2_test_summary.csv')

        row = {
            'Method': spec['method'],
            'Backbone': spec['backbone'],
            'Samples': str(int(float(topo['samples']))),
            'Config': spec['config'],
            'Checkpoint': spec['checkpoint'],
            'Work_dir': f'work_dirs/TP2Yellow/{key}',
        }
        for k in SEG_KEYS:
            row[k] = f'{seg[k]:.2f}'

        row['Topo_Dice'] = fmt_pct(topo['mean_dice'])
        row['Topo_IoU'] = fmt_pct(topo['mean_iou'])
        row['T_prec'] = fmt_pct(topo2['mean_T_prec'])
        row['T_sens'] = fmt_pct(topo2['mean_T_sens'])
        row['clDice'] = fmt_pct(topo2['mean_clDice'])
        row['LCR'] = fmt_pct(topo2['mean_LCR_pred'])
        row['LCR_pred'] = fmt_pct(topo2['mean_LCR_pred'])
        row['LCR_gt'] = fmt_pct(topo2['mean_LCR_gt'])
        row['abs_dLCR'] = fmt_pct(topo2['mean_abs_dLCR'])
        row['Betti_Error'] = fmt_count(topo2['mean_Betti_Error'])
        row['abs_dBeta0'] = fmt_count(topo2['mean_abs_dBeta0'])
        row['abs_dBeta1'] = fmt_count(topo2['mean_abs_dBeta1'])
        row['CC_pred'] = fmt_count(topo['mean_cc_pred'])
        row['CC_gt'] = fmt_count(topo['mean_cc_gt'])
        row['Holes_pred'] = fmt_count(topo['mean_holes_pred'])
        row['Holes_gt'] = fmt_count(topo['mean_holes_gt'])
        out_rows.append(row)
    return out_rows


def write_markdown(rows: List[Dict[str, str]], path: Path) -> None:
    md_cols = [
        'Method', 'Backbone', 'aAcc', 'mIoU', 'mDice', 'Topo_Dice',
        'Topo_IoU', 'T_prec', 'T_sens', 'clDice', 'LCR',
        'Betti_Error', 'CC_pred', 'Holes_pred',
    ]
    lines = []
    lines.append('| ' + ' | '.join(md_cols) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(md_cols)) + ' |')
    for r in rows:
        lines.append('| ' + ' | '.join(r[c] for c in md_cols) + ' |')
    lines.append('')
    lines.append('Note: percentage columns are reported in %. LCR follows the paper Table2 convention, i.e. LCR = LCR_pred.')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    out_dir = ROOT / 'docs/TP2Yellow'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    fieldnames = [
        'Method', 'Backbone', 'Samples',
        *SEG_KEYS,
        'Topo_Dice', 'Topo_IoU', 'T_prec', 'T_sens', 'clDice',
        'LCR', 'LCR_pred', 'LCR_gt', 'abs_dLCR',
        *COUNT_KEYS,
        'Config', 'Checkpoint', 'Work_dir',
    ]
    csv_path = out_dir / 'tp2yellow_generalization_table.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / 'tp2yellow_generalization_table.md'
    write_markdown(rows, md_path)
    print(f'[OK] {csv_path}')
    print(f'[OK] {md_path}')


if __name__ == '__main__':
    main()
