#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]

DISPLAY_NAME = {
    "vmunet": "VM-UNet",
    "emcad": "EMCAD",
    "bevanet_s": "BEVANet-S",
}


def parse_existing_table1(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    rows: List[Dict[str, str]] = []
    for ln in text.splitlines():
        if not ln.strip().startswith("|"):
            continue
        if "Model" in ln or "---" in ln:
            continue
        parts = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "Model": parts[0],
                "Best step": parts[1],
                "aAcc": parts[2],
                "mIoU": parts[3],
                "mDice": parts[4],
                "mFscore": parts[5],
                "mPrecision": parts[6],
                "mRecall": parts[7],
            }
        )
    return rows


def read_topo_v2_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_new_summaries(pattern_root: Path) -> List[Dict]:
    files = sorted(pattern_root.glob("*/metrics/final_summary.json"))
    out = []
    for p in files:
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: Path, title: str, rows: List[Dict[str, str]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r[c] for c in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build revised table1/table2 with three new control models.")
    p.add_argument("--existing-table1", default="outputs/table1.txt")
    p.add_argument("--existing-topov2", default="docs/tables/topo_v2/topology_v2_summary.csv")
    p.add_argument("--new-summary-root", default="work_dirs/revise")
    p.add_argument("--out-dir", default="docs/revise")
    args = p.parse_args()

    table1_rows = parse_existing_table1(ROOT / args.existing_table1)
    topo_old_rows = read_topo_v2_csv(ROOT / args.existing_topov2)
    new_summaries = read_new_summaries(ROOT / args.new_summary_root)

    # Append 3 new models to table1.
    for s in new_summaries:
        mname = s["model_name"]
        if mname not in DISPLAY_NAME:
            continue
        met = s["best_val_metrics"]
        table1_rows.append(
            {
                "Model": DISPLAY_NAME[mname],
                "Best step": str(s["best_iter"]),
                "aAcc": f"{met['aAcc'] * 100:.4f}",
                "mIoU": f"{met['mIoU'] * 100:.4f}",
                "mDice": f"{met['mDice'] * 100:.4f}",
                "mFscore": f"{met['mFscore'] * 100:.4f}",
                "mPrecision": f"{met['mPrecision'] * 100:.4f}",
                "mRecall": f"{met['mRecall'] * 100:.4f}",
            }
        )

    table2_rows: List[Dict[str, str]] = []
    for r in topo_old_rows:
        table2_rows.append(
            {
                "Model": r["model"],
                "T_prec": f"{float(r['mean_T_prec']) * 100:.4f}",
                "T_sens": f"{float(r['mean_T_sens']) * 100:.4f}",
                "clDice": f"{float(r['mean_clDice']) * 100:.4f}",
                "Betti_Error": f"{float(r['mean_Betti_Error']):.4f}",
            }
        )

    for s in new_summaries:
        mname = s["model_name"]
        if mname not in DISPLAY_NAME:
            continue
        t2 = s["topology_v2_summary"]
        table2_rows.append(
            {
                "Model": DISPLAY_NAME[mname],
                "T_prec": f"{float(t2['mean_T_prec']) * 100:.4f}",
                "T_sens": f"{float(t2['mean_T_sens']) * 100:.4f}",
                "clDice": f"{float(t2['mean_clDice']) * 100:.4f}",
                "Betti_Error": f"{float(t2['mean_Betti_Error']):.4f}",
            }
        )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_cols = ["Model", "Best step", "aAcc", "mIoU", "mDice", "mFscore", "mPrecision", "mRecall"]
    t2_cols = ["Model", "T_prec", "T_sens", "clDice", "Betti_Error"]

    write_csv(out_dir / "table1_revise.csv", table1_rows, t1_cols)
    write_md(out_dir / "table1_revise.md", "Revised Table 1 (Region Metrics)", table1_rows, t1_cols)

    write_csv(out_dir / "table2_revise.csv", table2_rows, t2_cols)
    write_md(out_dir / "table2_revise.md", "Revised Table 2 (Topology Metrics)", table2_rows, t2_cols)

    print(f"[OK] wrote {(out_dir / 'table1_revise.csv').relative_to(ROOT)}")
    print(f"[OK] wrote {(out_dir / 'table1_revise.md').relative_to(ROOT)}")
    print(f"[OK] wrote {(out_dir / 'table2_revise.csv').relative_to(ROOT)}")
    print(f"[OK] wrote {(out_dir / 'table2_revise.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
