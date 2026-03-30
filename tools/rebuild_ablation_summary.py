#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

CASE_DEFS: Dict[str, Tuple[float, float]] = {
    "core_none": (0.0, 0.0),
    "core_main_only": (0.1, 0.0),
    "core_aux_only": (0.0, 0.1),
    "core_both": (0.1, 0.1),
    "w_0p05": (0.05, 0.05),
    "w_0p10": (0.10, 0.10),
    "w_0p20": (0.20, 0.20),
    "w_0p30": (0.30, 0.30),
}
TARGET_ITERS = 10000


def max_iter_in_ckpt(work_dir: Path) -> int:
    max_iter = 0
    for p in work_dir.glob("ckpt/**/iter_*.pth"):
        m = re.search(r"iter_(\d+)\.pth$", str(p))
        if m:
            max_iter = max(max_iter, int(m.group(1)))
    return max_iter


def best_from_scalars(work_dir: Path) -> Dict[str, str]:
    best_val = -1.0
    best_step = ""
    best_row: Dict[str, str] = {}
    best_file = ""

    for sf in sorted(work_dir.glob("*/vis_data/scalars.json")):
        for line in sf.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "mIoU" not in obj or "step" not in obj:
                continue
            v = float(obj["mIoU"])
            if v > best_val:
                best_val = v
                best_step = str(int(obj["step"]))
                best_file = str(sf.relative_to(ROOT))
                best_row = {
                    "aAcc": f"{obj.get('aAcc', '')}",
                    "mIoU": f"{obj.get('mIoU', '')}",
                    "mAcc": f"{obj.get('mAcc', '')}",
                    "mDice": f"{obj.get('mDice', '')}",
                    "mFscore": f"{obj.get('mFscore', '')}",
                    "mPrecision": f"{obj.get('mPrecision', '')}",
                    "mRecall": f"{obj.get('mRecall', '')}",
                }

    if best_val < 0:
        return {"status": "no_scalar_best"}

    out = {
        "status": "ok_from_scalars",
        "best_iter": best_step,
        "best_mIoU": f"{best_val:.4f}",
        "best_source": best_file,
    }
    out.update(best_row)
    return out


def case_logs(case: str) -> List[Path]:
    return sorted((ROOT / "logs" / "ablation").glob(f"train_{case}_*.log"))


def main() -> None:
    out_csv = ROOT / "work_dirs" / "ablation" / "ablation_results_master.csv"
    logs_csv = ROOT / "work_dirs" / "ablation" / "ablation_logs_index.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    log_rows: List[Dict[str, str]] = []

    for case, (d_w, a_w) in CASE_DEFS.items():
        wd = ROOT / "work_dirs" / "ablation" / case
        max_iter = max_iter_in_ckpt(wd)
        best = best_from_scalars(wd)

        logs = case_logs(case)
        for lp in logs:
            log_rows.append(
                {
                    "name": case,
                    "log": str(lp.relative_to(ROOT)),
                    "mtime": str(int(lp.stat().st_mtime)),
                    "size": str(lp.stat().st_size),
                }
            )

        rec: Dict[str, str] = {
            "name": case,
            "decode_cldice_weight": f"{d_w}",
            "aux_cldice_weight": f"{a_w}",
            "work_dir": str(wd.relative_to(ROOT)),
            "max_iter_ckpt": str(max_iter),
            "is_complete": "yes" if max_iter >= TARGET_ITERS else "no",
            "latest_log": str(logs[-1].relative_to(ROOT)) if logs else "",
            "log_count": str(len(logs)),
        }
        rec.update(best)
        rows.append(rec)

    def write(path: Path, data: List[Dict[str, str]]) -> None:
        fields = sorted({k for r in data for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in data:
                w.writerow(r)

    write(out_csv, rows)
    write(logs_csv, log_rows)
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {logs_csv}")


if __name__ == "__main__":
    main()
