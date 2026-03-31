#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
MMSegTrain = ROOT / "third_party" / "mmsegmentation" / "tools" / "train.py"
REBUILD_SCRIPT = ROOT / "tools" / "rebuild_ablation_summary.py"
EXPORT_SCRIPT = ROOT / "tools" / "export_preds_unique_by_index.py"
TOPO_SCRIPT = ROOT / "tools" / "eval_tp_topology.py"

MASTER_CSV = ROOT / "work_dirs" / "ablation" / "ablation_results_master.csv"
TARGET_ITERS = 10000

CASE_CFG: Dict[str, Path] = {
    "core_none": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "core_none.py",
    "core_main_only": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "core_main_only.py",
    "core_aux_only": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "core_aux_only.py",
    "core_both": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "core_both.py",
    "w_0p05": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "w_0p05.py",
    "w_0p10": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "w_0p10.py",
    "w_0p20": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "w_0p20.py",
    "w_0p30": ROOT / "configs" / "tp_dataset" / "ablation_generated" / "w_0p30.py",
}


def run_cmd(cmd: List[str], log_fp: Path, env: Optional[Dict[str, str]] = None, append: bool = True) -> int:
    log_fp.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_fp.open(mode, encoding="utf-8") as f:
        f.write("\n")
        f.write(f"===== CMD {time.strftime('%F %T')} =====\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def active_cmdlines() -> List[str]:
    out: List[str] = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        p = Path("/proc") / d / "cmdline"
        try:
            b = p.read_bytes()
        except Exception:
            continue
        if not b:
            continue
        s = b.replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
        if s:
            out.append(s)
    return out


def has_active_case(case: str) -> bool:
    needle = f"ablation_generated/{case}.py"
    for cmd in active_cmdlines():
        if needle in cmd and "tools/train.py" in cmd:
            return True
    return False


def has_any_active_ablation_train() -> bool:
    for cmd in active_cmdlines():
        if "tools/train.py" in cmd and "configs/tp_dataset/ablation_generated/" in cmd:
            return True
    return False


def max_iter_in_ckpt(work_dir: Path) -> int:
    max_iter = 0
    for p in work_dir.glob("ckpt/**/iter_*.pth"):
        m = re.search(r"iter_(\d+)\.pth$", str(p))
        if m:
            max_iter = max(max_iter, int(m.group(1)))
    return max_iter


def find_case_logs(case: str) -> List[Path]:
    return sorted((ROOT / "logs" / "ablation").glob(f"train_{case}_*.log"))


def pick_canonical_case_log(case: str) -> Path:
    logs = find_case_logs(case)
    if logs:
        return logs[0]
    return ROOT / "logs" / "ablation" / f"train_{case}.log"


def _safe_state_name(path: Path) -> str:
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{path.name}.{h}.json"


def sync_case_logs_incremental(case: str, canonical_log: Path, state_root: Path) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    state_file = state_root / _safe_state_name(canonical_log)
    state: Dict[str, int] = {}
    if state_file.exists():
        try:
            state = {k: int(v) for k, v in json.loads(state_file.read_text(encoding="utf-8")).items()}
        except Exception:
            state = {}

    with canonical_log.open("a", encoding="utf-8") as out:
        for lp in find_case_logs(case):
            if lp.resolve() == canonical_log.resolve():
                continue
            key = str(lp)
            old_pos = int(state.get(key, 0))
            size = lp.stat().st_size
            if size <= old_pos:
                continue
            with lp.open("r", encoding="utf-8", errors="ignore") as src:
                src.seek(old_pos)
                chunk = src.read()
            if old_pos == 0:
                out.write(f"\n===== IMPORTED_STREAM {lp.name} @ {time.strftime('%F %T')} =====\n")
            out.write(chunk)
            state[key] = size

    state_file.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def rebuild_master(guard_log: Path) -> None:
    cmd = ["python", "-u", str(REBUILD_SCRIPT)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'third_party' / 'mmsegmentation'}:{env.get('PYTHONPATH', '')}"
    run_cmd(cmd, guard_log, env=env, append=True)


def read_master_rows() -> List[Dict[str, str]]:
    if not MASTER_CSV.exists():
        return []
    with MASTER_CSV.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def incomplete_cases(rows: List[Dict[str, str]], scope_cases: List[str]) -> List[str]:
    by_name = {r.get("name", ""): r for r in rows}
    out: List[str] = []
    for c in scope_cases:
        r = by_name.get(c)
        if not r:
            out.append(c)
            continue
        if (r.get("is_complete", "").strip().lower() != "yes"):
            out.append(c)
    return out


def pick_best_checkpoint(ckpt_dir: Path) -> Path:
    bests = sorted(ckpt_dir.glob("best_mIoU_iter_*.pth"))
    if bests:
        # choose highest iter best checkpoint
        def key(p: Path) -> int:
            m = re.search(r"iter_(\d+)\.pth$", p.name)
            return int(m.group(1)) if m else -1

        return sorted(bests, key=key)[-1]

    last = ckpt_dir / "iter_10000.pth"
    if last.exists():
        return last

    candidates = sorted(ckpt_dir.glob("iter_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")

    def key2(p: Path) -> int:
        m = re.search(r"iter_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=key2)[-1]


def run_table2(out_root: Path, guard_log: Path) -> None:
    table2_root = out_root / "table2_topology"
    pred_root = table2_root / "preds"
    csv_root = table2_root / "csv"
    log_root = table2_root / "logs"

    table2_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    models = [
        {
            "name": "deeplabv3plus_r50",
            "config": ROOT / "configs" / "tp_dataset" / "deeplabv3plus_r50_tp_main_table1_B.py",
            "ckpt_dir": ROOT / "work_dirs" / "deeplabv3plus_r50_tp_main_table1_B" / "ckpt" / "deeplabv3plus_r50_tp_main_table1_B",
            "note": "Table2 control baseline (DeepLabV3+)"
        },
        {
            "name": "pspnet_r50",
            "config": ROOT / "configs" / "tp_dataset" / "pspnet_r50_tp_main_table1_B.py",
            "ckpt_dir": ROOT / "work_dirs" / "pspnet_r50_tp_main_table1_B" / "ckpt" / "pspnet_r50_tp_main_table1_B",
            "note": "Table2 control baseline (PSPNet)"
        },
        {
            "name": "upernet_r50",
            "config": ROOT / "configs" / "tp_dataset" / "upernet_r50_tp_main_table1_B.py",
            "ckpt_dir": ROOT / "work_dirs" / "upernet_r50_tp_main_table1_B" / "ckpt" / "upernet_r50_tp_main_table1_B",
            "note": "Table2 control baseline (UPerNet)"
        },
        {
            "name": "core_innovation_segformer_b2_skel_cldice_v2",
            "config": ROOT / "configs" / "tp_dataset" / "segformer_b2_tp_skel_clDice_v2_4090_bs4.py",
            "ckpt_dir": ROOT / "work_dirs" / "segformer_b2_tp_skel_cldice_v2_4090_bs4" / "ckpt" / "segformer_b2_tp_skel_cldice_v2_4090_bs4",
            "note": "Core innovation model (ours)"
        },
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'third_party' / 'mmsegmentation'}:{env.get('PYTHONPATH', '')}"

    rows: List[Dict[str, str]] = []

    for m in models:
        ckpt = pick_best_checkpoint(m["ckpt_dir"])
        pred_dir = pred_root / m["name"]
        mapping_csv = pred_dir / "pred_mapping.csv"
        per_csv = csv_root / f"{m['name']}_topology_per_sample.csv"
        sum_csv = csv_root / f"{m['name']}_topology_summary.csv"
        m_log = log_root / f"{m['name']}.log"

        cmd_export = [
            "python", "-u", str(EXPORT_SCRIPT),
            "--config", str(m["config"]),
            "--checkpoint", str(ckpt),
            "--out-dir", str(pred_dir),
            "--mapping-csv", str(mapping_csv),
            "--device", "cuda:0",
        ]
        rc1 = run_cmd(cmd_export, m_log, env=env, append=True)
        if rc1 != 0:
            raise RuntimeError(f"Export failed for {m['name']}, see {m_log}")

        cmd_topo = [
            "python", "-u", str(TOPO_SCRIPT),
            "--pred-dir", str(pred_dir),
            "--out-csv", str(per_csv),
            "--summary-csv", str(sum_csv),
        ]
        rc2 = run_cmd(cmd_topo, m_log, env=env, append=True)
        if rc2 != 0:
            raise RuntimeError(f"Topology eval failed for {m['name']}, see {m_log}")

        with sum_csv.open("r", newline="", encoding="utf-8") as f:
            summary = next(csv.DictReader(f))

        rows.append({
            "model": m["name"],
            "config": str(m["config"].relative_to(ROOT)),
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "samples": summary.get("samples", ""),
            "mean_dice": summary.get("mean_dice", ""),
            "mean_iou": summary.get("mean_iou", ""),
            "mean_clDice": summary.get("mean_clDice", ""),
            "mean_cc_pred": summary.get("mean_cc_pred", ""),
            "mean_lcr_pred": summary.get("mean_lcr_pred", ""),
            "mean_holes_pred": summary.get("mean_holes_pred", ""),
            "notes": m["note"],
            "per_sample_csv": str(per_csv.relative_to(ROOT)),
            "summary_csv": str(sum_csv.relative_to(ROOT)),
            "log": str(m_log.relative_to(ROOT)),
        })

    table2_csv = table2_root / "table2_topology_metrics.csv"
    with table2_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    table2_md = table2_root / "table2_topology_metrics.md"
    with table2_md.open("w", encoding="utf-8") as f:
        f.write("# Table2 Topology Metrics\n\n")
        f.write("说明:\n")
        f.write("- mean_dice/mean_iou/mean_clDice: 拓扑评估脚本在验证集上的平均指标。\n")
        f.write("- mean_cc_pred: 预测前景连通域平均数，越接近目标结构通常越好。\n")
        f.write("- mean_lcr_pred: 最大连通域占预测前景比例。\n")
        f.write("- mean_holes_pred: 预测孔洞平均数。\n\n")
        f.write("| model | mean_dice | mean_iou | mean_clDice | mean_cc_pred | mean_lcr_pred | mean_holes_pred | checkpoint | notes |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | {r['mean_dice']} | {r['mean_iou']} | {r['mean_clDice']} | "
                f"{r['mean_cc_pred']} | {r['mean_lcr_pred']} | {r['mean_holes_pred']} | "
                f"{r['checkpoint']} | {r['notes']} |\n"
            )

    with guard_log.open("a", encoding="utf-8") as f:
        f.write(f"\n[OK] table2 csv: {table2_csv}\n")
        f.write(f"[OK] table2 md : {table2_md}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard ablation completion and then run table2 topology eval.")
    parser.add_argument(
        "--cases",
        default="core_main_only,w_0p20,w_0p30",
        help="Comma-separated cases that must reach completion.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=90,
        help="Polling interval while waiting for active external trainer.",
    )
    parser.add_argument(
        "--max-retries-per-case",
        type=int,
        default=30,
        help="Max retries for each case if interrupted.",
    )
    parser.add_argument(
        "--out-root",
        default="work_dirs",
        help="Root output dir for final table2 artifacts.",
    )
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in CASE_CFG:
            raise ValueError(f"Unknown case: {c}")

    guard_log = ROOT / "logs" / "ablation" / f"guard_ablation_table2_{time.strftime('%Y%m%d_%H%M%S')}.log"
    guard_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'third_party' / 'mmsegmentation'}:{env.get('PYTHONPATH', '')}"

    with guard_log.open("a", encoding="utf-8") as f:
        f.write(f"[START] {time.strftime('%F %T')} guard started\n")
        f.write(f"[CASES] {','.join(cases)}\n")

    state_root = ROOT / "logs" / "ablation" / ".guard_state"

    for case in cases:
        canonical_log = pick_canonical_case_log(case)
        work_dir = ROOT / "work_dirs" / "ablation" / case
        cfg = CASE_CFG[case]
        retries = 0

        sync_case_logs_incremental(case, canonical_log, state_root)

        while max_iter_in_ckpt(work_dir) < TARGET_ITERS:
            sync_case_logs_incremental(case, canonical_log, state_root)

            if has_any_active_ablation_train():
                with guard_log.open("a", encoding="utf-8") as f:
                    f.write(f"[WAIT] {time.strftime('%F %T')} another ablation train is active; hold {case}\n")
                time.sleep(args.poll_seconds)
                continue

            if has_active_case(case):
                with guard_log.open("a", encoding="utf-8") as f:
                    f.write(f"[WAIT] {time.strftime('%F %T')} active external trainer for {case}\n")
                time.sleep(args.poll_seconds)
                continue

            if retries >= args.max_retries_per_case:
                raise RuntimeError(f"Exceeded retries for case {case}")

            retries += 1
            with canonical_log.open("a", encoding="utf-8") as f:
                f.write(f"\n===== RETRY {retries} @ {time.strftime('%F %T')} =====\n")

            cmd = [
                "python",
                "-u",
                str(MMSegTrain),
                str(cfg),
                "--work-dir",
                str(work_dir),
                "--resume",
            ]
            rc = run_cmd(cmd, canonical_log, env=env, append=True)

            with guard_log.open("a", encoding="utf-8") as f:
                f.write(
                    f"[RUN] {time.strftime('%F %T')} case={case} rc={rc} "
                    f"max_iter={max_iter_in_ckpt(work_dir)}\n"
                )

            rebuild_master(guard_log)
            sync_case_logs_incremental(case, canonical_log, state_root)

            if max_iter_in_ckpt(work_dir) >= TARGET_ITERS:
                break

            time.sleep(15)

    rebuild_master(guard_log)
    run_table2(ROOT / args.out_root, guard_log)

    with guard_log.open("a", encoding="utf-8") as f:
        f.write(f"[DONE] {time.strftime('%F %T')} guard finished\n")


if __name__ == "__main__":
    main()
