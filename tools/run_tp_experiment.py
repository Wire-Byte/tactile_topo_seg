#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified launcher for TP baseline / +skel experiments.

Supports:
- training
- evaluation (MMSeg IoUMetric: mIoU/mDice/mFscore)
- optional topology-style evaluation CSV generation
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
MMSEG_TRAIN = ROOT / "third_party/mmsegmentation/tools/train.py"
MMSEG_TEST = ROOT / "third_party/mmsegmentation/tools/test.py"
TOPO_EVAL = ROOT / "tools/eval_tp_topology.py"

CONFIGS: Dict[str, Path] = {
    "baseline": ROOT / "configs/tp_dataset/segformer_b2_tp.py",
    "skel": ROOT / "configs/tp_dataset/segformer_b2_tp_skel.py",
}

WORK_DIRS: Dict[str, Path] = {
    "baseline": ROOT / "work_dirs/segformer_b2_tp",
    "skel": ROOT / "work_dirs/segformer_b2_tp_skel",
}


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[RUN] {printable}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    required = f"{ROOT}:{ROOT / 'third_party/mmsegmentation'}"
    env["PYTHONPATH"] = required if not py_path else f"{required}:{py_path}"
    return env


def _resolve_checkpoint(variant: str, ckpt: Optional[str]) -> Path:
    if ckpt:
        p = Path(ckpt)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    work_dir = WORK_DIRS[variant]

    last_ckpt_ptr = work_dir / "last_checkpoint"
    if last_ckpt_ptr.exists():
        content = last_ckpt_ptr.read_text(encoding="utf-8").strip()
        if content:
            p = Path(content)
            if not p.is_absolute():
                p = work_dir / content
            if p.exists():
                return p

    candidates = sorted(work_dir.rglob("best*.pth"))
    if candidates:
        return candidates[-1]

    candidates = sorted(work_dir.rglob("iter_*.pth"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found for variant={variant}. Pass --checkpoint explicitly."
    )


def _distributed_prefix(gpus: int) -> List[str]:
    if gpus <= 1:
        return [sys.executable]
    return [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpus}"]


def cmd_train(args: argparse.Namespace) -> None:
    cfg = CONFIGS[args.variant]
    work_dir = Path(args.work_dir) if args.work_dir else WORK_DIRS[args.variant]
    env = _build_env()

    cmd: List[str] = _distributed_prefix(args.gpus)
    cmd += [str(MMSEG_TRAIN), str(cfg), "--work-dir", str(work_dir), "--launcher", args.launcher]

    if args.resume:
        cmd.append("--resume")
    if args.amp:
        cmd.append("--amp")
    if args.cfg_options:
        cmd += ["--cfg-options", *args.cfg_options]

    _run(cmd, env)


def _build_eval_out_dir(variant: str, checkpoint: Path, out_dir: Optional[str]) -> Path:
    if out_dir:
        out = Path(out_dir)
        return out if out.is_absolute() else ROOT / out
    stem = checkpoint.stem
    return WORK_DIRS[variant] / f"preds_val_{stem}"


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = CONFIGS[args.variant]
    work_dir = Path(args.work_dir) if args.work_dir else WORK_DIRS[args.variant]
    checkpoint = _resolve_checkpoint(args.variant, args.checkpoint)
    out_dir = _build_eval_out_dir(args.variant, checkpoint, args.out_dir)
    env = _build_env()

    cmd: List[str] = _distributed_prefix(args.gpus)
    cmd += [
        str(MMSEG_TEST),
        str(cfg),
        str(checkpoint),
        "--work-dir",
        str(work_dir),
        "--out",
        str(out_dir),
        "--launcher",
        args.launcher,
    ]
    if args.tta:
        cmd.append("--tta")
    if args.cfg_options:
        cmd += ["--cfg-options", *args.cfg_options]

    _run(cmd, env)

    print("[INFO] MMSeg evaluator metrics are from config val/test evaluator: mIoU, mDice, mFscore")
    print(f"[INFO] prediction masks saved to: {out_dir}")

    if args.with_topology:
        topo_csv = args.topo_csv or str(work_dir / f"topo_{args.variant}_{checkpoint.stem}.csv")
        topo_summary_csv = args.topo_summary_csv or str(Path(topo_csv).with_name(Path(topo_csv).stem + "_summary.csv"))

        topo_cmd = [
            sys.executable,
            str(TOPO_EVAL),
            "--pred-dir",
            str(out_dir),
            "--split-file",
            args.split_file,
            "--gt-root",
            args.gt_root,
            "--out-csv",
            topo_csv,
            "--summary-csv",
            topo_summary_csv,
        ]
        _run(topo_cmd, env)


def cmd_train_eval(args: argparse.Namespace) -> None:
    train_args = argparse.Namespace(**vars(args))
    cmd_train(train_args)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.checkpoint = args.eval_checkpoint
    cmd_eval(eval_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TP baseline/+skel train and eval.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--variant", choices=["baseline", "skel"], required=True, help="Experiment variant.")
    common.add_argument("--gpus", type=int, default=1, help="Number of GPUs. >1 uses torch.distributed.run.")
    common.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    common.add_argument("--work-dir", default=None, help="Override work dir.")
    common.add_argument("--cfg-options", nargs="+", default=None, help="Pass-through cfg-options for MMSeg tools.")

    p_train = sub.add_parser("train", parents=[common], help="Train baseline or skel model.")
    p_train.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in work_dir.")
    p_train.add_argument("--amp", action="store_true", help="Enable amp in train.py.")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", parents=[common], help="Evaluate checkpoint and dump prediction masks.")
    p_eval.add_argument("--checkpoint", default=None, help="Checkpoint path. If omitted, auto-resolve from work_dir.")
    p_eval.add_argument("--out-dir", default=None, help="Prediction output directory for --out.")
    p_eval.add_argument("--tta", action="store_true", help="Enable test-time augmentation.")
    p_eval.add_argument("--with-topology", action="store_true", help="Also compute topology-style metrics csv.")
    p_eval.add_argument("--split-file", default="data/TP-Dataset/Index/val.txt", help="Split file for topology eval.")
    p_eval.add_argument("--gt-root", default="data/TP-Dataset/GroundTruth", help="GT root for topology eval.")
    p_eval.add_argument("--topo-csv", default=None, help="Output path of per-sample topology csv.")
    p_eval.add_argument("--topo-summary-csv", default=None, help="Output path of topology summary csv.")
    p_eval.set_defaults(func=cmd_eval)

    p_train_eval = sub.add_parser("train-eval", parents=[common], help="Train then evaluate.")
    p_train_eval.add_argument("--resume", action="store_true", help="Resume training.")
    p_train_eval.add_argument("--amp", action="store_true", help="Enable amp in training.")
    p_train_eval.add_argument("--eval-checkpoint", default=None, help="Checkpoint to evaluate after training. Auto if omitted.")
    p_train_eval.add_argument("--out-dir", default=None, help="Prediction output directory for eval.")
    p_train_eval.add_argument("--tta", action="store_true", help="Enable TTA for eval.")
    p_train_eval.add_argument("--with-topology", action="store_true", help="Also compute topology-style metrics csv.")
    p_train_eval.add_argument("--split-file", default="data/TP-Dataset/Index/val.txt", help="Split file for topology eval.")
    p_train_eval.add_argument("--gt-root", default="data/TP-Dataset/GroundTruth", help="GT root for topology eval.")
    p_train_eval.add_argument("--topo-csv", default=None, help="Output path of per-sample topology csv.")
    p_train_eval.add_argument("--topo-summary-csv", default=None, help="Output path of topology summary csv.")
    p_train_eval.set_defaults(func=cmd_train_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
