import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmseg.registry import DATASETS
import mmseg_ext  # noqa

def setup(cfg):
    init_default_scope("mmseg")
    ci = cfg.get("custom_imports", None)
    if ci:
        import_modules_from_strings(
            ci.get("imports", []),
            allow_failed_imports=ci.get("allow_failed_imports", False)
        )

def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def main():
    cfg = Config.fromfile("configs/tp_dataset/segformer_b2_tp_skel_morph.py")
    setup(cfg)
    ds = DATASETS.build(cfg.train_dataloader.dataset)

    dh = cfg.model.decode_head
    tube_mode = dh.get("tube_mode", "quantile")
    tube_ratio = float(dh.get("tube_ratio", 0.2))
    tau_fixed  = float(dh.get("tau_fixed", 32.0))
    print("tube_mode =", tube_mode, "tube_ratio =", tube_ratio, "tau_fixed =", tau_fixed)

    pick = [0, 1, 2, 3, 10, 50, 100, 200]
    for i in pick:
        item = ds[i]
        d = item["data_samples"]
        pid = d.metainfo.get("part_id", f"idx{i}")

        gt = to_np(d.gt_sem_seg.data).squeeze().astype(np.uint8)
        fg = (gt > 0)
        if fg.sum() == 0:
            print(f"\n=== {pid} === fg=0 (skip)")
            continue

        edge = to_np(d.gt_edge_dist.data).squeeze().astype(np.float32)
        vals = edge[fg]
        q = np.quantile(vals, [0.1, 0.2, 0.5, 0.8, 0.9])

        if tube_mode == "quantile":
            thr = np.quantile(vals, 1.0 - tube_ratio)
            tube = fg & (edge >= thr)
        else:
            tube = fg & (edge >= tau_fixed)

        cover_all = tube.mean()
        cover_in_fg = tube[fg].mean()

        print(f"\n=== {pid} === fg_pix={int(fg.sum())}")
        print("edge_q10/20/50/80/90:", q)
        print(f"tube cover(all)={cover_all:.4f}  cover_in_fg={cover_in_fg:.4f}")
        print("edge min/max/mean on fg:", float(vals.min()), float(vals.max()), float(vals.mean()))

if __name__ == "__main__":
    main()