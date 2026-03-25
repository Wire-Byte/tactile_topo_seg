import numpy as np
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

def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def main():
    cfg = Config.fromfile("configs/tp_dataset/segformer_b2_tp_skel_morph.py")
    setup(cfg)

    # morph maps are generated in train pipeline
    ds = DATASETS.build(cfg.train_dataloader.dataset)

    tau = float(cfg.model.decode_head.get("tau_fixed", 8.0))
    pick = [0, 1, 2, 3, 10, 50, 100, 200]

    print("tau_fixed =", tau)
    for i in pick:
        item = ds[i]
        d = item["data_samples"]

        gt = to_numpy(d.gt_sem_seg.data).squeeze().astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)
        fg = (gt == 1)

        pid = d.metainfo.get("part_id", f"idx{i}")

        if not hasattr(d, "gt_edge_dist"):
            raise RuntimeError("Missing gt_edge_dist. Check GenerateTPMorphMaps + Pack.")

        edge = to_numpy(d.gt_edge_dist.data).squeeze().astype(np.float32)

        if fg.sum() == 0:
            print(f"\n=== {pid} === fg=0 (skip)")
            continue

        vals = edge[fg]
        q = np.quantile(vals, [0.1, 0.2, 0.5, 0.8, 0.9])
        tube = ((edge >= tau) & fg).astype(np.uint8)

        cover = tube.mean()
        cover_in_fg = tube[fg].mean()
        print(f"\n=== {pid} === fg_pix={int(fg.sum())}")
        print("edge_q10/20/50/80/90:", q)
        print(f"tube(edge>=tau) cover(all)={cover:.4f}  cover_in_fg={cover_in_fg:.4f}")

        # quick sanity on edge semantics: is center larger than boundary?
        # if edge is distance-to-boundary inside, it should have many zeros near boundary and larger values inside
        print("edge min/max/mean on fg:", float(vals.min()), float(vals.max()), float(vals.mean()))

if __name__ == "__main__":
    main()

