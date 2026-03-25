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

def main():
    cfg = Config.fromfile("configs/tp_dataset/segformer_b2_tp_skel_morph.py")
    setup(cfg)
    ds = DATASETS.build(cfg.train_dataloader.dataset)

    dh = cfg.model.decode_head
    tube_mode = dh.get("tube_mode", "quantile")
    tube_ratio = float(dh.get("tube_ratio", 0.2))
    tau_fixed = float(dh.get("tau_fixed", 32.0))

    # scan settings
    limit = int(dh.get("debug_scan_limit", 800))  # you can add this field in config
    limit = min(limit, len(ds))

    cover_in_fg_list = []
    fg_frac_list = []

    for i in range(limit):
        d = ds[i]["data_samples"]
        gt = d.gt_sem_seg.data.squeeze().cpu().numpy()
        fg = (gt > 0)
        if fg.sum() == 0:
            continue
        edge = d.gt_edge_dist.data.squeeze().cpu().numpy().astype(np.float32)
        vals = edge[fg]

        if tube_mode == "quantile":
            thr = np.quantile(vals, 1.0 - tube_ratio)
            tube = fg & (edge >= thr)
        else:
            tube = fg & (edge >= tau_fixed)

        cover_in_fg = tube[fg].mean()
        cover_in_fg_list.append(float(cover_in_fg))
        fg_frac_list.append(float(fg.mean()))

    arr = np.array(cover_in_fg_list, dtype=np.float32)
    fgarr = np.array(fg_frac_list, dtype=np.float32)

    print("tube_mode =", tube_mode, "tube_ratio =", tube_ratio, "tau_fixed =", tau_fixed)
    print("scanned:", len(arr))
    if len(arr) == 0:
        print("No FG samples found.")
        return

    qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    print("cover_in_fg mean/std:", float(arr.mean()), float(arr.std()))
    print("cover_in_fg q05/25/50/75/95:", qs)
    print("fg_frac mean/std:", float(fgarr.mean()), float(fgarr.std()))

if __name__ == "__main__":
    main()

