import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image

def read_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)

def resize_nearest(mask: np.ndarray, target_hw) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8))
    im = im.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
    return np.array(im).astype(np.uint8)

def binarize(mask: np.ndarray, fg_id: int = 1) -> np.ndarray:
    return (mask == fg_id).astype(np.uint8)

def connected_components(bin_mask: np.ndarray):
    try:
        from scipy import ndimage as ndi
        structure = np.ones((3, 3), dtype=np.uint8)  # 8-connect
        lab, n = ndi.label(bin_mask.astype(bool), structure=structure)
        return int(n), lab.astype(np.int32)
    except Exception:
        # BFS fallback
        H, W = bin_mask.shape
        visited = np.zeros((H, W), dtype=bool)
        lab = np.zeros((H, W), dtype=np.int32)
        comp = 0
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for y in range(H):
            for x in range(W):
                if bin_mask[y, x] and not visited[y, x]:
                    comp += 1
                    stack = [(y, x)]
                    visited[y, x] = True
                    lab[y, x] = comp
                    while stack:
                        cy, cx = stack.pop()
                        for dy, dx in dirs:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and bin_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                lab[ny, nx] = comp
                                stack.append((ny, nx))
        return comp, lab

def largest_component_ratio(bin_mask: np.ndarray, labels: np.ndarray, num: int) -> float:
    fg = int(bin_mask.sum())
    if fg == 0 or num == 0:
        return 0.0
    sizes = np.bincount(labels.reshape(-1))
    if sizes.shape[0] <= 1:
        return 0.0
    largest = int(sizes[1:].max())
    return float(largest) / float(fg)

def hole_count(bin_mask: np.ndarray) -> int:
    try:
        from scipy import ndimage as ndi
        filled = ndi.binary_fill_holes(bin_mask.astype(bool))
        holes = (filled.astype(np.uint8) - bin_mask).astype(np.uint8)
        n, _ = connected_components(holes)
        return int(n)
    except Exception:
        return 0

def skeletonize(bin_mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        sk = sk_skeletonize(bin_mask.astype(bool))
        return sk.astype(np.uint8)
    except Exception:
        # last resort: return original mask (clDice becomes closer to Dice)
        return bin_mask

def cldice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    sp = skeletonize(pred_bin)
    sg = skeletonize(gt_bin)

    sp_sum = int(sp.sum())
    sg_sum = int(sg.sum())

    tprec = float((sp & gt_bin).sum()) / float(sp_sum) if sp_sum > 0 else 0.0
    tsens = float((sg & pred_bin).sum()) / float(sg_sum) if sg_sum > 0 else 0.0

    if (tprec + tsens) == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))

def dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = int((pred_bin & gt_bin).sum())
    s = int(pred_bin.sum()) + int(gt_bin.sum())
    if s == 0:
        return 1.0
    return float(2 * inter) / float(s)

def iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = int((pred_bin & gt_bin).sum())
    uni = int((pred_bin | gt_bin).sum())
    if uni == 0:
        return 1.0
    return float(inter) / float(uni)

_num_re = re.compile(r"(\d+)")

def numeric_key(p: Path):
    m = _num_re.search(p.stem)
    return int(m.group(1)) if m else 10**18

def load_index_lines(index_file: str):
    lines = []
    with open(index_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            lines.append(ln)  # e.g., Part08/1225
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--index-file", required=True, help="data/TP-Dataset/Index/predict.txt")
    ap.add_argument("--gt-root", default="data/TP-Dataset/SegmentationClassPNG",
                    help="Root for GT masks, containing PartXX folders.")
    ap.add_argument("--fg-id", type=int, default=1)
    ap.add_argument("--out-csv", default="topology_metrics.csv")
    ap.add_argument("--pred-ext", default=".png")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_root = Path(args.gt_root)

    idx = load_index_lines(args.index_file)

    pred_files = sorted(pred_dir.glob(f"*{args.pred_ext}"), key=numeric_key)

    if args.limit > 0:
        idx = idx[:args.limit]
        pred_files = pred_files[:args.limit]

    print(f"[INFO] index lines: {len(idx)}")
    print(f"[INFO] pred files : {len(pred_files)} (sorted by numeric stem)")
    if len(idx) != len(pred_files):
        print("[WARN] index count != pred count. Will evaluate up to min(len).")
    n = min(len(idx), len(pred_files))
    if n == 0:
        raise RuntimeError("No samples to evaluate (check pred-dir/index-file).")

    rows = []
    agg = {k: [] for k in ["dice","iou","cldice","cc_pred","cc_gt","lcr_pred","lcr_gt","holes_pred","holes_gt",
                           "fgpix_pred","fgpix_gt"]}

    missing_gt = 0

    for i in range(n):
        part_id = idx[i]  # Part08/1225
        part, name = part_id.split("/")
        gt_path = gt_root / part / f"{name}.png"
        pred_path = pred_files[i]

        if not gt_path.exists():
            missing_gt += 1
            continue

        pred = read_mask(str(pred_path))
        gt = read_mask(str(gt_path))

        if pred.shape != gt.shape:
            pred = resize_nearest(pred, gt.shape)

        # GT: 前景用 fg_id（TP-Dataset GroundTruth 前景是 255）
        gt_b = (gt == args.fg_id).astype(np.uint8)

        # Pred: 只要非0就当作前景（避免 0/1、0/255、或其它编码差异）
        pred_b = (pred > 0).astype(np.uint8)


        cc_p, lab_p = connected_components(pred_b)
        cc_g, lab_g = connected_components(gt_b)
        lcr_p = largest_component_ratio(pred_b, lab_p, cc_p)
        lcr_g = largest_component_ratio(gt_b, lab_g, cc_g)
        holes_p = hole_count(pred_b)
        holes_g = hole_count(gt_b)

        d = dice(pred_b, gt_b)
        j = iou(pred_b, gt_b)
        cd = cldice(pred_b, gt_b)

        fgp = int(pred_b.sum())
        fgg = int(gt_b.sum())

        rows.append((i, part_id, pred_path.name, str(gt_path), d, j, cd, cc_p, cc_g, lcr_p, lcr_g, holes_p, holes_g, fgp, fgg))

        agg["dice"].append(d); agg["iou"].append(j); agg["cldice"].append(cd)
        agg["cc_pred"].append(cc_p); agg["cc_gt"].append(cc_g)
        agg["lcr_pred"].append(lcr_p); agg["lcr_gt"].append(lcr_g)
        agg["holes_pred"].append(holes_p); agg["holes_gt"].append(holes_g)
        agg["fgpix_pred"].append(fgp); agg["fgpix_gt"].append(fgg)

    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","part_id","pred_file","gt_file",
                    "dice","iou","clDice",
                    "cc_pred","cc_gt","lcr_pred","lcr_gt",
                    "holes_pred","holes_gt",
                    "fgpix_pred","fgpix_gt"])
        w.writerows(rows)

    def mean(x): return float(np.mean(x)) if len(x) else 0.0
    def std(x): return float(np.std(x)) if len(x) else 0.0

    print(f"[OK] evaluated {len(rows)} samples (missing_gt={missing_gt})")
    print(f"Dice:   {mean(agg['dice']):.4f} ± {std(agg['dice']):.4f}")
    print(f"IoU:    {mean(agg['iou']):.4f} ± {std(agg['iou']):.4f}")
    print(f"clDice: {mean(agg['cldice']):.4f} ± {std(agg['cldice']):.4f}")
    print(f"CC(pred):  {mean(agg['cc_pred']):.2f} ± {std(agg['cc_pred']):.2f}")
    print(f"LCR(pred): {mean(agg['lcr_pred']):.4f} ± {std(agg['lcr_pred']):.4f}")
    print(f"Holes(pred): {mean(agg['holes_pred']):.2f} ± {std(agg['holes_pred']):.2f}")
    print(f"FGpix pred/gt: {mean(agg['fgpix_pred']):.1f} / {mean(agg['fgpix_gt']):.1f}")
    print(f"[CSV] saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
