import os, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def read_mask(path: Path):
    arr = np.array(Image.open(path).convert("L"))
    return ((arr >= 128).astype(np.uint8) * 255)

def overlay(rgb_pil, mask255, color, alpha=0.45):
    rgb = np.array(rgb_pil).astype(np.float32)
    m = (mask255 > 0).astype(np.float32)[..., None]
    col = np.array(color, dtype=np.float32)[None, None, :]
    out = rgb * (1 - m * alpha) + col * (m * alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def add_title(im, title, h=34):
    w, hh = im.size
    canvas = Image.new("RGB", (w, hh + h), (0, 0, 0))
    canvas.paste(im, (0, h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(255, 255, 255), font=font)
    return canvas

def hstack(ims, gap=10, bg=(25, 25, 25)):
    H = max(im.size[1] for im in ims)
    W = sum(im.size[0] for im in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.size[0] + gap
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-file", required=True)
    ap.add_argument("--img-root", required=True)  # .../JPEGImages
    ap.add_argument("--gt-root", required=True)   # .../GroundTruth
    ap.add_argument("--pred-base", required=True)
    ap.add_argument("--pred-skel", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-num", type=int, default=120)
    ap.add_argument("--resize", type=int, default=640)
    ap.add_argument("--only", default="", help="comma-separated keys like Part02/0141,Part03/0545")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = None
    if args.only.strip():
        only = set([x.strip() for x in args.only.split(",") if x.strip()])

    with open(args.index_file, "r") as f:
        keys = [ln.strip() for ln in f if ln.strip()]

    missing = []
    saved = 0

    for key in keys:
        if only is not None and key not in only:
            continue

        part, stem = key.split("/")  # key has no suffix
        # image
        img_path = Path(args.img_root) / part / f"{stem}.jpg"
        if not img_path.exists():
            # fallback extensions
            for ext in [".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                p2 = Path(args.img_root) / part / f"{stem}{ext}"
                if p2.exists():
                    img_path = p2
                    break

        gt_path = Path(args.gt_root) / part / f"{stem}.png"
        pb = Path(args.pred_base) / f"{stem}.png"
        ps = Path(args.pred_skel) / f"{stem}.png"

        if not (img_path.exists() and gt_path.exists() and pb.exists() and ps.exists()):
            missing.append((key, int(img_path.exists()), int(gt_path.exists()), int(pb.exists()), int(ps.exists())))
            continue

        rgb = Image.open(img_path).convert("RGB")
        gt = read_mask(gt_path)
        mb = read_mask(pb)
        ms = read_mask(ps)

        if args.resize > 0:
            rgb = rgb.resize((args.resize, args.resize), Image.BILINEAR)
            gt = np.array(Image.fromarray(gt).resize((args.resize, args.resize), Image.NEAREST))
            mb = np.array(Image.fromarray(mb).resize((args.resize, args.resize), Image.NEAREST))
            ms = np.array(Image.fromarray(ms).resize((args.resize, args.resize), Image.NEAREST))

        p1 = add_title(rgb, f"RGB  {part}/{stem}")
        p2 = add_title(overlay(rgb, gt, (0, 255, 0)), "GT (green)")
        p3 = add_title(overlay(rgb, mb, (255, 0, 0)), "Baseline (red)")
        p4 = add_title(overlay(rgb, ms, (0, 128, 255)), "Skeleton (blue)")

        panel = hstack([p1, p2, p3, p4])
        panel.save(out_dir / f"{part}_{stem}.jpg", quality=92)
        saved += 1
        if saved >= args.max_num:
            break

    rep = out_dir / "missing_report.txt"
    with open(rep, "w") as f:
        f.write("key\timg\tgt\tbase\tskel\n")
        for k, a, b, c, d in missing:
            f.write(f"{k}\t{a}\t{b}\t{c}\t{d}\n")

    print(f"[done] saved={saved}, missing={len(missing)}")
    print(f"[report] {rep}")

if __name__ == "__main__":
    main()
