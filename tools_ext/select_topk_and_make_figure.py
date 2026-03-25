import argparse
from pathlib import Path
import pandas as pd
from PIL import Image

def detect_key_column(df: pd.DataFrame):
    # Prefer columns that look like "Part02/0141" or "Part02/0141.png"
    for col in df.columns:
        s = df[col].astype(str)
        if s.str.contains(r"Part\d{2}[/_]\d+", regex=True).mean() > 0.5:
            return col
    # fallback common names
    for col in ["part_id", "img_id", "name", "id", "filename", "file"]:
        if col in df.columns:
            return col
    return None

def key_to_panel_name(key: str):
    # Accept "Part02/0141", "Part02/0141.png", "Part02_0141", "Part02_0141.jpg"
    key = str(key).strip()
    key = key.replace("\\", "/")
    key = key.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
    if "/" in key:
        part, stem = key.split("/")[-2], key.split("/")[-1]
    elif "_" in key and key.startswith("Part"):
        part, stem = key.split("_", 1)
    else:
        # if only stem (rare), return None and let caller handle
        return None
    return f"{part}_{stem}.jpg"

def find_metric_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy contains
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample-csv", required=True)
    ap.add_argument("--panels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-iou-drop", type=float, default=0.01,
                    help="allow iou_delta to drop at most this (e.g. 0.01=1%) if iou_delta exists")
    args = ap.parse_args()

    per_csv = Path(args.per_sample_csv)
    panels_dir = Path(args.panels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_csv)

    key_col = detect_key_column(df)
    if key_col is None:
        raise RuntimeError(f"Cannot detect key column from columns={list(df.columns)}")

    # metric columns (robust to different naming)
    cc_col = find_metric_col(df, ["cc_absdiff_delta", "cc_delta", "delta_cc_absdiff", "cc_absdiffΔ"])
    holes_col = find_metric_col(df, ["holes_absdiff_delta", "holes_delta", "delta_holes_absdiff", "holes_absdiffΔ"])
    cldice_col = find_metric_col(df, ["clDice_delta", "cldice_delta", "delta_clDice", "clDiceΔ"])
    iou_col = find_metric_col(df, ["iou_delta", "IoU_delta", "delta_iou"])

    # Build a score: larger is better
    # - We want cc_absdiff_delta NEGATIVE (reduction), same for holes.
    # - We want clDice_delta POSITIVE (increase).
    score = pd.Series(0.0, index=df.index)

    if cc_col is not None:
        score += (-df[cc_col].astype(float)).rank(pct=True)
    if holes_col is not None:
        score += (-df[holes_col].astype(float)).rank(pct=True)
    if cldice_col is not None:
        score += (df[cldice_col].astype(float)).rank(pct=True)

    # Optional constraint: avoid samples where IoU drops too much (if exists)
    if iou_col is not None:
        ok = df[iou_col].astype(float) >= -float(args.max_iou_drop)
        df = df[ok].copy()
        score = score.loc[df.index].copy()

    # Map each row to existing panel image
    panel_paths = []
    for idx, row in df.iterrows():
        panel_name = key_to_panel_name(row[key_col])
        if panel_name is None:
            continue
        p = panels_dir / panel_name
        if p.exists():
            panel_paths.append((idx, p, row[key_col]))
    if len(panel_paths) == 0:
        raise RuntimeError("No panel images matched. Check key format and panels naming.")

    # Keep only rows with panels
    keep_idx = [idx for idx, _, _ in panel_paths]
    df2 = df.loc[keep_idx].copy()
    score2 = score.loc[keep_idx].copy()
    df2["__score"] = score2

    # Select top-k
    df_sel = df2.sort_values("__score", ascending=False).head(args.k).copy()

    # Save selection table
    cols_out = [key_col]
    for c in [cc_col, holes_col, cldice_col, iou_col]:
        if c is not None and c not in cols_out:
            cols_out.append(c)
    cols_out.append("__score")
    df_sel[cols_out].to_csv(out_dir / f"selected_top{args.k}.csv", index=False)

    # Make montage (2 rows x (k/2) cols)
    k = len(df_sel)
    cols = (k + 1) // 2
    rows = 2 if k > 1 else 1

    # Load images
    imgs = []
    for _, row in df_sel.iterrows():
        panel_name = key_to_panel_name(row[key_col])
        p = panels_dir / panel_name
        imgs.append(Image.open(p).convert("RGB"))

    w, h = imgs[0].size
    gap = 12
    canvas_w = cols * w + (cols - 1) * gap
    canvas_h = rows * h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))

    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        x = c * (w + gap)
        y = r * (h + gap)
        canvas.paste(im, (x, y))

    fig_path = out_dir / f"Figure_qualitative_top{k}.jpg"
    canvas.save(fig_path, quality=92)

    print("[done] selected:", k)
    print(" - selection csv:", out_dir / f"selected_top{args.k}.csv")
    print(" - figure:", fig_path)
    print(" - key_col:", key_col)
    print(" - cc_col:", cc_col, "holes_col:", holes_col, "cldice_col:", cldice_col, "iou_col:", iou_col)

if __name__ == "__main__":
    main()
