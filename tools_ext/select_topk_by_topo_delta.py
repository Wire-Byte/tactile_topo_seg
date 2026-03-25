import argparse
from pathlib import Path
import pandas as pd
import shutil
import re

def detect_key_column(df: pd.DataFrame):
    # Prefer columns containing "PartXX/NNNN" or "PartXX/NNNN.png"
    for col in df.columns:
        s = df[col].astype(str)
        score = s.str.contains(r"Part\d{2}[/_]\d+", regex=True).mean()
        if score > 0.5:
            return col
    # Common fallbacks
    for col in ["part_id", "img_id", "id", "name", "filename", "file"]:
        if col in df.columns:
            return col
    return None

def find_col(df: pd.DataFrame, candidates):
    # exact
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive exact
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    # fuzzy contains
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None

def key_to_panel_name(key: str):
    key = str(key).strip().replace("\\", "/")
    key = re.sub(r"\.(png|jpg|jpeg)$", "", key, flags=re.IGNORECASE)
    if "/" in key:
        part, stem = key.split("/")[-2], key.split("/")[-1]
        return f"{part}_{stem}.jpg", part, stem
    if "_" in key and key.startswith("Part"):
        part, stem = key.split("_", 1)
        return f"{part}_{stem}.jpg", part, stem
    return None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample-csv", required=True)
    ap.add_argument("--triplet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-iou-drop", type=float, default=0.01,
                    help="if iou_delta exists, allow it to drop at most this (0.01=1%)")
    ap.add_argument("--copy", action="store_true", help="copy selected images into out-dir/images")
    ap.add_argument("--symlink", action="store_true", help="symlink selected images into out-dir/images (faster)")
    args = ap.parse_args()

    per_csv = Path(args.per_sample_csv)
    trip_dir = Path(args.triplet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_csv)

    key_col = detect_key_column(df)
    if key_col is None:
        raise RuntimeError(f"Cannot detect key column. columns={list(df.columns)}")

    cc_col = find_col(df, ["cc_absdiff_delta", "cc_delta", "delta_cc_absdiff"])
    holes_col = find_col(df, ["holes_absdiff_delta", "holes_delta", "delta_holes_absdiff"])
    cldice_col = find_col(df, ["clDice_delta", "cldice_delta", "delta_cldice"])
    iou_col = find_col(df, ["iou_delta", "delta_iou", "IoU_delta"])

    # Build composite score: structure improvements dominate
    score = pd.Series(0.0, index=df.index)
    used = []
    if cc_col is not None:
        score += (-df[cc_col].astype(float)).rank(pct=True); used.append(cc_col)
    if holes_col is not None:
        score += (-df[holes_col].astype(float)).rank(pct=True); used.append(holes_col)
    if cldice_col is not None:
        score += (df[cldice_col].astype(float)).rank(pct=True); used.append(cldice_col)

    df["__score"] = score

    # Optional constraint: don't pick samples where IoU drops too much
    if iou_col is not None:
        df = df[df[iou_col].astype(float) >= -float(args.max_iou_drop)].copy()

    # Keep only rows that have an existing triplet image
    panel_paths = []
    parts, stems = [], []
    for _, row in df.iterrows():
        panel_name, part, stem = key_to_panel_name(row[key_col])
        if panel_name is None:
            continue
        p = trip_dir / panel_name
        if p.exists():
            panel_paths.append(p)
            parts.append(part)
            stems.append(stem)
        else:
            panel_paths.append(None)
            parts.append(part)
            stems.append(stem)

    df["__part"] = parts
    df["__stem"] = stems
    df["__panel"] = [str(p) if p is not None else "" for p in panel_paths]
    df = df[df["__panel"] != ""].copy()

    # Select top-k by score (descending)
    df_sel = df.sort_values("__score", ascending=False).head(args.k).copy()

    # Export selection tables
    cols_out = [key_col, "__part", "__stem", "__score"]
    for c in [cc_col, holes_col, cldice_col, iou_col]:
        if c is not None and c not in cols_out:
            cols_out.append(c)
    cols_out.append("__panel")

    df_sel[cols_out].to_csv(out_dir / f"selected_top{len(df_sel)}.csv", index=False)

    with open(out_dir / f"selected_top{len(df_sel)}_paths.txt", "w") as f:
        for p in df_sel["__panel"].tolist():
            f.write(p + "\n")

    # Optionally copy/symlink images
    if args.copy or args.symlink:
        for p_str in df_sel["__panel"].tolist():
            p = Path(p_str)
            dst = out_dir / "images" / p.name
            if dst.exists():
                continue
            if args.symlink:
                dst.symlink_to(p.resolve())
            else:
                shutil.copy2(p, dst)

    print("[done] selected:", len(df_sel))
    print(" - csv:", out_dir / f"selected_top{len(df_sel)}.csv")
    print(" - paths:", out_dir / f"selected_top{len(df_sel)}_paths.txt")
    if args.copy or args.symlink:
        print(" - images:", out_dir / "images")
    print(" - key_col:", key_col)
    print(" - used metric cols:", used, "iou_col:", iou_col)

if __name__ == "__main__":
    main()
