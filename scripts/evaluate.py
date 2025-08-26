# scripts/evaluate.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import csv
from tools.entangled import entangled_from_files

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def _index_dir(d: Path):
    d = d.resolve()
    m = {}
    for p in d.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            m[p.stem] = p
    return m

def main():
    ap = argparse.ArgumentParser("Entangled metric evaluator")
    ap.add_argument("--orig_dir", type=Path, default=None, help="Original images dir (paired mode)")
    ap.add_argument("--bg_dir",   type=Path, required=True,   help="Background images dir (required)")
    ap.add_argument("--mask_dir", type=Path, required=True,   help="Binary masks dir (1=inner/forget region)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta",  type=float, default=0.5)
    ap.add_argument("--out_csv", type=Path, default=Path("entangled_results.csv"))
    args = ap.parse_args()

    bg_map   = _index_dir(args.bg_dir)
    mask_map = _index_dir(args.mask_dir)
    orig_map = _index_dir(args.orig_dir) if args.orig_dir else None

    keys = sorted(set(bg_map.keys()) & set(mask_map.keys()))
    if args.orig_dir:
        keys = [k for k in keys if k in orig_map]

    if not keys:
        raise SystemExit("No matched filenames across dirs.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        if args.orig_dir:
            wr.writerow(["key", "mode", "alpha", "beta", "entangled"])
        else:
            wr.writerow(["key", "mode", "entangled"])

        for k in keys:
            p_bg   = bg_map[k]
            p_mask = mask_map[k]
            if args.orig_dir:
                p_orig = orig_map[k]
                val = entangled_from_files(p_bg, p_mask, p_orig, alpha=args.alpha, beta=args.beta)
                wr.writerow([k, "paired", args.alpha, args.beta, f"{val:.6f}"])
            else:
                val = entangled_from_files(p_bg, p_mask, None)
                wr.writerow([k, "single", f"{val:.6f}"])

    print(f"[OK] Wrote: {args.out_csv.resolve()}  | samples={len(keys)}")

if __name__ == "__main__":
    main()
