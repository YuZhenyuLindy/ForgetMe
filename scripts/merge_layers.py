# scripts/merge_layers.py
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageFilter

def load_img(p: Path): return Image.open(p).convert("RGBA")

def main():
    ap = argparse.ArgumentParser("Layer merging (Side Story)")
    ap.add_argument("--bg",   type=Path, required=True, help="Background RGBA/RGB")
    ap.add_argument("--fg",   type=Path, required=True, help="Foreground RGBA/RGB (aligned)")
    ap.add_argument("--pos",  type=Path, required=True, help="Position mask (L, 1=place fg)")
    ap.add_argument("--out",  type=Path, required=True, help="Output path")
    ap.add_argument("--feather", type=int, default=5,    help="Feather radius (px)")
    args = ap.parse_args()

    bg = load_img(args.bg).convert("RGBA")
    fg = load_img(args.fg).convert("RGBA")
    pos = Image.open(args.pos).convert("L")

    # Feather mask to avoid seams
    if args.feather > 0:
        pos = pos.filter(ImageFilter.GaussianBlur(radius=args.feather))

    # Scale fg to bg if needed
    if fg.size != bg.size:
        fg = fg.resize(bg.size, Image.LANCZOS)
    if pos.size != bg.size:
        pos = pos.resize(bg.size, Image.NEAREST)

    # Normalize mask into alpha matte
    alpha = np.asarray(pos).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0, 1)
    alpha = np.stack([alpha]*4, axis=-1)  # RGBA alpha

    bg_np = np.asarray(bg).astype(np.float32)/255.0
    fg_np = np.asarray(fg).astype(np.float32)/255.0

    out_np = alpha * fg_np + (1.0 - alpha) * bg_np
    out_np = (np.clip(out_np, 0, 1) * 255.0).astype(np.uint8)

    Image.fromarray(out_np, mode="RGBA").save(args.out)
    print(f"[OK] Saved: {args.out.resolve()}")

if __name__ == "__main__":
    main()
