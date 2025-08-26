# scripts/create_dataset.py
# -*- coding: utf-8 -*-
"""
Automatic dataset creation for ForgetMe:
- Candidate mask generation: SAM (if provided) or simple CV-based proposals
- CLIP scoring: pick the mask that best matches the text prompt (optional)
- Background reconstruction: OpenCV inpaint (default) or leave hooks for LaMa/SD

Example:
python scripts/create_dataset.py \
  --input_dir ./raw_images \
  --prompt "<cat>" \
  --out_dir ./data/forgetme_cat \
  --use_clip \
  --sam_checkpoint /path/to/sam_vit_h.pth --sam_model vit_h

Minimal (no SAM/CLIP):
python scripts/create_dataset.py --input_dir ./raw_images --prompt "<dog>" --out_dir ./data/forgetme_dog
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Optional
try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False

# Optional SAM
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)  # H,W,3 uint8


def save_image(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def ensure_binary_mask(mask: np.ndarray, thresh: int = 128) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
    return m


def simple_candidate_masks(img: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """
    Pure OpenCV proposals as a fallback:
    - Edge map + contour filtering
    - Return top-k largest contour masks
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:k]

    masks = []
    h, w = gray.shape
    for c in contours:
        m = np.zeros((h,w), dtype=np.uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=-1)
        masks.append(m)
    # Always add center rectangle as a backup
    cx0, cy0 = int(w*0.2), int(h*0.2)
    cx1, cy1 = int(w*0.8), int(h*0.8)
    rect = np.zeros((h,w), dtype=np.uint8)
    rect[cy0:cy1, cx0:cx1] = 255
    masks.append(rect)
    return masks


def sam_candidate_masks(img: np.ndarray, sam_ckpt: Path, model_type: str = "vit_h", points_per_side: int = 32) -> list[np.ndarray]:
    assert _HAS_SAM, "segment-anything not installed."
    sam = sam_model_registry[model_type](checkpoint=str(sam_ckpt))
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=256
    )
    masks = generator.generate(img)
    # Convert to list of binary masks
    out = []
    for m in masks:
        # m["segmentation"] is bool mask
        out.append((m["segmentation"].astype(np.uint8))*255)
    # limit
    out = sorted(out, key=lambda x: x.sum(), reverse=True)[:20]
    return out


def clip_score(img: np.ndarray, prompt: str, model_ctx=None) -> float:
    """
    Compute cosine similarity between image embedding and text embedding.
    Requires open-clip. If not available, return 0.5 as neutral score.
    """
    if not _HAS_OPENCLIP:
        return 0.5
    model, preprocess, tokenizer, device = model_ctx
    pil = Image.fromarray(img)
    with torch.no_grad():
        img_in = preprocess(pil).unsqueeze(0).to(device)
        text_in = tokenizer([prompt]).to(device)
        img_feat = model.encode_image(img_in)
        txt_feat = model.encode_text(text_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze().item()
    return float(sim)


def best_mask_by_clip(img: np.ndarray, masks: list[np.ndarray], prompt: str, model_ctx=None) -> np.ndarray:
    best, best_score = None, -1e9
    for m in masks:
        bbox = cv2.boundingRect(m)
        x, y, w, h = bbox
        if w < 8 or h < 8:
            continue
        crop = cv2.bitwise_and(img, img, mask=m)
        crop = crop[y:y+h, x:x+w]
        s = clip_score(crop, prompt, model_ctx=model_ctx)
        if s > best_score:
            best_score = s
            best = m
    if best is None:
        best = max(masks, key=lambda x: x.sum())
    return best


def inpaint_opencv(img: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3) -> np.ndarray:
    """
    Inpaint masked region to get background.
    """
    mask_bin = ensure_binary_mask(mask)
    if method == "ns":
        res = cv2.inpaint(img, (mask_bin>0).astype(np.uint8), radius, cv2.INPAINT_NS)
    else:
        res = cv2.inpaint(img, (mask_bin>0).astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--prompt", type=str, required=True, help="Target concept text, e.g., '<cat>'")
    ap.add_argument("--out_dir", type=Path, required=True)

    # Options
    ap.add_argument("--use_clip", action="store_true", help="Use CLIP scoring to select best mask")
    ap.add_argument("--sam_checkpoint", type=Path, default=None, help="Use SAM for candidate masks")
    ap.add_argument("--sam_model", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--opencv_inpaint", type=str, default="telea", choices=["telea", "ns"])
    ap.add_argument("--feather", type=int, default=3, help="Edge feather for mask refining")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_orig = args.out_dir / "original"
    out_fg   = args.out_dir / "foreground"
    out_bg   = args.out_dir / "background"
    out_mask = args.out_dir / "mask"

    out_orig.mkdir(exist_ok=True)
    out_fg.mkdir(exist_ok=True)
    out_bg.mkdir(exist_ok=True)
    out_mask.mkdir(exist_ok=True)

    # Init CLIP if needed
    model_ctx = None
    if args.use_clip:
        if not _HAS_OPENCLIP:
            print("[WARN] open-clip-torch not installed; --use_clip will be ignored.")
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model = model.eval().to(device)
            model_ctx = (model, preprocess, tokenizer, device)

    # Collect images
    paths = [p for p in args.input_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise SystemExit(f"No images in {args.input_dir}")

    for p in tqdm(paths, desc="Creating ForgetMe samples"):
        img = load_image(p)

        # 1) candidate masks
        if args.sam_checkpoint is not None:
            if not _HAS_SAM:
                print("[WARN] SAM not installed, fallback to simple proposals.")
                masks = simple_candidate_masks(img)
            else:
                masks = sam_candidate_masks(img, args.sam_checkpoint, args.sam_model)
        else:
            masks = simple_candidate_masks(img)

        # 2) choose best mask by CLIP (optional)
        if args.use_clip and _HAS_OPENCLIP:
            m = best_mask_by_clip(img, masks, args.prompt, model_ctx=model_ctx)
        else:
            m = max(masks, key=lambda x: x.sum())

        # Feather edges
        if args.feather > 0:
            k = args.feather * 2 + 1
            blur = cv2.GaussianBlur(m, (k, k), 0)
            _, m = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)

        # 3) build foreground crop (keep only masked region; rest transparent)
        fg = img.copy()
        inv = cv2.bitwise_not(m)
        fg_bg = np.zeros_like(img)
        fg = cv2.bitwise_and(img, img, mask=m)

        # 4) background reconstruction
        bg = inpaint_opencv(img, m, method=args.opencv_inpaint, radius=3)

        # Save four-pack
        stem = p.stem
        save_image(img, out_orig / f"{stem}.png")
        save_image(fg,  out_fg   / f"{stem}.png")
        save_image(bg,  out_bg   / f"{stem}.png")
        Image.fromarray(m).save(out_mask / f"{stem}.png")

    print(f"[OK] Saved to: {args.out_dir}")
    print("Subfolders: original / foreground / background / mask")


if __name__ == "__main__":
    main()
