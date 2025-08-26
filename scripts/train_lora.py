# scripts/train_lora.py
# -*- coding: utf-8 -*-
"""
Train LoRA on Stable Diffusion v1.5 (UNet cross-attn) using background images + a constant prompt.
This is a minimal, single-GPU friendly reference.

Example:
python scripts/train_lora.py \
  --data_dir ./data/forgetme_cat/bg \
  --prompt "<cat>" \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --output_dir ./lora/cat_unlearn \
  --train_steps 3000 --batch_size 4 --lr 1e-4 --rank 4 --lora_alpha 1.0
"""

import os
import math
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def add_lora_to_unet(unet: UNet2DConditionModel, rank: int = 4, lora_alpha: float = 1.0):
    """
    Attach LoRA processors to all attention blocks in UNet.
    """
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        # Cross-attn dim: None for self-attn ("attn1"), otherwise use UNet config's cross-attn dim
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # Hidden size depends on block
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks."):].split(".")[0])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks."):].split(".")[0])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unexpected attn processor name: {name}")
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            # diffusers 当前的 LoRAAttnProcessor 不一定暴露 alpha；这里用 scale 在权重融合时体现，
            # 训练时等价于 rank 控制容量；推理时可通过 load 时设置 scale。
        )
    unet.set_attn_processor(lora_attn_procs)
    # 只训练 LoRA 参数
    trainable_params = []
    for _, module in unet.attn_processors.items():
        trainable_params += list(module.parameters())
    # 记录 alpha 以便推理时缩放（保存到 metadata）
    unet._custom_lora_alpha = float(lora_alpha)
    return trainable_params


class ImageFolderDataset(Dataset):
    def __init__(self, root, size=512):
        self.paths = [p for p in Path(root).iterdir() if p.suffix.lower() in IMG_EXTS]
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5], [0.5])  # to [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


def encode_prompts(tokenizer, text_encoder, prompts: list[str], device):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc = text_encoder(**tokens)
    return enc.last_hidden_state  # (B, seq, dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of background images.")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="<imagenet>")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output_dir", type=str, required=True)

    # Train hyper-params
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Scaling used later at inference.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer/text encoder/vae/unet
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)

    # Freeze all base weights
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to UNet
    trainable_lora_params = add_lora_to_unet(unet, rank=args.rank, lora_alpha=args.lora_alpha)
    optimizer = torch.optim.AdamW(trainable_lora_params, lr=args.lr)

    # Noise scheduler for training
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # Dataset/Dataloader
    ds = ImageFolderDataset(args.data_dir, size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # Training loop
    unet.train()
    global_step = 0
    pbar = tqdm(total=args.train_steps, desc="LoRA training", dynamic_ncols=True)

    while global_step < args.train_steps:
        for batch in dl:
            batch = batch.to(device)  # [-1,1]

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(batch).latent_dist.sample() * 0.18215  # SD scaling

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text embeddings (constant prompt)
            encoder_hidden_states = encode_prompts(tokenizer, text_encoder, [args.prompt] * latents.shape[0], device)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if global_step >= args.train_steps:
                break

    pbar.close()

    # Save LoRA weights
    # diffusers 提供 save_attn_procs 用于仅保存 LoRA attention processors
    unet.save_attn_procs(args.output_dir)

    # 同时保存一个 metadata，记录 lora_alpha
    with open(Path(args.output_dir) / "lora_meta.txt", "w") as f:
        f.write(f"lora_alpha={getattr(unet, '_custom_lora_alpha', 1.0)}\n")
        f.write(f"rank={args.rank}\n")
        f.write(f"prompt={args.prompt}\n")

    print(f"[OK] Saved LoRA to {args.output_dir}")
    print("Usage (inference):")
    print(f"from diffusers import StableDiffusionPipeline")
    print(f"pipe = StableDiffusionPipeline.from_pretrained('{args.pretrained_model}', torch_dtype=torch.float16).to('cuda')")
    print(f"pipe.unet.load_attn_procs('{args.output_dir}', weight_name=None)  # load LoRA")
    print(f"pipe('a photo of ...').images[0].save('out.png')")
    print("你也可以按需要实现 scale 调整（合并 LoRA 前对权重乘以 alpha）。")


if __name__ == "__main__":
    main()
