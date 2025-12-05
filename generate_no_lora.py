# generate_no_lora.py
# ----------------------------------------
# LoRA を一切読み込まず、純粋な SD1.5 のみで画像生成
# ----------------------------------------

import argparse
from pathlib import Path
import os
import torch
from diffusers import StableDiffusionPipeline

def main():
    parser = argparse.ArgumentParser(description="SD1.5 素の状態で生成（LoRAなし）")

    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--negative_prompt",
        default="low quality, bad anatomy, blurry, watermark, noisy background",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./output/no_lora.png")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.0)

    args = parser.parse_args()

    width, height = 912, 512  # 固定16:9

    print("=== 素の SD1.5 をロード中 ===")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    pipe.to("cpu")

    generator = torch.Generator("cpu").manual_seed(args.seed)

    print("=== 生成開始（LoRAなし） ===")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(str(output_path))
    print("保存:", output_path)

if __name__ == "__main__":
    main()
