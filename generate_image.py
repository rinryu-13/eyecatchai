# generate_image.py
# ---------------------------------------------------------
# CPUサーバー上で ver1 LoRA を使ってアイキャッチ画像を生成するスクリプト
# - ベース   : runwayml/stable-diffusion-v1-5
# - LoRA    : models/lora_ver1_fp16.safetensors
# - 解像度  : 16:9（912 x 512 固定）
# - CPU向けにメモリ節約設定
# ---------------------------------------------------------

import argparse
from pathlib import Path
import os
import traceback

import torch
from diffusers import StableDiffusionPipeline


# ---------------------------------------------------------
# SD1.5 + LoRA(ver1) をロード
# ---------------------------------------------------------
def load_pipeline(model_name: str, lora_path: Path, device: str = "cpu") -> StableDiffusionPipeline:
    print("[INFO] Base model:", model_name)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # CPU なので fp32
        safety_checker=None,
    )

    # メモリ節約オプション
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        print("[INFO] attention_slicing enabled")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("[INFO] vae_slicing enabled")

    # ---- LoRA 適用（公式ルートのみ。fallback は一切しない）----
    if lora_path.is_file():
        print("[INFO] Loading LoRA (ver1):", lora_path)
        pipe.load_lora_weights(str(lora_path))

        # diffusers >= 0.24 では fuse_lora で重みに焼き込める
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora()
            print("[INFO] fuse_lora done → LoRA merged into UNet/TextEncoder")
        else:
            print("[WARN] fuse_lora not available, but LoRA adapter is still active.")
    else:
        print("[WARN] LoRA file not found:", lora_path)
        print("       → pure SD1.5 will be used (no training effect)")

    pipe.to(device)
    return pipe


# ---------------------------------------------------------
# 画像生成（常に 912x512 / 16:9）
# ---------------------------------------------------------
def generate(
    prompt: str,
    negative: str,
    output_path: Path,
    steps: int,
    guidance: float,
    seed: int,
    device: str = "cpu",
):
    width, height = 912, 512  # 16:9 固定

    project_root = Path(__file__).resolve().parent
    lora_path = project_root / "models" / "lora_ver1_fp16.safetensors"

    pipe = load_pipeline(
        "runwayml/stable-diffusion-v1-5",
        lora_path,
        device=device,
    )

    generator = torch.Generator(device=device).manual_seed(seed)

    print("[INFO] Generation start")
    print(f"       prompt   : {prompt}")
    print(f"       steps    : {steps}")
    print(f"       guidance : {guidance}")
    print(f"       size     : {width}x{height}")
    print(f"       output   : {output_path}")

    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    print("[INFO] Saved:", output_path)


# ---------------------------------------------------------
# メイン
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ver1 LoRA blog thumbnail generator (CPU)"
    )
    parser.add_argument("--prompt", required=True, help="生成プロンプト（記事タイトルなど）")
    parser.add_argument(
        "--negative",
        default="text artifact, watermark, bad anatomy, low quality, blurry",
        help="ネガティブプロンプト",
    )
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output/ver1_sample.png")
    parser.add_argument("--device", default="cpu", choices=["cpu"])

    args = parser.parse_args()

    output_path = Path(args.output).resolve()

    print("[INFO] generate_image.py start")
    print("[INFO] device:", args.device)

    generate(
        prompt=args.prompt,
        negative=args.negative,
        output_path=output_path,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        device=args.device,
    )

    print("[INFO] finished")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("[ERROR] RuntimeError:", repr(e))
        if "not enough memory" in str(e) or "DefaultCPUAllocator" in str(e):
            print("[HINT] CPUメモリ不足の可能性: steps を減らす or VPSメモリを増やす")
        traceback.print_exc()
    except Exception as e:
        print("[ERROR] Unexpected error:", repr(e))
        traceback.print_exc()
