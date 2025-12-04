# generate_image.py
# ----------------------------------------
# CPUサーバー上で ver1 LoRA を使ってアイキャッチ画像を生成するスクリプト
# - ベース    : runwayml/stable-diffusion-v1-5
# - LoRA     : models/lora_ver1_fp16.safetensors
# - 解像度   : 完全固定 16:9（912 x 512）
# - CPU向け最適化（メモリ節約・速度改善）
# - LoRA が確実に適用される最新版コード
# ----------------------------------------

import argparse
from pathlib import Path
import os
import traceback

import torch
from diffusers import StableDiffusionPipeline


# -----------------------------------------------------
# Stable Diffusion v1.5 + LoRA(ver1) を読み込む
# -----------------------------------------------------
def load_pipeline(base_model_id: str, lora_path: Path, device: str = "cpu") -> StableDiffusionPipeline:
    print("[INFO] Base model を読み込み中:", base_model_id)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,  # CPU は fp32 固定
        safety_checker=None,
    )

    # CPU向けメモリ節約
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        print("[INFO] attention_slicing を有効化しました")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("[INFO] vae_slicing を有効化しました")

    # -------------------------------------------------
    # 🔥 LoRA 読み込み（最新版の安定動作）
    # -------------------------------------------------
    if lora_path.is_file():
        print("[INFO] LoRA を読み込み中:", lora_path)

        pipe.load_lora_weights(str(lora_path))

        # diffusers ≥ 0.24 は fuse_lora が必要
        if hasattr(pipe, "fuse_lora"):
            print("[INFO] fuse_lora を実行（LoRA をモデルに統合）")
            pipe.fuse_lora()
            print("[INFO] LoRA 統合完了（ver1 有効化）")

    else:
        print("[WARN] LoRA(ver1) が見つからなかったため、素のSD1.5で生成します")

    pipe.to(device)
    return pipe


# -----------------------------------------------------
# 画像生成（常に 16:9 = 912 × 512）
# -----------------------------------------------------
def generate_image(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    device: str = "cpu",
) -> None:

    # ★ 16:9 完全固定
    width = 912
    height = 512

    project_root = Path(__file__).resolve().parent
    lora_path = project_root / "models" / "lora_ver1_fp16.safetensors"

    pipe = load_pipeline("runwayml/stable-diffusion-v1-5", lora_path, device=device)

    generator = torch.Generator(device=device).manual_seed(seed)

    os.makedirs(output_path.parent, exist_ok=True)

    print("[INFO] 画像生成を開始します...")
    print(f"       解像度: {width}x{height} (16:9固定)")
    print(f"       steps: {num_inference_steps}")
    print(f"       guidance: {guidance_scale}")
    print(f"       LoRA: lora_ver1_fp16.safetensors")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

    image = result.images[0]
    image.save(str(output_path))

    print("[INFO] 保存しました:", output_path)


# -----------------------------------------------------
# メイン処理
# -----------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="16:9 固定 ver1 LoRA アイキャッチ画像生成スクリプト（CPUサーバー用）")

    parser.add_argument("--prompt", required=False, help="生成プロンプト（未指定ならサンプルを使用）")

    parser.add_argument(
        "--negative_prompt",
        default=(
            "low quality, bad anatomy, blurry, watermark, text artifact, "
            "photo, realistic photo, 3d render, noisy background, distorted figure"
        ),
        help="ネガティブプロンプト",
    )

    parser.add_argument("--output", default="./output/ver1_sample.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"])

    args = parser.parse_args()

    # プロンプト未指定 → アイキャッチ用の汎用プロンプト
    if args.prompt is None:
        args.prompt = (
            "eye-catching blog thumbnail, clean flat illustration, pastel colors, "
            "Japanese blog style, layout-friendly, simple vector design"
        )

    output_path = Path(args.output).resolve()

    print("[INFO] generate_image.py を開始")
    print("[INFO] 使用デバイス:", args.device)

    generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=output_path,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        device=args.device,
    )

    print("[INFO] 正常終了しました")


# -----------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        msg = str(e)
        print("[ERROR] RuntimeError:", repr(e))
        if "not enough memory" in msg or "DefaultCPUAllocator" in msg:
            print("[HINT] CPUメモリ不足の可能性:")
            print("  - steps を 20 か 15 に下げる")
            print("  - VPS のRAMプランを増やす")
        traceback.print_exc()
    except Exception as e:
        print("[ERROR] 予期しない例外:", repr(e))
        traceback.print_exc()
