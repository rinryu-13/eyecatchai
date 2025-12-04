# generate_image.py
# ----------------------------------------
# CPUサーバー上で ver1 LoRA を使ってアイキャッチ画像を生成するスクリプト
# - ベース    : runwayml/stable-diffusion-v1-5
# - LoRA     : models/lora_ver1_fp16.safetensors
# - 解像度   : 完全固定 16:9（912 x 512）
# - CPU向け最適化（メモリ節約・速度改善）
# - LoRA が確実に適用されるように二段構えで読み込み
# ----------------------------------------

import argparse
from pathlib import Path
import os
import traceback

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor


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
    # 🔥 LoRA 読み込み（attn_procs / adapter の両方を試す）
    # -------------------------------------------------
    has_lora = False

    if lora_path.is_file():
        print("[INFO] LoRA を読み込み中:", lora_path)

        # ① diffusers の attn_procs 形式を試す
        try:
            pipe.unet.load_attn_procs(str(lora_path))
            has_lora = any(
                isinstance(p, LoRAAttnProcessor)
                for p in pipe.unet.attn_processors.values()
            )
            print(f"[DEBUG] load_attn_procs 後 Has LoRAAttnProcessor?: {has_lora}")
        except Exception as e:
            print("[WARN] unet.load_attn_procs に失敗:", repr(e))

        # ② まだ刺さっていない場合は adapter 形式を試す
        if not has_lora:
            try:
                pipe.load_lora_weights(str(lora_path))
                has_lora = any(
                    isinstance(p, LoRAAttnProcessor)
                    for p in pipe.unet.attn_processors.values()
                )
                print(f"[DEBUG] load_lora_weights 後 Has LoRAAttnProcessor?: {has_lora}")
            except Exception as e:
                print("[WARN] pipe.load_lora_weights に失敗:", repr(e))

        if has_lora:
            # diffusers ≥ 0.24 系なら fuse_lora で統合可能
            if hasattr(pipe, "fuse_lora"):
                try:
                    print("[INFO] fuse_lora を実行（LoRA をモデルに統合）")
                    pipe.fuse_lora()
                    print("[INFO] LoRA 統合完了（ver1 有効化）")
                except Exception as e:
                    # fuse に失敗しても、動的 LoRA としては効いているので致命傷ではない
                    print("[WARN] fuse_lora は失敗しましたが、LoRA 自体は適用済みの可能性があります:", repr(e))
        else:
            print("[WARN] LoRA(ver1) を読み込みましたが、LoRAAttnProcessor が見つかりませんでした。")
            print("       → ver1 の学習結果が効いていない可能性があります。")
    else:
        print("[WARN] LoRA(ver1) が見つからなかったため、素のSD1.5で生成します")

    # 最終的な確認
    final_has_lora = any(
        isinstance(p, LoRAAttnProcessor)
        for p in pipe.unet.attn_processors.values()
    )
    print(f"[CHECK] 最終的な Has LoRAAttnProcessor?: {final_has_lora}")

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
    num_inference_steps: int = 24,   # ← CPUなので 30 → 24 に少しだけ短縮（必要なら 20〜15 まで下げてもOK）
    guidance_scale: float = 7.0,     # 少しだけ下げて収束を早める
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

    # CPU なので grad 無効 & Inference Mode
    torch.set_grad_enabled(False)
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
    parser.add_argument("--steps", type=int, default=24)   # デフォルトも 24 に寄せる
    parser.add_argument("--guidance", type=float, default=7.0)
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
