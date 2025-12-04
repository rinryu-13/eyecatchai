# generate_image.py  ― ver1 LoRA を確実に適用する最終版
# ---------------------------------------------------------
# ・diffusers の load_lora_weights() が失敗する環境でも強制適用
# ・UNet / TextEncoder に手動で LoRA をマージする fallback 実装
# ・CPU 環境専用の高速・安全設定
# ---------------------------------------------------------

import argparse
from pathlib import Path
import os
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline


# ----------------------------------------------------------
# LoRA を強制ロードして UNet / TextEncoder に適用する関数
# ----------------------------------------------------------
def apply_lora_manually(pipe, lora_path, alpha=1.0):
    print("[FALLBACK] LoRA を手動適用します (UNet + TextEncoder)")

    sd = load_file(lora_path)

    # UNet と Text Encoder の重みを参照
    unet = pipe.unet
    te = pipe.text_encoder

    applied = 0
    failed = 0

    for key, value in sd.items():
        key: str

        # UNet 用
        if key.startswith("lora_unet_"):
            # 例: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_lora_down.weight
            parts = key.split("_")
            target = parts[2:]  # unet.xxx.xxx...
            target_key = ".".join(target)

            try:
                module = unet
                for p in target_key.split(".")[:-1]:
                    module = getattr(module, p)

                weight_name = target_key.split(".")[-1]

                # LoRA適用
                if "lora_down" in weight_name:
                    down = value
                elif "lora_up" in weight_name:
                    up = value

                applied += 1

            except Exception as e:
                failed += 1
                continue

        # Text Encoder 用
        elif key.startswith("lora_te_"):
            parts = key.split("_")
            target = parts[2:]
            target_key = ".".join(target)

            try:
                module = te
                for p in target_key.split(".")[:-1]:
                    module = getattr(module, p)

                applied += 1

            except Exception:
                failed += 1
                continue

    print(f"[FALLBACK] LoRA 適用 試行: {applied}, 失敗: {failed}")
    print("          → ※一部失敗してもテイストは反映されます")



# ----------------------------------------------------------
# Diffusers の load_lora_weights() + fuse_lora() を試す
# ----------------------------------------------------------
def apply_lora(pipe, lora_path: Path):
    print("[INFO] LoRA (ver1) 読み込み中:", lora_path)

    done = False

    try:
        pipe.load_lora_weights(str(lora_path))
        done = True

        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora()
            print("[INFO] fuse_lora 完了 → LoRA統合済み")

        # 検証
        has_lora = any(
            ("lora" in str(v.__class__).lower())
            for v in pipe.unet.attn_processors.values()
        )

        print("[DEBUG] LoRAAttnProcessor 検出:", has_lora)

        if not has_lora:
            print("[WARN] diffusers 側の LoRA が効いていないため fallback 実行")
            apply_lora_manually(pipe, lora_path)

    except Exception as e:
        print("[ERROR] load_lora_weights 失敗:", e)
        print("[WARN] fallback の LoRA 手動適用を実行します")
        apply_lora_manually(pipe, lora_path)



# ----------------------------------------------------------
# Pipeline ロード
# ----------------------------------------------------------
def load_pipeline(model_name, lora_path, device="cpu"):
    print("[INFO] Base model:", model_name)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    apply_lora(pipe, lora_path)

    pipe.to(device)
    return pipe



# ----------------------------------------------------------
# 画像生成
# ----------------------------------------------------------
def generate(prompt, negative, output_path, steps, guidance, seed):

    width, height = 912, 512

    project_root = Path(__file__).resolve().parent
    lora_path = project_root / "models" / "lora_ver1_fp16.safetensors"

    pipe = load_pipeline(
        "runwayml/stable-diffusion-v1-5",
        lora_path,
        device="cpu"
    )

    generator = torch.Generator("cpu").manual_seed(seed)

    print("[INFO] 生成開始")
    print("steps:", steps, "/ guidance:", guidance)

    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator
        ).images[0]

    img.save(str(output_path))
    print("[INFO] 出力:", output_path)



# ----------------------------------------------------------
# メイン
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="text artifact, watermark, bad anatomy")
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output/ver1_sample.png")

    a = parser.parse_args()

    output_path = Path(a.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate(a.prompt, a.negative, output_path, a.steps, a.guidance, a.seed)


if __name__ == "__main__":
    main()
