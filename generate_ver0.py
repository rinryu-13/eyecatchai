# generate_ver0.py
# ----------------------------------------
# CPUサーバー上で ver0 LoRA を使ってアイキャッチ画像を生成するスクリプト
# - ベース: runwayml/stable-diffusion-v1-5
# - LoRA : models/lora_ver0_fp16.safetensors
# ----------------------------------------

import argparse
from pathlib import Path
import os
import traceback

import torch
from diffusers import StableDiffusionPipeline


def load_pipeline(base_model_id: str, lora_path: Path, device: str = "cpu") -> StableDiffusionPipeline:
    """
    Stable Diffusion v1.5 + LoRA(ver0) をロードして、指定デバイスに載せる。
    本番環境（CPUサーバー）想定。
    """
    print("[INFO] Base model を読み込み中:", base_model_id)

    # CPU なので float32 でロードする（fp16 は CPU 非対応）
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        safety_checker=None,  # 公開サービスにする場合は True/独自フィルタを検討
    )

    # メモリ節約オプション（あれば有効化）
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        print("[INFO] attention_slicing を有効化しました")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("[INFO] vae_slicing を有効化しました")

    # LoRA の読み込み
    if lora_path is not None and lora_path.is_file():
        print("[INFO] LoRA を読み込み中:", lora_path)
        pipe.load_lora_weights(str(lora_path))

        # diffusers のバージョンによっては fuse_lora が無い場合もあるので、存在チェック
        if hasattr(pipe, "fuse_lora"):
            print("[INFO] LoRA をモデルに適用 (fuse_lora)")
            pipe.fuse_lora()
    else:
        print("[WARN] LoRA ファイルが見つからなかったので、素の SD1.5 で生成します。")

    pipe.to(device)
    return pipe


def generate_image(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    device: str = "cpu",
) -> None:
    """
    1枚だけ画像を生成して保存する。
    CPU サーバー前提。メモリ不足が発生した場合は解像度や steps を下げる。
    """
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    lora_path = models_dir / "lora_ver0_fp16.safetensors"

    base_model_id = "runwayml/stable-diffusion-v1-5"

    pipe = load_pipeline(base_model_id, lora_path, device=device)

    # 乱数シード固定
    generator = torch.Generator(device=device).manual_seed(seed)

    os.makedirs(output_path.parent, exist_ok=True)

    print("[INFO] 画像生成を開始します...")
    print(f"[INFO] 解像度: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}")
    print("[INFO] 出力先:", output_path)

    # 実際の生成処理
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
    print("[INFO] 画像を保存しました:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ver0 LoRA を使って記事アイキャッチ画像を生成するスクリプト（CPUサーバー用）"
    )
    parser.add_argument(
        "--prompt",
        required=False,
        help="生成に使うプロンプト（タイトル＋要約など）。未指定ならサンプル文を使用。",
    )
    parser.add_argument(
        "--negative_prompt",
        default="low quality, bad anatomy, blurry, text artifact, watermark",
        help="ネガティブプロンプト（省略可）",
    )
    parser.add_argument(
        "--output",
        default="./output/ver0_sample.png",
        help="出力画像パス（既定: ./output/ver0_sample.png）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード（同じシードなら同じ画像になります）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="サンプリングステップ数（多いほど綺麗になるが時間がかかる）",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="ガイダンススケール（プロンプトの反映の強さ）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,  # 本番用のデフォルト解像度
        help="生成画像の横幅（既定: 512）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="生成画像の高さ（既定: 512）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu"],  # 将来GPU対応するなら ["cpu", "cuda"] に拡張
        help="使用するデバイス（現在は cpu のみ想定）",
    )

    args = parser.parse_args()

    # プロンプト未指定ならサンプルを使う
    if args.prompt is None:
        args.prompt = (
            "アイキャッチ用サムネイル, ピラティス, 日本人女性, 全身, "
            "明るい, 清潔感, ブログ用イラスト, 日本語テキスト入り"
        )

    output_path = Path(args.output).resolve()

    print("[INFO] generate_ver0.py を開始します")
    print("[INFO] 使用デバイス:", args.device)

    generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=output_path,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        device=args.device,
    )

    print("[INFO] スクリプトが正常終了しました")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        # メモリ不足などの RuntimeError を少し親切に表示
        msg = str(e)
        print("[ERROR] RuntimeError が発生しました:", repr(e))
        if "DefaultCPUAllocator" in msg or "not enough memory" in msg:
            print("[HINT] CPU メモリ不足の可能性があります。以下を試してください:")
            print("  - VPS のメモリサイズを増やすプランに変更する")
            print("  - --width と --height を 384 や 320 に下げて再実行する")
            print("  - --steps を 20 や 15 に下げる")
        traceback.print_exc()
    except Exception as e:
        print("[ERROR] 予期しない例外が発生しました:", repr(e))
        traceback.print_exc()
