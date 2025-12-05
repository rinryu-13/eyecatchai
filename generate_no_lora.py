# generate_no_lora.py

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="text artifact, watermark, bad anatomy")
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output/no_lora_sample.png")
    a = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.to("cpu")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator("cpu").manual_seed(a.seed)

    img = pipe(
        prompt=a.prompt,
        negative_prompt=a.negative,
        num_inference_steps=a.steps,
        guidance_scale=a.guidance,
        width=912,
        height=512,
        generator=g,
    ).images[0]

    img.save(a.output)
    print("saved:", a.output)

if __name__ == "__main__":
    main()
