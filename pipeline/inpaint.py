# FLUX.1 Fill inpainting: loads pipeline, applies stacked defects onto masked regions

import json
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from .prompts import DEFECT_PROMPTS

FLUX_MODEL_ID   = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_STEPS      = 30
FLUX_STEPS_TEST = 15
FLUX_GUIDANCE   = 10.0

NUM_VARIANTS      = 3
MAX_DEFECTS       = 3
TEST_NUM_VARIANTS = 3
TEST_MAX_DEFECTS  = 3

ROOT      = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
MASKS_DIR = ROOT / "data" / "masks"
MESSY_DIR = ROOT / "data" / "messy"


# Load FLUX.1 Fill with int8 transformer to fit within 16 GB VRAM
def load_flux_pipeline():
    # Load transformer in bfloat16 then quantize to int8 in-place
    transformer = FluxTransformer2DModel.from_pretrained(
        FLUX_MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    quantize_(transformer, Int8WeightOnlyConfig(version=2))
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    # T5 text encoder is offloaded to CPU (only runs once per image, not per step)
    pipe.enable_model_cpu_offload()
    return pipe


# Convert boolean numpy mask to PIL "L" image (255=inpaint, 0=keep)
def mask_to_pil(mask_array, image_size):
    arr = mask_array.squeeze().astype(np.uint8) * 255
    return Image.fromarray(arr).resize(image_size, Image.NEAREST)


# Pick one random defect per detected object, return pool for sampling
def build_variant_defects(detections):
    pool = []
    for obj_key, det in detections.items():
        if obj_key not in DEFECT_PROMPTS:
            continue
        prompt, label = random.choice(DEFECT_PROMPTS[obj_key])
        pool.append((obj_key, prompt, label, det["mask"]))
    return pool


# Chain FLUX inpainting calls so each defect stacks on the previous result
def apply_defects_sequentially(pipe, source_image, defect_list, steps=FLUX_STEPS):
    current_image = source_image.copy()
    for obj_key, prompt, label, mask_array in defect_list:
        mask_pil = mask_to_pil(mask_array, current_image.size)
        result = pipe(
            prompt=prompt,
            image=current_image,
            mask_image=mask_pil,
            height=current_image.height,
            width=current_image.width,
            num_inference_steps=steps,
            guidance_scale=FLUX_GUIDANCE,
        ).images[0]
        current_image = result
    return current_image


# Run FLUX inpainting over all images, write manifest.json to messy_dir
def run_inpaint(pipe, all_detections, image_paths, messy_dir, num_variants, max_defects, flux_steps):
    messy_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = messy_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for img_path in tqdm(image_paths, desc="Inpainting"):
        # Skip if all variant files already exist on disk
        all_done = all(
            (messy_dir / f"{img_path.stem}_v{v}.jpg").exists()
            for v in range(1, num_variants + 1)
        )
        if all_done and img_path.name in manifest:
            print(f"  Skipping {img_path.name}: already processed.")
            continue

        source_image = Image.open(img_path).convert("RGB")
        detections = all_detections[img_path]

        if not detections:
            print(f"  Skipping {img_path.name}: no objects detected above threshold.")
            continue

        variants_list = []
        for variant_num in range(1, num_variants + 1):
            defect_pool = build_variant_defects(detections)
            num_defects = random.randint(1, min(max_defects, len(defect_pool)))
            chosen = random.sample(defect_pool, num_defects)

            output_image = apply_defects_sequentially(pipe, source_image, chosen, steps=flux_steps)

            out_name = f"{img_path.stem}_v{variant_num}.jpg"
            output_image.save(messy_dir / out_name, "JPEG", quality=95)

            variants_list.append({
                "output": out_name,
                "defects": [
                    {"object": obj_key, "label": label, "prompt": prompt}
                    for obj_key, prompt, label, _ in chosen
                ],
            })

        # Write manifest after each image so progress is never lost on crash
        manifest[img_path.name] = variants_list
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    from .detect import load_detections

    parser = argparse.ArgumentParser(description="Run FLUX inpainting from saved masks in data/masks/.")
    parser.add_argument("--test", action="store_true", help="Process a random sample of 5 images.")
    args = parser.parse_args()

    if not CLEAN_DIR.is_dir():
        raise SystemExit(f"Directory not found: {CLEAN_DIR}")

    image_paths = sorted(CLEAN_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No .jpg images found in {CLEAN_DIR}")

    if args.test:
        image_paths = random.sample(image_paths, min(3, len(image_paths)))
        print(f"--test mode: {len(image_paths)} images, {TEST_NUM_VARIANTS} variant(s), {TEST_MAX_DEFECTS} defect(s), {FLUX_STEPS_TEST} steps.")

    num_variants = TEST_NUM_VARIANTS if args.test else NUM_VARIANTS
    max_defects  = TEST_MAX_DEFECTS  if args.test else MAX_DEFECTS
    flux_steps   = FLUX_STEPS_TEST   if args.test else FLUX_STEPS

    all_detections = load_detections(MASKS_DIR, image_paths)

    print("Loading FLUX.1 Fill ...")
    pipe = load_flux_pipeline()
    print("FLUX loaded.\n")

    run_inpaint(pipe, all_detections, image_paths, MESSY_DIR, num_variants, max_defects, flux_steps)
    print(f"Done. Results in {MESSY_DIR}")
