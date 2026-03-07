# One-time script: runs SAM3 on HIW1.jpg then FLUX to inpaint 4 specific defects, saves as HIW3.jpg
# Output: frontend/src/assets/HIW3.jpg
#
# Run from project root: python scripts/gen_HIW3.py

import gc
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from datagen.inpaint import mask_to_pil, load_flux_pipeline

ASSETS_DIR  = ROOT / "frontend" / "src" / "assets"
INPUT_IMAGE = ASSETS_DIR / "HIW1.jpg"
OUT_FILE    = ASSETS_DIR / "HIW3.jpg"

CONFIDENCE_THRESHOLD = 0.5
MAX_EDGE = 1920
FLUX_STEPS   = 40


# Defects to apply sequentially, each mapped to the SAM3 object prompt to detect
DEFECTS = [
    # ("bed",    "bed",        "a brown soup stain on the white bed",              "stain"),
    ("floor",  "floor",      "a few tissue papers scattered on the floor",            "litter"),
    # ("chair",  "chair",      "food crumbs scattered on the seat cushion",        "debris"),
]


def run():
    if not INPUT_IMAGE.exists():
        raise SystemExit(f"HIW1.jpg not found at {INPUT_IMAGE}")

    image = Image.open(INPUT_IMAGE).convert("RGB")
    w, h = image.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"Resized image to {image.size}")

    # Phase 1: SAM3 — detect only the objects needed for the 4 defects
    print("Loading SAM3 ...")
    checkpoint = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
    sam_model  = build_sam3_image_model(checkpoint_path=checkpoint)
    processor  = Sam3Processor(sam_model)
    state      = processor.set_image(image)
    print("SAM3 loaded. Running detection ...\n")

    masks = {}
    needed_objects = {obj_key: prompt for obj_key, prompt, _, _ in DEFECTS}
    for obj_key, prompt in needed_objects.items():
        output = processor.set_text_prompt(state=state, prompt=prompt)
        scores = output.get("scores", [])
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not scores or max(scores) < CONFIDENCE_THRESHOLD:
            print(f"  {obj_key}: not detected (skipping defect)")
            continue
        best_idx  = scores.index(max(scores))
        masks[obj_key] = output["masks"][best_idx].cpu().numpy()
        print(f"  {obj_key}: detected (score={max(scores):.3f})")

    # Free all SAM3 objects from VRAM before loading FLUX
    del sam_model, processor, state
    gc.collect()
    torch.cuda.empty_cache()
    print("\nSAM3 unloaded.")

    # Phase 2: FLUX — apply each defect sequentially onto the previous result
    print("Loading FLUX.1 Fill ...")
    pipe = load_flux_pipeline()
    print("FLUX loaded. Inpainting ...\n")

    current_image = image.copy()
    for obj_key, _, prompt, label in DEFECTS:
        if obj_key not in masks:
            print(f"  Skipping {label} on {obj_key}: no mask available")
            continue
        print(f"  Applying {label} to {obj_key} ...")
        mask_pil = mask_to_pil(masks[obj_key], current_image.size)
        current_image = pipe(
            prompt=prompt,
            image=current_image,
            mask_image=mask_pil,
            height=current_image.height,
            width=current_image.width,
            num_inference_steps=FLUX_STEPS,
            guidance_scale=10.0,
        ).images[0]

    current_image.save(OUT_FILE, "JPEG", quality=95)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    run()
