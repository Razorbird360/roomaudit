# SAM3 detection: loads model, runs object prompts over all images, returns masks

import json
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from .prompts import OBJECT_PROMPTS

SAM3_MODEL_ID = "facebook/sam3"

ROOT      = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
MASKS_DIR = ROOT / "data" / "masks"


# Run SAM3 over all images, storing the best mask per detected object
def detect_all_images(image_paths, confidence_threshold=0.5):
    print(f"Loading SAM3 from {SAM3_MODEL_ID} ...")
    checkpoint = hf_hub_download(repo_id=SAM3_MODEL_ID, filename="sam3.pt")
    model = build_sam3_image_model(checkpoint_path=checkpoint)
    processor = Sam3Processor(model)
    print(f"SAM3 loaded. Scanning {len(image_paths)} images ...\n")

    all_detections = {}
    for img_path in tqdm(image_paths, desc="SAM3 detection"):
        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)
        detections = {}

        for obj_key, sam_prompt in OBJECT_PROMPTS.items():
            output = processor.set_text_prompt(state=state, prompt=sam_prompt)
            scores = output.get("scores", [])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if not scores or max(scores) < confidence_threshold:
                continue
            best_idx = scores.index(max(scores))
            detections[obj_key] = {
                "mask":  output["masks"][best_idx].cpu().numpy(),
                "score": max(scores),
            }

        all_detections[img_path] = detections

    # Free VRAM before FLUX phase
    del model
    torch.cuda.empty_cache()
    print("SAM3 unloaded.\n")
    return all_detections


# Persist detection results to data/masks/ as .npz files + index.json
def save_detections(all_detections, masks_dir):
    masks_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for img_path, detections in all_detections.items():
        arrays = {obj_key: det["mask"] for obj_key, det in detections.items()}
        np.savez_compressed(masks_dir / img_path.stem, **arrays)
        index[img_path.stem] = {obj_key: det["score"] for obj_key, det in detections.items()}
    (masks_dir / "index.json").write_text(json.dumps(index, indent=2))


# Load saved detections back into the same dict structure detect_all_images returns
def load_detections(masks_dir, image_paths):
    all_detections = {}
    for img_path in image_paths:
        npz_path = masks_dir / f"{img_path.stem}.npz"
        if not npz_path.exists():
            all_detections[img_path] = {}
            continue
        npz = np.load(npz_path)
        all_detections[img_path] = {key: {"mask": npz[key]} for key in npz.files}
    return all_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 detection and save masks to data/masks/.")
    parser.add_argument("--test", action="store_true", help="Process a random sample of 5 images.")
    args = parser.parse_args()

    if not CLEAN_DIR.is_dir():
        raise SystemExit(f"Directory not found: {CLEAN_DIR}")

    image_paths = sorted(CLEAN_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No .jpg images found in {CLEAN_DIR}")

    if args.test:
        image_paths = random.sample(image_paths, min(5, len(image_paths)))
        print(f"--test mode: {len(image_paths)} images.")

    all_detections = detect_all_images(image_paths)
    save_detections(all_detections, MASKS_DIR)
    print(f"Masks saved to {MASKS_DIR}")
