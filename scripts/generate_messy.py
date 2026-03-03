# Diagnostic script: loads SAM3, runs object detection on all clean images, logs results.
# Useful for verifying SAM3 confidence scores before running the full pipeline.
#
# Before running:
#   pip install git+https://github.com/facebookresearch/sam3.git
#   huggingface-cli login

import sys
from pathlib import Path

# Allow imports from project root (datagen package)
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from datagen.prompts import OBJECT_PROMPTS

ROOT      = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
LOG_FILE  = Path(__file__).parent / "detections.txt"

SAM3_MODEL_ID        = "facebook/sam3"
CONFIDENCE_THRESHOLD = 0.5


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    if not CLEAN_DIR.is_dir():
        raise SystemExit(f"Directory not found: {CLEAN_DIR}")

    image_paths = sorted(CLEAN_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No .jpg images found in {CLEAN_DIR}")

    print(f"Loading SAM3 from {SAM3_MODEL_ID} ...")
    checkpoint = hf_hub_download(repo_id=SAM3_MODEL_ID, filename="sam3.pt")
    model = build_sam3_image_model(checkpoint_path=checkpoint)
    processor = Sam3Processor(model)
    print(f"SAM3 loaded. Processing {len(image_paths)} images ...\n")

    with open(LOG_FILE, "w") as f:
        for img_path in tqdm(image_paths, desc="Scanning images"):
            log(f, f"[{img_path.name}]")
            image = Image.open(img_path).convert("RGB")
            state = processor.set_image(image)

            any_detected = False
            for obj_key, sam_prompt in OBJECT_PROMPTS.items():
                output = processor.set_text_prompt(state=state, prompt=sam_prompt)
                scores = output.get("scores", [])
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                if scores and max(scores) >= CONFIDENCE_THRESHOLD:
                    log(f, f"  '{sam_prompt}' detected (score={max(scores):.2f})")
                    any_detected = True

            if not any_detected:
                log(f, "  (no objects detected above threshold)")
            log(f, "")


if __name__ == "__main__":
    main()
