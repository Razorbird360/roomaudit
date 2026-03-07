# One-time script: runs SAM3 on HIW1.jpg, saves a colour-coded mask overlay as HIW2.jpg
# Output: frontend/src/assets/HIW2.jpg
#
# Run from project root: python scripts/gen_HIW2.py

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from datagen.prompts import OBJECT_PROMPTS

ASSETS_DIR  = ROOT / "frontend" / "src" / "assets"
INPUT_IMAGE = ASSETS_DIR / "HIW1.jpg"
OUT_FILE    = ASSETS_DIR / "HIW2.jpg"

# Distinct saturated colours per object key
COLOURS = {
    "pillow":     (200,  30,  30, 180),   # deep red
    "bed":        (210, 110,   0, 180),   # dark orange
    "blanket":    (180, 160,   0, 180),   # dark yellow
    "floor":      ( 20, 140,  20, 170),   # dark green
    "carpet":     (  0,  90,  10, 175),   # very dark green
    "chair":      (  0, 110, 210, 180),   # strong blue
    "desk":       (  0,  50, 180, 180),   # deep blue
    "mirror":     (140,   0, 210, 180),   # deep purple
    "sofa":       (  0, 170, 200, 180),   # dark cyan
    "bath_towel": (200,  30, 140, 180),   # deep pink
    "bin":        (110,  60,   0, 180),   # dark brown
    "window":     (100, 100, 100, 175),   # dark grey
}

CONFIDENCE_THRESHOLD = 0.5


def run():
    if not INPUT_IMAGE.exists():
        raise SystemExit(f"HIW1.jpg not found at {INPUT_IMAGE}")

    image = Image.open(INPUT_IMAGE).convert("RGB")

    print("Loading SAM3 ...")
    checkpoint = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
    model      = build_sam3_image_model(checkpoint_path=checkpoint)
    processor  = Sam3Processor(model)
    state      = processor.set_image(image)
    print("SAM3 loaded. Running detection ...\n")

    detections = {}
    for obj_key, prompt in OBJECT_PROMPTS.items():
        output = processor.set_text_prompt(state=state, prompt=prompt)
        scores = output.get("scores", [])
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not scores or max(scores) < CONFIDENCE_THRESHOLD:
            continue
        best_idx = scores.index(max(scores))
        detections[obj_key] = {
            "mask":  output["masks"][best_idx].cpu().numpy(),
            "score": round(max(scores), 3),
        }

    # Print results
    print(f"Detected {len(detections)} objects:\n")
    for obj_key, det in sorted(detections.items(), key=lambda x: -x[1]["score"]):
        print(f"  {obj_key:<12}  score={det['score']:.3f}")

    # Build colour-coded overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for obj_key, det in detections.items():
        colour = COLOURS.get(obj_key, (100, 100, 100, 175))
        mask   = det["mask"]
        if mask.ndim == 3:
            mask = mask[0]
        mask_bool    = mask.astype(bool)
        colour_layer = Image.new("RGBA", image.size, colour)
        mask_img     = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
        overlay.paste(colour_layer, mask=mask_img)

    result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    result.save(OUT_FILE, quality=92)
    print(f"\nSaved: {OUT_FILE}")

    # Save detections JSON alongside for hardcoding the frontend tooltip
    tooltip_data = {
        obj_key: {"score": det["score"], "colour": COLOURS.get(obj_key, (100, 100, 100, 175))[:3]}
        for obj_key, det in detections.items()
    }
    (ASSETS_DIR / "HIW2_detections.json").write_text(json.dumps(tooltip_data, indent=2))
    print(f"Saved: {ASSETS_DIR / 'HIW2_detections.json'}")


if __name__ == "__main__":
    run()
