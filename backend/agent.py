# Agent loop: forced scout (always crops 1-2 regions) → final classification with full context
import json
from PIL import Image

SCOUT_SYSTEM_PROMPT = (
    "You are a hotel room cleanliness inspector. "
    "Respond ONLY with valid JSON."
)

SCOUT_QUESTION = (
    "Before assessing this hotel room, identify 1-2 specific regions you want to inspect more closely. "
    "Focus on areas most likely to have defects: beds, pillows, floors, bins, towels, chairs. "
    "Respond ONLY with this schema: "
    '{"regions": [{"region": [x1, y1, x2, y2], "reason": "..."}, ...]} '
    "where coordinates are fractions of the image size (0.0 to 1.0), "
    "x1,y1 is top-left and x2,y2 is bottom-right. Identify exactly 1 or 2 regions."
)

CROP_FOLLOWUP = (
    "Here are the regions you requested. "
    "Now give your final cleanliness assessment using only this schema: "
    '{"clean": true/false, "defects": [{"object": "...", "type": "...", "description": "..."}]} '
    "Valid objects: pillow, bed_sheet, blanket, floor, carpet, chair, desk, mirror, sofa, bath_towel, bin, window. "
    "Valid defect types: stain, hair, debris, litter, not_emptied, dirty."
)


def inspect_with_agent(image: Image.Image) -> dict:
    from model import inspect, _scale, _run_raw, _validate

    scaled = _scale(image)
    round1_messages = [
        {"role": "system", "content": [{"type": "text", "text": SCOUT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": scaled},
            {"type": "text", "text": SCOUT_QUESTION},
        ]},
    ]

    # Round 1: model identifies regions to crop — always expected
    raw1 = _run_raw(round1_messages)
    regions = raw1.get("regions", [])

    # Clamp and validate each region, keep up to 2
    crops = []
    crop_regions = []
    for entry in regions[:2]:
        r = entry.get("region", [])
        if len(r) != 4:
            continue
        x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in r]
        w, h = image.size
        box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        if box[2] > box[0] and box[3] > box[1]:
            crops.append(_scale(image.crop(box)))
            crop_regions.append([x1, y1, x2, y2])

    # Fallback: if model didn't return usable regions, run standard single-turn pass
    if not crops:
        return inspect(image)

    # Round 2: full conversation — original image, assistant region list, all cropped images
    crop_content = [{"type": "image", "image": c} for c in crops]
    crop_content.append({"type": "text", "text": CROP_FOLLOWUP})

    round2_messages = round1_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(raw1)}]},
        {"role": "user", "content": crop_content},
    ]

    result = _validate(_run_raw(round2_messages))

    # Attach crop metadata for the frontend
    result["crop_regions"] = crop_regions
    result["crop_reasons"] = [entry.get("reason") for entry in regions[:len(crop_regions)]]

    return result
