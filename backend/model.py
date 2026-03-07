import json
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

ADAPTER_PATH = Path(__file__).resolve().parent.parent / "outputs" / "lora_adapter"

VALID_OBJECTS = {
    "pillow", "bed", "blanket", "floor", "carpet",
    "chair", "desk", "mirror", "sofa", "bath_towel", "bin", "window",
}
VALID_TYPES = {"stain", "hair", "debris", "litter", "not_emptied", "dirty"}

SYSTEM_PROMPT = (
    "You are a hotel room cleanliness inspector. "
    "The valid objects are: pillow, bed_sheet, blanket, floor, carpet, chair, "
    "desk, mirror, sofa, bath_towel, bin, window. "
    "The valid defect types are: stain, hair, debris, litter, not_emptied, dirty. "
    "Respond ONLY with valid JSON."
)

QUESTION = (
    "Inspect this hotel room for cleanliness defects. "
    "Respond ONLY with valid JSON using this schema: "
    '{"clean": true/false, "defects": [{"object": "...", "type": "...", "description": "..."}]}'
)

model = None
tokenizer = None
device = None


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load base model + LoRA adapter. Uses Unsloth on CUDA, transformers on MPS/CPU."""
    global model, tokenizer, device
    device = _get_device()

    if device == "cuda":
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        if ADAPTER_PATH.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        FastVisionModel.for_inference(model)
    else:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        dtype = torch.float16 if device == "mps" else torch.float32
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype=dtype,
        ).to(device).eval()
        if ADAPTER_PATH.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(ADAPTER_PATH)).eval()
        else:
            model = base
        tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    print(f"Model loaded on {device}")


MAX_EDGE = 1920


def _scale(image: Image.Image) -> Image.Image:
    w, h = image.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / max(w, h)
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def _run_raw(messages: list) -> dict:
    """Run inference on a pre-built messages list. Returns raw parsed JSON (no validation)."""
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = tokenizer(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.05,
        )

    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    output_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    try:
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        return {}


def _validate(raw: dict) -> dict:
    """Validate and normalise a classification result dict."""
    validated_defects = []
    for d in raw.get("defects", []):
        obj = d.get("object", "")
        dtype = d.get("type", "")
        if obj in VALID_OBJECTS and dtype in VALID_TYPES:
            validated_defects.append({
                "object": obj,
                "type": dtype,
                "description": d.get("description", ""),
            })
    is_clean = len(validated_defects) == 0 and raw.get("clean", True)
    return {"clean": is_clean, "defects": validated_defects}


def inspect(image: Image.Image, system_prompt: str = SYSTEM_PROMPT, question: str = QUESTION) -> dict:
    """Run single-turn inference on a single image."""
    image = _scale(image)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    return _validate(_run_raw(messages))
