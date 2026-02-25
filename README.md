# roomaudit

Fine-tuning of a vision model to detect hotel room cleanliness defects.

**Pipeline:** clean room images → SAM3 segmentation → FLUX.1 Fill inpainting → messy images → fine-tune Qwen3-VL

---

## How it works

1. **Normalize** source images to JPG, max 1920px longest edge
2. **Detect** objects in each image (pillows, bedsheets, floor, etc.) using SAM3
3. **Inpaint** defects onto detected objects using FLUX.1 Fill (hair, stains, crumples, litter, etc.)
4. **Fine-tune** Qwen3-VL-8B-Instruct on the generated defect images using Unsloth

---

## Project structure

```
pipeline/
  prompts.py   — OBJECT_PROMPTS and DEFECT_PROMPTS
  detect.py    — SAM3 detection, saves masks to data/masks/
  inpaint.py   — FLUX.1 Fill inpainting, saves results to data/messy/
  run.py       — full pipeline entry point
scripts/
  normalize_images.py   — one-off: resize + PNG→JPG in-place
  generate_messy.py     — diagnostic: run SAM3 and log detection scores
data/
  clean/    — source images (JPG, normalized)
  masks/    — SAM3 output masks (.npz per image)
  messy/    — generated defect images + manifest.json
```

---

## Setup

**1. Install PyTorch with CUDA 12.8+ (required for RTX 5070 Ti):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**2. Install SAM3:**
```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

**3. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

**4. Log in to HuggingFace** (SAM3 and FLUX.1 Fill are gated models):
```bash
huggingface-cli login
```

---

## Running the pipeline

**Step 1 — normalize source images** (one-time):
```bash
python scripts/normalize_images.py
```

**Step 2 — run full pipeline** (detection + inpainting):
```bash
python -m pipeline.run
```

**Test run** (5 images, 1 variant, 8 steps — verifies the pipeline before a full overnight run):
```bash
python -m pipeline.inpaint --test
```

**Inpainting only** (if masks already exist in `data/masks/`):
```bash
python -m pipeline.inpaint
```

---

## Models

| Role | Model |
|---|---|
| Segmentation | `facebook/sam3` |
| Inpainting | `black-forest-labs/FLUX.1-Fill-dev` |
| Fine-tuning target | `Qwen/Qwen3-VL-8B-Instruct` (via Unsloth) |
