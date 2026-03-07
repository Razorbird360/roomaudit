# roomaudit

Fine-tuning of a vision model to detect hotel room cleanliness defects, with an agentic web interface for live inspection. The inference pipeline is multi-step: the model first identifies regions worth closer inspection, receives cropped views of those areas, then gives its final verdict with full context.

**Pipeline:** clean room images → SAM3 segmentation → FLUX.1 Fill inpainting → messy images → fine-tune Qwen3-VL

---

## Project structure

```
frontend/              — React + Vite web app
  src/
    components/        — HowItWorks, Inspect, StepCard, CurvedArrow, ResultCard
    components/ui/     — shadcn/ui (tabs, card, badge, sonner)
  public/pipeline/     — step images for "How It Works" tab
backend/               — FastAPI inference server
  main.py              — API: POST /inspect
  model.py             — Qwen3-VL + LoRA loading (CUDA/MPS/CPU)
  agent.py             — two-round agentic inspection loop
datagen/
  prompts.py           — OBJECT_PROMPTS and DEFECT_PROMPTS
  detect.py            — SAM3 detection, saves masks to data/masks/
  inpaint.py           — FLUX.1 Fill inpainting, saves results to data/messy/
  run.py               — full data generation entry point
training/
  train.ipynb          — QLoRA fine-tuning notebook (dataset build, training, metrics plot)
scripts/
  normalize_images.py  — one-off: resize + PNG→JPG in-place
  generate_messy.py    — diagnostic: run SAM3 and log detection scores
data/
  clean/               — source images (JPG, normalized)
  masks/               — SAM3 output masks (.npz per image)
  messy/               — generated defect images + manifest.json
outputs/
  lora_adapter/        — saved LoRA adapter after training
```

---

## Quick start (web app)

```bash
npm install        # installs frontend deps + backend pip deps
npm run dev        # starts both frontend (:5173) and backend (:8000)
```

Requires Python 3.11+ with a venv activated. On CUDA machines, install `backend/requirements-cuda.txt` instead for Unsloth support.

The backend auto-detects your device — CUDA (4-bit via Unsloth), MPS (fp16), or CPU (fp32).

---

## Data generation setup

These must be installed in order — PyTorch and Unsloth have to come before everything else.

**1. PyTorch with CUDA 12.8+** (required for RTX 5070 Ti / Blackwell):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**2. SAM3** (not on PyPI, install from GitHub):
```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

**3. Unsloth** (must come after PyTorch):
```bash
pip install unsloth
```

**4. Everything else:**
```bash
pip install -r datagen/requirements.txt
```

**5. Log in to HuggingFace** (SAM3 and FLUX.1 Fill are gated models):
```bash
huggingface-cli login
```

---

## Running the pipeline

**Step 1 — normalize source images** (one-time):
```bash
python scripts/normalize_images.py
```

**Step 2 — generate defect images** (detection + inpainting):
```bash
python -m datagen.run
```

**Test run** (3 images, fewer steps — verify before a full overnight run):
```bash
python -m datagen.inpaint --test
```

**Step 3 — fine-tune:**
Open `training/train.ipynb` and run cells top to bottom.

---

## Models

| Role | Model |
|---|---|
| Segmentation | `facebook/sam3` |
| Inpainting | `black-forest-labs/FLUX.1-Fill-dev` |
| Fine-tuning target | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` |
| Inference (MPS/CPU) | `Qwen/Qwen3-VL-4B-Instruct` |


