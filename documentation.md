# roomaudit — project log

## The data problem

The obvious first instinct was to find an existing dataset. I looked at:

- **Hotels-50K** — the images weren't suitable, wrong kind of content
- **Places365** — too low resolution, wouldn't survive inpainting
- **Booking.com scraping** — ToS violation, and hard to find many suitable images anyway
- **Flickr** — manually browsed but requires a Pro account ($8/month) to download in bulk, not worth it

Eventually settled on manually downloading from **Unsplash**, **Pexels**, and **Flickr** (individual downloads). Ended up with 218 clean hotel room images.

---

## Approach

Since there's no messy room data, the plan is to generate it synthetically:

1. Take clean room images
2. Segment objects in each room (bed, floor, mirror, etc.)
3. Use an inpainting model to paint realistic defects onto those regions (stains, hair, litter, etc.)
4. Use the generated messy images + defect labels to fine-tune a vision model

That gives me (clean image, messy image, defect labels) pairs without needing any real messy room photos.

---

## Model choices and what I rejected

### Segmentation

Went with **SAM3** (Meta, `facebook/sam3`, ~848M params, ~4GB VRAM bfloat16). It supports text-prompt-based segmentation which is exactly what I needed — I can just say "pillow" and get a mask back. Uses `build_sam3_image_model` + `Sam3Processor` from the sam3 package. Has to be installed directly from GitHub, not on PyPI.

### Inpainting

This took more research.

- **Base FLUX.1 Dev** — rejected. It's text-to-image only, not inpainting. Would alter regions outside the mask.
- **FLUX.2 Dev** — rejected. 32B parameters, needs ~62GB VRAM. Way out of range.
- **Hyper-FLUX step distillation LoRA** — looked into this for speeding up inference. Rejected because it was trained on base FLUX.1 Dev, not Fill Dev. The Fill model has different input conditioning (masked image + binary mask as extra channels), so base LoRAs don't transfer.

Settled on **FLUX.1 Fill Dev** (`black-forest-labs/FLUX.1-Fill-dev`). Purpose-built for masked inpainting, doesn't bleed outside the mask, takes a prompt + image + mask and fills only the masked region.

To fit it on 16GB VRAM: load the transformer in bfloat16 then quantize it to int8 in-place using `torchao`. That brings it down enough. Then use `enable_model_cpu_offload()` so components that aren't actively computing live in system RAM.

### Runtime

Did the math on running the full pipeline at full settings (28 steps, 5 variants, 3 stacked defects, 218 images) — came out to 75–175 hours. Not viable.

Considered **cloud** (Vast.ai, priced it out at ~$6–8 on an A100). Decided to run locally instead to explore local optimisation first.

Ended up reducing: **768px resolution, 3 variants per image, up to 3 defects per variant, 30 steps**. Brings it down to something that can run across a few overnight sessions.

### Fine-tuning model

- **Qwen3-VL 4B** — considered as a smaller option. Scores 92–95% of the 7B on vision benchmarks, which is close but the quality gap matters for an inspection task.
- **Qwen3-VL 8B** — needs ~24GB VRAM, doesn't fit on 16GB even with QLoRA.
- **Qwen2.5-VL 7B** — what I'm going with. Loads in bfloat16 at ~14–16GB, then run 4-bit QLoRA via Unsloth's `FastVisionModel` to fit training within 16GB. LoRA adapters train on top of frozen quantized weights.

---

## Pipeline

```
scripts/normalize_images.py   — resize all images to max 1920px, snap to multiple of 16, PNG→JPG
pipeline/detect.py            — SAM3 detection, saves binary masks as .npz files to data/masks/
pipeline/inpaint.py           — FLUX.1 Fill inpainting, saves variants + manifest.json to data/messy/
pipeline/run.py               — entry point, runs both phases in sequence
```

VRAM strategy: SAM3 and FLUX can't coexist in 16GB. SAM3 runs first, then gets deleted and VRAM is freed with `torch.cuda.empty_cache()`, then FLUX loads.

### Defect prompts

12 object categories: pillow, bed sheet, blanket, floor, carpet, chair, desk, mirror, sofa, bath towel, bin, window.

~35 defect prompts across those. Went through several revision passes — removed things like geometry deformations, object removals, and anything where the defect would visually overflow outside the mask boundary. The prompts need to describe something that stays contained within the object region.

### Crash resume

The full inpainting run across 218 images takes multiple overnight sessions. Added resume logic to `run_inpaint`:
- Loads existing `manifest.json` at startup so completed entries survive a restart
- Skips images where all variant files already exist on disk
- Writes `manifest.json` after each image instead of once at the end, so a crash never loses more than one image's work

---

## Hardware

RTX 5070 Ti, 16GB VRAM. CUDA 12.8+ required (PyTorch for this GPU isn't on the standard index, needs `--index-url https://download.pytorch.org/whl/cu128`).

During inpainting, the observed behaviour is:
- GPU maxed out during the 30 denoising steps (transformer fully in VRAM)
- Between images: system RAM spikes to ~31GB, SSD hits 100%, GPU idle — this is the cpu offload swapping transformer weights back from pagefile into VRAM
- A possible improvement: keep transformer permanently in VRAM (`pipe.to("cuda")`), keep T5 on CPU since it only runs once per image. Would eliminate the inter-image transfer overhead, but need to verify it doesn't OOM.

---

## Current status

- normalize: done
- SAM3 detection: done, all 218 images masked
- FLUX inpainting: in progress (running overnight, resume support added)
- Dataset formatting (VQA pairs): not started
- Fine-tuning: not started
