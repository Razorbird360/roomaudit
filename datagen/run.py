# Entry point: runs SAM3 detection then FLUX inpainting, saves results to data/messy/
# Run from project root: python -m datagen.run [--test]

import argparse
import random
import time
from pathlib import Path
from .detect import detect_all_images, save_detections, load_detections
from .inpaint import (
    load_flux_pipeline, run_inpaint,
    FLUX_STEPS, FLUX_STEPS_TEST,
    NUM_VARIANTS, MAX_DEFECTS, TEST_NUM_VARIANTS, TEST_MAX_DEFECTS,
)

ROOT      = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
MASKS_DIR = ROOT / "data" / "masks"
MESSY_DIR = ROOT / "data" / "messy"


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Generate defect variants of clean hotel room images.")
    parser.add_argument("--test", action="store_true", help="Process a random sample of 5 images.")
    args = parser.parse_args()

    if not CLEAN_DIR.is_dir():
        raise SystemExit(f"Directory not found: {CLEAN_DIR}")

    image_paths = sorted(CLEAN_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No .jpg images found in {CLEAN_DIR}")

    if args.test:
        image_paths = random.sample(image_paths, min(5, len(image_paths)))
        print(f"--test mode: {len(image_paths)} images, {TEST_NUM_VARIANTS} variant(s), {TEST_MAX_DEFECTS} defect(s), {FLUX_STEPS_TEST} steps.")

    num_variants = TEST_NUM_VARIANTS if args.test else NUM_VARIANTS
    max_defects  = TEST_MAX_DEFECTS  if args.test else MAX_DEFECTS
    flux_steps   = FLUX_STEPS_TEST   if args.test else FLUX_STEPS

    # Phase 1 — detect objects with SAM3 and persist masks to disk
    all_detections = detect_all_images(image_paths)
    save_detections(all_detections, MASKS_DIR)

    # Phase 2 — load FLUX and inpaint variants from saved masks
    print("Loading FLUX.1 Fill ...")
    pipe = load_flux_pipeline()
    print("FLUX loaded.\n")

    all_detections = load_detections(MASKS_DIR, image_paths)
    run_inpaint(pipe, all_detections, image_paths, MESSY_DIR, num_variants, max_defects, flux_steps)

    elapsed = int(time.time() - start)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}min")
    parts.append(f"{s}s")
    print(f"Done in {' '.join(parts)}.")


if __name__ == "__main__":
    main()
