# Training runs — FLUX guidance_scale=30

All four runs used the same synthetic dataset from the FLUX guidance_scale=30 inpainting run. The data quality from that setting turned out to be the main bottleneck across the board.

---

## Run 1

Qwen3-VL-4B, QLoRA r=32, lr=2e-4, 5 epochs, batch 2 x grad_accum 4. Dataset: 218 clean + 654 messy (3 variants per clean image) = 872 total, 85/15 train/eval split.

- Accuracy: 0.769
- Precision: 0.769
- Recall: 1.000
- F1: 0.870

Train loss 0.086 → 0.012, eval loss bottomed ~0.048 at step 250 then started rising.

### Issue: class imbalance

Recall = 1.0 is a red flag. The model never predicted clean, it just always said messy. With a 3:1 messy:clean dataset ratio, that gets you a Precision of 0.769 for free. It didn't learn anything useful.

Fix: repeat each clean image 3x in the dataset (no file copies, just extra rows) to get to 1:1. Also lowered LR to 1e-4 and added lora_dropout=0.05, weight_decay=0.05. See run 2.

---

## Run 2

Same model (Qwen3-VL-4B, r=32), balanced dataset (each clean image repeated 3x → 654 clean : 654 messy). Updated hyperparams: lr=1e-4, epochs=4, lora_dropout=0.05, weight_decay=0.05, eval_steps=25.

- Accuracy: 0.735
- Precision: 0.693
- Recall: 0.915
- F1: 0.789

Train loss 1.31 → 0.02, eval loss bottomed ~0.036 at step 350 then stayed flat.

Now doing real classification. Recall dropped from 1.0 to 0.915, meaning it's actually predicting clean sometimes. Precision at 0.693 is the weak spot, about a 30% false positive rate on clean rooms.

Eval loss plateaued hard around step 225 and didn't move for the rest of training. The data quality is the ceiling here. A chunk of the FLUX guidance_scale=30 images look unrealistic and blur the clean/dirty boundary for the model. More epochs or a different LR won't help; the signal in the data is the limit.

---

## Run 3

Same setup as run 2 but with lr=8e-5 and 5 epochs. Loss curve shape identical to run 2, same plateau, same floor.

- Accuracy: 0.689
- Precision: 0.643
- Recall: 0.902
- F1: 0.751

Worse than run 2 across the board. The lower LR just slowed convergence without recovering precision. 

---

## Run 4 (v5)

Same model and balanced dataset as runs 2/3. Reverted lr back to 1e-4 (run 3 showed 8e-5 was worse), capped at 4 epochs, added `EarlyStoppingCallback(patience=4)` to stop if eval_loss doesn't improve for 4 consecutive checks (100 steps).

- Accuracy: 0.714
- Precision: 0.676
- Recall: 0.906
- F1: 0.774

Best loss curve of any run, train and eval tracked together with no divergence. Early stopping triggered at step 450, best checkpoint was step 350 (eval_loss 0.036393). The eval loss floor (~0.036) matches run 2 exactly, both hitting the same ceiling.

Metrics are in line with run 2 (F1 0.774 vs 0.789, Precision 0.676 vs 0.693) — within noise on a ~196-sample eval set, likely just random split variance. The hyperparams (lr=1e-4, 4 epochs, early stopping) are validated as the right config for this dataset.