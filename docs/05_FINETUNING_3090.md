# Fine-Tuning OpenVLA on a Single RTX 3090 (24 GB)

> Short version: **full fine-tuning is impossible on 24 GB; plain LoRA barely fits; QLoRA (4-bit base
> + LoRA) fits comfortably.** This doc gives the memory math, the exact recipe, and a runnable demo.

---

## 1. Why you can't just fine-tune a 7.5B model on 24 GB (the memory math)

Training memory ≈ **weights + gradients + optimizer states + activations**. For OpenVLA (~7.5B params):

| Scheme | Weights | Grads | Optimizer (AdamW = 2 fp32 moments) | Rough total (ex-activations) | Fits 24 GB? |
|---|---|---|---|---|---|
| **Full FT (bf16)** | 15 GB | 15 GB | 2 × 4 B × 7.5B = **60 GB** | **~90 GB** | ❌ (needs ~8×A100) |
| **LoRA (bf16 base)** | 15 GB (frozen) | ~0.2 GB | ~0.5 GB (only adapters) | **~16 GB + activations** | ⚠️ tight (README quotes ~27 GB @ bs16) |
| **QLoRA (4-bit base + LoRA)** | **~5 GB** | ~0.2 GB | ~0.5 GB | **~6 GB + activations** | ✅ comfortable |

Key insights (these generalize to any LLM/VLM fine-tuning, so they connect to what you know):
- **Optimizer state dominates full FT.** AdamW keeps two fp32 moments per trainable parameter → 8 bytes ×
  params. That alone is 60 GB here. LoRA slashes this because only the tiny adapter params are trainable.
- **LoRA freezes the base**, so the base contributes only its (inference) weight memory — no grads, no
  optimizer states. But in bf16 that's still 15 GB, leaving little headroom for activations on 24 GB.
- **QLoRA quantizes the frozen base to 4-bit (nf4)** → ~5 GB. Now activations + adapter training fit
  easily. Compute still happens in bf16 (`bnb_4bit_compute_dtype`), and LoRA adapters stay bf16/fp32.
- **The one knob that most directly trades memory ↔ throughput is `batch_size`** (activation memory is
  ~linear in it). Use `grad_accumulation_steps` to keep an effective batch large while `batch_size=1–2`.
- **Gradient checkpointing** (enabled by `prepare_model_for_kbit_training`) trades compute for a big
  activation-memory saving — essential on 24 GB.

---

## 2. The 3090 recipe (with the repo's script)

`vla-scripts/finetune.py` supports exactly this via `--use_quantization True` (4-bit) + `--use_lora True`:

```bash
# Effective batch 8 (=1×8) on a single 3090. Requires an RLDS dataset (see §3).
MUJOCO_GL=egl .venv-openvla/bin/python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir <RLDS_DATA_DIR> \
  --dataset_name <DATASET> \
  --use_lora True --use_quantization True \
  --lora_rank 32 --batch_size 1 --grad_accumulation_steps 8 \
  --learning_rate 5e-4 --image_aug True \
  --save_steps 1000
```

Notes:
- `--use_quantization True` is the 24 GB enabler. The repo warns it "reduces memory but hurts
  performance" — for a 3090 that's the trade you make; for best quality you'd rent an A100 and drop it.
- Keep `--image_aug True` (random crops) and remember to eval with `--center_crop True`.
- `lora_alpha` is set to `min(rank, 16)` and `target_modules="all-linear"` in the repo (LoRA on every
  linear, incl. the vision backbone + projector).
- Constant LR (no schedule) is the repo default; 5e-4 works well for LoRA.
- It saves a LoRA adapter, then merges it into a full HF checkpoint you can load with AutoClasses and
  eval in LIBERO exactly like the released `openvla-7b-finetuned-libero-*` checkpoints.

---

## 3. Getting fine-tuning data (the honest caveat on this box)

`finetune.py` uses the **RLDS/TFDS** dataloader. On this Python-3.12 container that stack is broken
(protobuf/tfds incompat — see `docs/04_REPRO_LOG.md` issue d). Two ways forward:

1. **Real RLDS fine-tuning** (recommended for actual results): do it in a Python-3.10 env where
   `tensorflow==2.15.0` + `tensorflow_datasets==4.9.3` install cleanly, then download a dataset:
   - LIBERO (modified, ~10 GB): `git clone git@hf.co:datasets/openvla/modified_libero_rlds`
   - or an OXE dataset via `rlds_dataset_mod/prepare_open_x.sh`.
2. **Custom PyTorch dataset** (no TF): `finetune.py` has a commented `DummyDataset` hook
   (`vla-scripts/finetune.py:200-208`). Swap in a `torch.utils.data.Dataset` that yields
   `{pixel_values, input_ids, labels}` and add an epoch loop. This is what our standalone demo does.

---

## 4. Runnable demo on this machine (mechanics + memory, no data download)

`study_scripts/05_lora_finetune_3090.py` is a self-contained QLoRA loop (VS Code config
"06 · LoRA fine-tune"). It overfits a tiny in-script set of (image, instruction, action) samples to
demonstrate: 4-bit+LoRA load, the **action-token-masked CE loss** (identical objective to the repo),
`action_accuracy`/`L1` climbing, adapter saving, and **peak VRAM**. It deliberately avoids TF/RLDS.

```bash
.venv-openvla/bin/python study_scripts/05_lora_finetune_3090.py --max_steps 60 --batch_size 2
```

What to watch:
- `trainable params` printed by `print_trainable_parameters()` — a tiny fraction of 7.5B (that's LoRA).
- `loss` ↓, `act_acc` ↑ toward ~1.0 (it's memorizing), `L1` ↓ — the loop works and gradients flow.
- `PEAK VRAM` — reported at the end; on a 3090 this stays far under 24 GB.

**Measured on this RTX 3090** (`--max_steps 60 --batch_size 2 --grad_accumulation_steps 4`, LoRA r=16):

| Metric | Value |
|---|---|
| Trainable params | **55.4 M / 7.60 B = 0.73%** (LoRA on all-linear) |
| VRAM after 4-bit load + LoRA wrap | 5.14 GB |
| **Peak VRAM during training** | **9.63 GB / 24 GB → FITS with ~14 GB to spare** |
| Throughput | ~1.0 s/step (bf16 compute, grad-checkpointed) |
| Loss (step 0 → 59) | **7.40 → 0.0010** |
| Action-token accuracy | **0.21 → 1.000** |
| L1 (continuous) | 0.25 → 0.000 |

The ~14 GB of headroom means you can comfortably push `batch_size` up (e.g., 4–8) or raise `lora_rank`
to 32 and still fit — the memory story on a 3090 is easy once you're in 4-bit. Loss →0 / acc →1.0 on a
tiny set just confirms gradients flow through LoRA and the objective is wired correctly.

**Round-trip proof the adapter is real** (`study_scripts/06_verify_lora_adapter.py`): load the base
4-bit model, then attach the *saved* adapter, and predict on the training images. Mean L1 (normalized
action) to the target:

| Model | Mean L1 to target |
|---|---|
| base openvla-7b (4-bit) | 0.276 (off — never saw this task) |
| **base + saved LoRA adapter** | **0.001** (snaps to the memorized target) |

e.g. "red block" → target `[-0.359,-0.359,-0.25,0,0,0,1.0]`, adapter output
`[-0.361,-0.361,-0.251,0,0,0,0.996]`. So the fine-tune genuinely changed the policy and the saved
adapter loads + works — this is exactly the load pattern (`PeftModel.from_pretrained(base, adapter_dir)`)
you'd use for a real fine-tuned checkpoint.

This is a *sanity/memory* demo, **not** a policy. It tells you "fine-tuning will run on my 3090 and here's
the memory headroom," which is exactly what you need before committing to a real RLDS run.

---

## 5. After fine-tuning: sanity checks & the frontier

- **Sanity (from the README):** (1) replay a demo's actions in the env → should succeed (data/env OK);
  (2) feed training images into your inference pipeline → reproduce training token accuracy (inference OK).
  Only then trust/blame the policy.
- **Data quality > model tweaks:** ~100 clean demos at 5–10 Hz control, continuous motion, no idle
  pauses, diverse initial conditions (README "VLA Performance Troubleshooting").
- **Inference speed:** vanilla OpenVLA emits one token per action-dim autoregressively (~4 Hz here on
  SDPA). For real robots prefer **OpenVLA-OFT** (parallel decoding + action chunking, 25–50× faster) or
  **FAST** (compressed action tokens). See `docs/01_STUDY_PLAN.md` §Frontier.

---

## 6. Quick reference — what fits on 24 GB

| Want | Do |
|---|---|
| Just run inference | bf16, ~15 GB, no training. ✅ |
| Fine-tune on 3090 | **QLoRA**: `--use_quantization True --use_lora True --batch_size 1 --grad_accumulation_steps 8+`. ✅ |
| Best-quality fine-tune | Plain LoRA or full FT on A100(s); quantization slightly hurts quality. |
| Fastest deployable policy | Switch to OFT (different recipe/repo). |
