"""
05_lora_finetune_3090.py  —  QLoRA fine-tuning of openvla-7b that FITS on a 24 GB RTX 3090.

Purpose: demonstrate the fine-tuning *mechanics* and *memory envelope* on a single 3090:
  * 4-bit (nf4) base + LoRA adapters  => a 7.5B VLA trains in well under 24 GB.
  * the action-token-masked cross-entropy loss (identical objective to vla-scripts/finetune.py).
  * loss down / action-token accuracy up / L1 down, plus peak-VRAM reporting.
  * saves a LoRA adapter you can load back.

This is a self-contained SANITY/MEMORY demo on a tiny in-script dataset (it overfits a handful of
(image, instruction, action) samples). It is NOT a real policy. For real fine-tuning use the repo's
`vla-scripts/finetune.py` with an RLDS dataset (see docs/05_FINETUNING_3090.md), or the OFT recipe.

We intentionally avoid importing TensorFlow / the RLDS pipeline here (keeps it robust on py3.12 and
sidesteps the TF/torch CUDA-init conflict documented in docs/04_REPRO_LOG.md).

Run:
    .venv-openvla/bin/python study_scripts/05_lora_finetune_3090.py --max_steps 60 --batch_size 2
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image

# Action discretization — MUST match modeling_prismatic / ActionTokenizer so a fine-tuned adapter
# round-trips through predict_action. bins = 256 uniform edges over [-1, 1]; action tokens overwrite
# the last 256 vocab ids: token_id = vocab_size - digitize(action, bins).
BINS = np.linspace(-1.0, 1.0, 256)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0
SPACE_TOKEN = 29871  # Llama SentencePiece leading-space token that must precede action tokens
LABELS = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]


def action_to_token_ids(action, vocab_size):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    disc = np.digitize(action, BINS)                # in [1..256]
    return (vocab_size - disc).astype(np.int64)     # -> last-256 vocab ids


def token_ids_to_action(ids, vocab_size):
    disc = np.clip(vocab_size - np.asarray(ids) - 1, 0, BIN_CENTERS.shape[0] - 1)
    return BIN_CENTERS[disc]


def synth_sample(i, size=224):
    """Deterministic (image, instruction, target-action) sample. Block position encodes the target
    direction, so the target depends on the image (a mini visuomotor mapping to memorize)."""
    rng = np.random.default_rng(1000 + i)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :, :] = np.array([120, 110, 100]) + rng.integers(-5, 5, 3)
    # block position on a 3x3 grid
    gx, gy = (i % 3), ((i // 3) % 3)
    cx, cy = 45 + gx * 65, 45 + gy * 65
    color = [[220, 40, 40], [40, 200, 40], [60, 90, 230]][i % 3]
    img[cy - 18:cy + 18, cx - 18:cx + 18] = color
    image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")
    # target action: move toward block (normalized), descend, close gripper
    action = np.array([
        (cx - size / 2) / (size / 2) * 0.6,
        (cy - size / 2) / (size / 2) * 0.6,
        -0.25, 0.0, 0.0, 0.0, 1.0,
    ], dtype=np.float32)
    instr = ["red", "green", "blue"][i % 3]
    return image, f"pick up the {instr} block", action


def build_example(processor, tokenizer, image, instruction, action):
    vocab_size = tokenizer.vocab_size
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids  # [BOS, ...]
    if prompt_ids[-1] != SPACE_TOKEN:
        prompt_ids = prompt_ids + [SPACE_TOKEN]
    action_ids = action_to_token_ids(action, vocab_size).tolist()
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + action_ids + [eos]
    labels = [-100] * len(prompt_ids) + action_ids + [eos]        # supervise only action tokens (+eos)
    px = processor(prompt, image)["pixel_values"][0]              # [6,224,224]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pixel_values": torch.as_tensor(px),
    }


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn, pix = [], [], [], []
    for b in batch:
        n = len(b["input_ids"]); pad = maxlen - n
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        pix.append(b["pixel_values"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
        "pixel_values": torch.stack(pix),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accumulation_steps", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--n_samples", type=int, default=9)
    ap.add_argument("--out_dir", default="runs/lora-3090-demo")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoConfig, AutoImageProcessor, AutoModelForVision2Seq,
                              AutoProcessor, BitsAndBytesConfig)
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("[*] Loading openvla-7b in 4-bit (nf4) for QLoRA ...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path, quantization_config=bnb, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        device_map={"": 0},   # place on GPU 0 during load; 4-bit models forbid a later .to()
    )
    vla = prepare_model_for_kbit_training(vla)     # enables grad checkpointing, casts norms to fp32
    vla.config.use_cache = False
    lora = LoraConfig(r=args.lora_rank, lora_alpha=min(args.lora_rank, 16), lora_dropout=0.0,
                      target_modules="all-linear", init_lora_weights="gaussian")
    vla = get_peft_model(vla, lora)
    vla.print_trainable_parameters()
    print(f"[*] Loaded + wrapped in {time.time()-t0:.1f}s. "
          f"VRAM now {torch.cuda.memory_allocated()/1e9:.2f} GB (4-bit base + LoRA)")

    # tiny dataset
    data = [build_example(processor, tokenizer, *synth_sample(i)) for i in range(args.n_samples)]
    print(f"[*] Built {len(data)} training examples (seq len ~{len(data[0]['input_ids'])} text tokens)")

    opt = torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad], lr=args.learning_rate)
    try:
        num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches  # 256 (attr-forwarded through PEFT)
    except AttributeError:
        num_patches = 256
    action_begin = tokenizer.vocab_size - (256 + 1)
    torch.cuda.reset_peak_memory_stats()

    vla.train()
    opt.zero_grad()
    rng = np.random.default_rng(0)
    step, t_start = 0, time.time()
    print(f"\n{'step':>4} {'loss':>8} {'act_acc':>8} {'L1':>8} {'peakVRAM_GB':>12}")
    while step < args.max_steps:
        idx = rng.choice(len(data), size=args.batch_size, replace=len(data) < args.batch_size)
        batch = collate([data[i] for i in idx], tokenizer.pad_token_id or 32000)
        dev = "cuda:0"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=batch["input_ids"].to(dev),
                      attention_mask=batch["attention_mask"].to(dev),
                      pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev),
                      labels=batch["labels"].to(dev))
            loss = out.loss
        (loss / args.grad_accumulation_steps).backward()

        # metrics (mirror vla-scripts/finetune.py): logits offset by num_patches, labels shifted by 1
        with torch.no_grad():
            action_logits = out.logits[:, num_patches:-1]
            preds = action_logits.argmax(-1)
            gt = batch["labels"][:, 1:].to(preds.device)
            mask = gt > action_begin
            acc = ((preds == gt) & mask).sum().float() / mask.sum().clamp(min=1)
            pa = token_ids_to_action(preds[mask].cpu().numpy(), tokenizer.vocab_size)
            ga = token_ids_to_action(gt[mask].cpu().numpy(), tokenizer.vocab_size)
            l1 = np.abs(pa - ga).mean() if mask.sum() > 0 else float("nan")

        if (step + 1) % args.grad_accumulation_steps == 0:
            opt.step(); opt.zero_grad()
        if step % 5 == 0 or step == args.max_steps - 1:
            print(f"{step:>4} {loss.item():>8.4f} {acc.item():>8.3f} {l1:>8.4f} "
                  f"{torch.cuda.max_memory_allocated()/1e9:>12.2f}")
        step += 1

    dt = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[RESULT] {args.max_steps} steps in {dt:.1f}s ({dt/args.max_steps:.2f}s/step). "
          f"PEAK VRAM = {peak:.2f} GB / 24 GB  ({'FITS' if peak < 24 else 'OOM RISK'})")
    print(f"[RESULT] final loss={loss.item():.4f} action_acc={acc.item():.3f} L1={l1:.4f}")

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    vla.save_pretrained(args.out_dir)
    print(f"[*] Saved LoRA adapter -> {args.out_dir}  (load with peft.PeftModel.from_pretrained)")


if __name__ == "__main__":
    main()
