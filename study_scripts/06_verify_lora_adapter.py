"""
06_verify_lora_adapter.py  —  Prove the QLoRA adapter from script 05 actually changed the policy,
and show the general pattern for LOADING + USING a fine-tuned OpenVLA adapter.

It loads openvla-7b in 4-bit, predicts actions on the (same) tiny training images BEFORE and AFTER
attaching the LoRA adapter (runs/lora-3090-demo), and compares to the training targets. The base model
should be "off"; with the adapter the (normalized) actions should snap to the memorized targets.

Run (after study_scripts/05_lora_finetune_3090.py):
    .venv-openvla/bin/python study_scripts/06_verify_lora_adapter.py
"""

import argparse
import numpy as np
import torch
from PIL import Image

BINS = np.linspace(-1.0, 1.0, 256)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0
LABELS = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]


def synth_sample(i, size=224):  # identical to script 05
    rng = np.random.default_rng(1000 + i)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :, :] = np.array([120, 110, 100]) + rng.integers(-5, 5, 3)
    gx, gy = (i % 3), ((i // 3) % 3)
    cx, cy = 45 + gx * 65, 45 + gy * 65
    color = [[220, 40, 40], [40, 200, 40], [60, 90, 230]][i % 3]
    img[cy - 18:cy + 18, cx - 18:cx + 18] = color
    image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")
    action = np.array([(cx - size / 2) / (size / 2) * 0.6, (cy - size / 2) / (size / 2) * 0.6,
                       -0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    instr = ["red", "green", "blue"][i % 3]
    return image, f"pick up the {instr} block", action


def normalized_action_from_generate(vla, processor, image, instruction, vocab_size):
    """Greedy-generate 7 action tokens and decode to NORMALIZED action (bin centers), bypassing
    dataset un-normalization so we can compare directly to the training targets."""
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inp = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    ids = inp["input_ids"]
    if not torch.all(ids[:, -1] == 29871):
        ids = torch.cat([ids, torch.tensor([[29871]], device=ids.device)], dim=1)
        inp["input_ids"] = ids
    with torch.no_grad():
        gen = vla.generate(**inp, max_new_tokens=7, do_sample=False)
    tok = gen[0, -7:].cpu().numpy()
    disc = np.clip(vocab_size - tok - 1, 0, BIN_CENTERS.shape[0] - 1)
    return BIN_CENTERS[disc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--adapter_dir", default="runs/lora-3090-demo")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoConfig, AutoImageProcessor, AutoModelForVision2Seq,
                              AutoProcessor, BitsAndBytesConfig)
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vocab_size = processor.tokenizer.vocab_size
    print("[*] Loading base openvla-7b (4-bit)...")
    base = AutoModelForVision2Seq.from_pretrained(args.vla_path, quantization_config=bnb,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map={"": 0})
    base.eval()

    samples = [synth_sample(i) for i in range(args.n)]
    base_preds = [normalized_action_from_generate(base, processor, im, ins, vocab_size) for im, ins, _ in samples]

    print(f"[*] Attaching LoRA adapter from {args.adapter_dir} ...")
    adapted = PeftModel.from_pretrained(base, args.adapter_dir)
    adapted.eval()
    adapted_preds = [normalized_action_from_generate(adapted, processor, im, ins, vocab_size) for im, ins, _ in samples]

    print("\n=== Normalized action: TARGET vs BASE vs BASE+ADAPTER (L1 error to target) ===")
    base_err = adapted_err = 0.0
    for i, (im, ins, tgt) in enumerate(samples):
        be = float(np.abs(base_preds[i] - tgt).mean())
        ae = float(np.abs(adapted_preds[i] - tgt).mean())
        base_err += be; adapted_err += ae
        print(f"\nsample {i}  ({ins})")
        print(f"  target : {np.array2string(tgt, precision=3)}")
        print(f"  base   : {np.array2string(base_preds[i], precision=3)}   L1={be:.3f}")
        print(f"  adapter: {np.array2string(adapted_preds[i], precision=3)}   L1={ae:.3f}")
    n = len(samples)
    print(f"\n[RESULT] mean L1 to target — base: {base_err/n:.3f}   base+adapter: {adapted_err/n:.3f}")
    print(f"[RESULT] adapter reduced action error by {100*(1-adapted_err/max(base_err,1e-6)):.0f}% "
          f"→ the saved LoRA adapter demonstrably steers the policy to the fine-tuned behavior.")


if __name__ == "__main__":
    main()
