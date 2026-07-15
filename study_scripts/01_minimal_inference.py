"""
01_minimal_inference.py  —  The absolute-minimum OpenVLA inference ("getting started").

What this proves:
  * openvla-7b loads on a single 24 GB GPU in bf16.
  * The full pipeline runs: image + instruction -> 7 action tokens -> 7-DoF action.

What this does NOT prove:
  * That the action is *good*. We feed a synthetic image, so the action is out-of-distribution
    and essentially meaningless. Meaningful behaviour is validated by the LIBERO closed-loop
    eval (see study_scripts/ and docs/03_LIBERO_EVAL.md).

This path uses `trust_remote_code=True`, so it pulls the model *code* from the HF Hub and does
NOT import the local `prismatic` package (hence no TensorFlow/dlimp needed). Flash-Attention is
optional here: we auto-fall back to PyTorch SDPA if flash_attn isn't installed.

Run (inside the isolated env):
    .venv-openvla/bin/python study_scripts/01_minimal_inference.py \
        --instruction "pick up the red block" --unnorm_key bridge_orig
"""

import argparse
import importlib.util
import time

import numpy as np
import torch
from PIL import Image


def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def make_synthetic_scene(size: int = 224, seed: int = 0) -> Image.Image:
    """A cheap, deterministic 'tabletop-ish' image so the demo is reproducible."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # gradient 'table'
    for y in range(size):
        img[y, :, 0] = 150 + int(40 * y / size)
        img[y, :, 1] = 130 + int(35 * y / size)
        img[y, :, 2] = 110 + int(30 * y / size)
    # a red 'block'
    img[120:160, 90:130] = np.array([200, 40, 40], dtype=np.uint8)
    # a little sensor noise
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--instruction", default="pick up the red block")
    ap.add_argument("--unnorm_key", default="bridge_orig")
    ap.add_argument("--out", default="study_outputs/minimal_inference.png")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    attn = "flash_attention_2" if has_flash_attn() else "sdpa"
    print(f"[*] Loading {args.model}  (attn_implementation={attn}, bf16)")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    print(f"[*] Loaded in {time.time()-t0:.1f}s. "
          f"Params ~{sum(p.numel() for p in vla.parameters())/1e9:.2f}B. "
          f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Which robots/datasets does this checkpoint know how to un-normalize for?
    keys = list(vla.norm_stats.keys())
    print(f"[*] {len(keys)} available unnorm_keys, e.g.: {keys[:8]}")
    unnorm_key = args.unnorm_key if args.unnorm_key in vla.norm_stats else keys[0]
    if unnorm_key != args.unnorm_key:
        print(f"[!] '{args.unnorm_key}' not found; using '{unnorm_key}'")

    image = make_synthetic_scene()
    image.save("study_outputs/dbg_input.png") 
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    # Warm-up + timed run
    for _ in range(1):
        _ = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.time()
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    labels = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]
    print(f"\n[RESULT] instruction={args.instruction!r}  unnorm_key={unnorm_key}")
    print(f"[RESULT] 7-DoF action = {np.array2string(action, precision=4)}")
    print(f"[RESULT] single-step latency = {dt*1000:.0f} ms  (~{1/dt:.1f} Hz)")
    for l, v in zip(labels, action):
        print(f"           {l:8s} = {v:+.4f}")

    # ---- visualization ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axi, axb) = plt.subplots(1, 2, figsize=(11, 4.2))
        axi.imshow(image)
        axi.set_title(f"input (synthetic)\n\"{args.instruction}\"", fontsize=10)
        axi.axis("off")
        colors = ["#4C78A8"] * 6 + ["#E45756"]
        axb.barh(range(7), action, color=colors)
        axb.set_yticks(range(7))
        axb.set_yticklabels(labels)
        axb.invert_yaxis()
        axb.axvline(0, color="k", lw=0.8)
        axb.set_title(f"predicted 7-DoF action  ({dt*1000:.0f} ms, attn={attn})", fontsize=10)
        axb.set_xlabel("un-normalized action value")
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"\n[*] Saved visualization -> {args.out}")
    except Exception as e:
        print(f"[!] Plot skipped: {e}")


if __name__ == "__main__":
    main()
