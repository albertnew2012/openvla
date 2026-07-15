"""
03_action_tokenization_explainer.py  —  Make the VLA-specific 70 lines concrete.

Shows, for ONE real prediction, the full chain:
    logits over 32064-vocab  ->  restrict to the 256 action bins  ->  softmax
    ->  argmax token id  ->  bin index  ->  bin center in [-1,1]  ->  un-normalized action

and plots the model's discrete action *distribution* over the 256 bins for each of the 7 DoF.
This is the discrete analogue of what a diffusion policy represents continuously.

Run:
    .venv-openvla/bin/python study_scripts/03_action_tokenization_explainer.py
"""

import argparse
import importlib.util

import numpy as np
import torch
from PIL import Image


def has_flash_attn():
    return importlib.util.find_spec("flash_attn") is not None


def make_synthetic_scene(size=224, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        img[y, :] = [150 + 40 * y // size, 130 + 35 * y // size, 110 + 30 * y // size]
    img[120:160, 90:130] = [200, 40, 40]
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--instruction", default="pick up the red block")
    ap.add_argument("--unnorm_key", default="bridge_orig")
    ap.add_argument("--out", default="study_outputs/action_tokenization.png")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    # --- LOCAL path: register OpenVLA's classes from THIS repo (prismatic/extern/hf) with the HF
    #     Auto machinery, so we can load WITHOUT trust_remote_code. Nothing is downloaded from the Hub;
    #     the code that runs is your local prismatic/extern/hf/*.py (byte-identical to the Hub copy). ---
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    attn = "flash_attention_2" if has_flash_attn() else "sdpa"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model, attn_implementation=attn, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=False,
    ).to("cuda:0").eval()

    unnorm_key = args.unnorm_key if args.unnorm_key in vla.norm_stats else list(vla.norm_stats)[0]
    action_dim = vla.get_action_dim(unnorm_key)   # 7
    stats = vla.get_action_stats(unnorm_key)
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(q01, dtype=bool)))

    image = make_synthetic_scene()
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    # Mirror predict_action's prompt fix: ensure trailing space token 29871
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], device=input_ids.device)], dim=1)
        inputs["input_ids"] = input_ids

    # Generate WITH scores so we can inspect the per-step distribution.
    with torch.no_grad():
        gen = vla.generate(
            **inputs, max_new_tokens=action_dim, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    gen_ids = gen.sequences[0, -action_dim:].cpu().numpy()

    # ---- de-tokenize exactly like modeling_prismatic.OpenVLAForActionPrediction.predict_action ----
    V = vla.vocab_size                      # 32000 (text vocab minus pad_to_multiple_of)
    n_bins = vla.bin_centers.shape[0]       # 255 centers (256 edges)
    disc = V - gen_ids                      # token id -> bin index
    disc = np.clip(disc - 1, 0, n_bins - 1)
    normalized = vla.bin_centers[disc]      # in [-1, 1]
    action = np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)

    labels = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"][:action_dim]
    print(f"\nmodel={args.model}  unnorm_key={unnorm_key}  action_dim={action_dim}  vocab_size={V}")
    print(f"{'dim':8s} {'token_id':>9s} {'bin_idx':>8s} {'norm[-1,1]':>11s} {'q01':>9s} {'q99':>9s} {'ACTION':>10s}")
    for i, l in enumerate(labels):
        print(f"{l:8s} {gen_ids[i]:9d} {disc[i]:8d} {normalized[i]:+11.4f} "
              f"{q01[i]:+9.4f} {q99[i]:+9.4f} {action[i]:+10.4f}")

    # ---- per-DoF distribution over the 256 action bins ----
    # For step t, the logits row over the vocab; the action bins are the top-256 token ids,
    # i.e. ids [V-256 .. V-1] map to bins [255..0]. Softmax over just those ids.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_ids = np.arange(V - 256, V)  # the 256 token ids reserved for action bins
    fig, axes = plt.subplots(action_dim, 1, figsize=(9, 1.5 * action_dim), sharex=True)
    for i, (ax, l) in enumerate(zip(np.atleast_1d(axes), labels)):
        logits = gen.scores[i][0].float().cpu().numpy()      # [vocab]
        sub = logits[bin_ids]
        p = np.exp(sub - sub.max()); p /= p.sum()
        # bin index for token id (V - id - 1); reverse so x-axis is bin value ascending
        bin_index_for_ids = np.clip((V - bin_ids) - 1, 0, n_bins - 1)
        order = np.argsort(bin_index_for_ids)
        xs = vla.bin_centers[bin_index_for_ids[order]]
        ax.fill_between(xs, p[order], step="mid", alpha=0.7, color="#4C78A8")
        ax.axvline(normalized[i], color="#E45756", lw=1.5)
        ax.set_ylabel(l, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
    np.atleast_1d(axes)[-1].set_xlabel("normalized action bin value in [-1, 1]  (red = argmax)")
    np.atleast_1d(axes)[0].set_title(f"OpenVLA discrete action distribution (256 bins/DoF)\n{args.instruction!r}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n[*] Saved per-DoF bin distributions -> {args.out}")


if __name__ == "__main__":
    main()
