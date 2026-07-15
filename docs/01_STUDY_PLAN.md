# OpenVLA / VLA Study Plan — Built on Your DETR · PETR · VLM · LLM Background

> You already have the hard prerequisites: transformers, attention, ViT backbones,
> query-based decoders (DETR/PETR), multimodal fusion (Qwen-VL), and autoregressive LLMs.
> This plan spends **zero** time re-teaching those and instead maps them onto the ~15% that
> is genuinely new: **turning perception into control via action tokenization, closed-loop
> evaluation, and robot data pipelines.**
>
> Everything here is runnable. Commands live in `.vscode/launch.json` (VS Code "Run and Debug")
> and as plain scripts in `study_scripts/`. Reproduction results are logged in `docs/04_REPRO_LOG.md`.

---

## 0. What already transfers (don't re-study these)

| You know (from…) | In OpenVLA it's… | Net new to learn |
|---|---|---|
| ViT backbones, SigLIP/CLIP (VLM) | DINOv2+SigLIP **fused** backbone | the *fusion by channel-stacking* trick, penultimate-layer features |
| Cross-attention decoder w/ queries (DETR/PETR) | **gone** — replaced by decoder-only self-attn over `[BOS,vision,text]` | why VLAs drop queries; prefix-fusion vs cross-attn |
| VLM visual-token prepending (Qwen-VL) | visual tokens inserted after BOS | almost identical — you already get this |
| Autoregressive next-token decoding (LLM) | action tokens decoded the same way | **action tokenization** (256 bins) + **un-normalization** |
| Box regression + Hungarian loss (DETR) | *no regression head*; CE over discrete bins | discretization trade-offs vs regression/diffusion |
| 3D position priors (PETR) | **no geometric prior** | why this limits VLAs; what OFT/chunking add |

**Mental one-liner:** OpenVLA = *(Qwen-VL-style VLM) + (action-as-language head) + (robot data)*.
If you can read `modeling_llava.py`, you can read all of OpenVLA in an afternoon.

## The 15% that is actually new (your real syllabus)
1. **Action representation** — how a continuous 7-DoF control becomes 7 discrete tokens, and back.
2. **Normalization per embodiment** — `unnorm_key`, q01/q99, gripper handling.
3. **Closed-loop evaluation** — success rate over rollouts, not a static metric like mAP.
4. **Robot data (RLDS / Open X-Embodiment)** — trajectories, not image-label pairs.
5. **The frontier** — action chunking, FAST tokenizer, OFT, diffusion/flow policies, π0, RT-2.

---

## 1. Phased plan (each phase = read → run → verify-you-understand)

### Phase 0 — Environment & first inference ✅ (already done for you)
- Read: `docs/04_REPRO_LOG.md` (how the isolated env was built and *why* — the version pinning
  story is itself a great lesson in how fragile VLA repos are).
- Run: `study_scripts/01_minimal_inference.py` → VS Code config **"01 · Minimal inference (openvla-7b)"**.
- **Checkpoint (understanding):** explain why loading needs `timm==0.9.10` exactly, and why the
  base model has 25 `unnorm_key`s. (Answer in `docs/02_ARCHITECTURE.md` §3 and §7.)

### Phase 1 — The architecture, end to end (½ day)
- Read: `docs/02_ARCHITECTURE.md` in full, then the actual code in this order:
  1. `prismatic/extern/hf/configuration_prismatic.py`
  2. `prismatic/extern/hf/modeling_prismatic.py:63-158` (backbone + projector)
  3. `:291-447` (the 3-regime forward + `[BOS,vision,text]` splice)
- **Checkpoint:** draw the tensor-shape diagram from memory (image → 256×2176 → 256×4096 →
  Llama → logits). Compare to how PETR routes features to queries — note that OpenVLA has *no*
  query/cross-attn stage at all.

### Phase 2 — Action tokenization (the crux) (½ day)
- Read: `modeling_prismatic.py:492-562` — `predict_action`, `get_action_dim`, `get_action_stats`.
- Run: `study_scripts/03_action_tokenization_explainer.py` → config **"03 · Action tokenization explainer"**.
  It prints the exact token-id ↔ bin ↔ physical-action mapping for a real prediction, and plots
  the 256-bin logit distribution per action dim (multi-modality you'd otherwise get "for free"
  from a diffusion head).
- **Checkpoint:** given a predicted token id `31900`, hand-compute the bin index and the
  normalized action, then the un-normalized value using a dataset's q01/q99. Verify against the script.

### Phase 3 — Closed-loop evaluation in LIBERO (1 day)
- Read: `experiments/robot/libero/run_libero_eval.py` (the control loop), `docs/03_LIBERO_EVAL.md`.
- Run: **"04 · LIBERO eval (spatial, quick subset)"** then watch the rollout MP4s under `rollouts/`.
  Compare your subset success rate to the paper's 84.7% (LIBERO-Spatial).
- **Checkpoint:** articulate why control needs *closed-loop* eval (compounding errors, covariate
  shift) whereas detection is evaluated open-loop on a fixed set. This is the single biggest
  methodological difference from your DETR/PETR world.
- **Failure-mode study:** find one failed rollout video and one success; describe where the policy
  diverged. (VLAs often fail by "getting stuck" — see the README troubleshooting section.)

### Phase 4 — How VLAs are *trained*: the data pipeline (½ day, reading-heavy)
- Read: `prismatic/vla/datasets/rlds/` (skim), `prismatic/vla/datasets/rlds/oxe/mixtures.py`
  (the "Open-X Magic Soup++" mixture), `transforms.py` (per-dataset standardization to a common
  7-DoF action). This is the robotics analogue of "dataset harmonization" — every robot speaks a
  different action dialect and gets normalized to one.
- **Checkpoint:** explain what an RLDS "trajectory" contains vs an image-caption pair, and why
  action normalization stats are *per dataset*.
- (Optional, heavy) the TF/RLDS stack is finicky on Python 3.12 — see `docs/04_REPRO_LOG.md`
  "Why we skipped TensorFlow for inference." You don't need it unless you fine-tune via RLDS.

### Phase 5 — Fine-tuning on your RTX 3090 (24 GB) (1 day)
- Read: `docs/05_FINETUNING_3090.md` (memory math, LoRA + 4-bit, batch/accum settings that fit 24 GB).
- Run: **"05 · LoRA fine-tune (LIBERO, 3090-sized)"** — a short LoRA run to see the training loop,
  `action_accuracy`/L1 logging, and adapter checkpointing. Full convergence isn't the goal; the
  *loop and the memory envelope* are.
- **Checkpoint:** compute the parameter/optimizer/activation memory budget for full FT vs LoRA vs
  4-bit-LoRA and explain why only the last fits on 24 GB. (Worked example in the doc.)

### Phase 6 — The frontier (ongoing)
Read these project pages/papers and place each on the "action representation" axis:
- **RT-1 / RT-2** (Google) — the original "actions as tokens" lineage OpenVLA descends from.
- **Octo** — transformer policy with a diffusion action head (contrast: continuous vs discrete).
- **OpenVLA-OFT** (`openvla-oft.github.io`) — parallel decoding + continuous actions + action
  chunking → 25–50× faster, higher success. *This is what you'd actually deploy today.*
- **FAST** (Physical Intelligence) — DCT-based action tokenizer, compresses chunks → up to 15× fewer
  tokens for discrete VLAs.
- **π0 / π0-FAST** (Physical Intelligence) — flow-matching VLA, current SOTA-ish generalist.
- **Diffusion Policy / ACT** — the non-VLA baselines LIBERO compares against; understand why
  chunked continuous policies are so strong on manipulation.
- **Checkpoint:** one paragraph — "If I were building a bimanual real-robot policy in 2026, would I
  use vanilla OpenVLA, OFT, or π0, and why?" (Hint: inference frequency + action chunking dominate.)

---

## 2. Reading list, ordered, with the "why"

1. **OpenVLA paper** (arXiv:2406.09246) — read §3 (model) and Appendix E (LIBERO). Skim the OXE mixture.
2. **Prismatic VLMs** (arXiv:2402.07865) — the VLM OpenVLA is built on; explains the fused
   DINOv2+SigLIP choice and the penultimate-feature finding. *Directly in your VLM comfort zone.*
3. **RT-2** — "actions as another language"; the conceptual parent.
4. **Octo** — diffusion action head; your first "continuous action" contrast.
5. **OpenVLA-OFT** — the practical upgrade; parallel decoding + chunking.
6. **π0** — flow matching for control; where the field is heading.
7. (Background, if robotics is new) a 30-min primer on **end-effector pose control** and
   **quaternions/axis-angle** — enough to know what the 7 numbers mean. See `docs/02_ARCHITECTURE.md`
   appendix glossary.

---

## 3. Concept bridges (use what you know)

- **DETR queries → ∅.** OpenVLA has no learned queries; the "slots" that hold task state are just
  the autoregressive positions of the 7 action tokens. If that feels too weak, you've already
  intuited *why action chunking / OFT exist*.
- **PETR 3D PE → nothing.** OpenVLA gets no explicit 3D/camera-geometry prior. Manipulation works
  anyway because the demos teach an image→action mapping, but it's why single-image VLAs struggle
  with precise depth and why multi-view/proprio inputs (OFT) help.
- **Qwen-VL visual tokens → identical mechanism.** The only twist is the *output* vocabulary is
  hijacked for actions and the loss is masked to action tokens.
- **LLM sampling → policy stochasticity.** `do_sample=False` (greedy) is the deterministic policy;
  temperature sampling gives you a stochastic policy. The 256-way softmax per dim is your
  (discretized) action distribution — the discrete cousin of a diffusion policy's continuous one.

---

## 4. "Do I actually get it?" self-test (answer without looking)
1. Why is `pixel_values` shaped `[B, 6, 224, 224]` and not `[B, 3, …]`?
2. Where in the sequence do the 256 visual tokens go, and why are their labels `-100`?
3. Token id → action: what are the exact 3 steps `predict_action` does?
4. What does `unnorm_key` select, and what breaks if it's wrong?
5. Why is LIBERO eval closed-loop, and why does `center_crop=True` matter at test time?
6. On a 24 GB GPU, which of {full FT, LoRA, 4-bit LoRA} fit, and which single hyperparameter
   most directly trades memory for it?
7. Name one concrete limitation of 256-bin discrete actions and the technique that fixes it.

(Answers are distributed across `docs/02_ARCHITECTURE.md`, `03_LIBERO_EVAL.md`, `05_FINETUNING_3090.md`.)

---

## 5. Where things are
- `docs/00_START_HERE.md` — index + what was done, read this first.
- `docs/02_ARCHITECTURE.md` — model deep-dive (this is your main text).
- `docs/03_LIBERO_EVAL.md` — closed-loop eval explained + how to run/scale it.
- `docs/04_REPRO_LOG.md` — exactly how the env was built and every issue+fix (great debugging study).
- `docs/05_FINETUNING_3090.md` — memory math + LoRA recipe for 24 GB.
- `study_scripts/` — small, commented, runnable scripts for each phase.
- `.vscode/launch.json` — every command as a one-click "Run and Debug" config.
- `study_outputs/`, `rollouts/` — generated figures and rollout videos.
