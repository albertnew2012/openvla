# The VLA Landscape — Where OpenVLA Sits (and What Came After)

> For someone who knows DETR/PETR (query-based detection), Qwen-VL (VLM), and LLMs. This positions
> OpenVLA among the major robot-policy families along the axes that actually differentiate them.
> The single most important axis is **how actions are represented** — everything else follows from it.

---

## The one axis that organizes the field: action representation

| Representation | How it works | Pros | Cons | Examples |
|---|---|---|---|---|
| **Discrete tokens** | bin each action dim → predict tokens with an LM head + CE loss | reuses the entire LLM/VLM stack; multimodal "for free" via softmax; simple | quantization ceiling; 1 token/dim/step → slow; no smoothness prior | RT-1, RT-2, **OpenVLA**, FAST (compressed) |
| **Continuous regression** | MLP head predicts action(s) directly; L1/MSE loss | fast (one forward), smooth | unimodal (can't represent "go left OR right"); needs chunking to shine | **OpenVLA-OFT**, ACT (+CVAE) |
| **Diffusion / flow** | iteratively denoise an action chunk conditioned on obs | multimodal, high-quality, great for manipulation | multiple denoising steps; more complex training | Diffusion Policy, Octo (head), **π0** (flow), GR00T (head) |

OpenVLA is the **discrete-token** flagship. Its successors mostly move *off* discretization (OFT →
continuous; π0/GR00T → flow/diffusion) or *compress* it (FAST) — because one-token-per-dim autoregression
is the throughput bottleneck (~4 Hz on our 3090; see `docs/04_REPRO_LOG.md`).

---

## The lineage (read top-to-bottom as "what problem did the next one fix?")

- **RT-1** (Google, 2022) — Transformer over image+language → **discrete action tokens**. EfficientNet+FiLM
  vision, no big LLM. Establishes "control as token classification."
- **RT-2** (Google, 2023) — take a *web-pretrained VLM* (PaLI-X/PaLM-E) and **co-fine-tune it to emit
  action tokens as text**. Gets semantic generalization from internet pretraining. **OpenVLA is the
  open reproduction of this idea.**
- **Octo** (2024) — small (27–93M) Transformer policy trained on Open X-Embodiment with a **diffusion
  action head**; the "continuous, non-VLM" contrast point. LIBERO baseline.
- **OpenVLA** (2024) — 7B, **DINOv2+SigLIP → MLP → Llama-2**, 256-bin discrete actions, trained on 970k
  OXE trajectories. Open weights + code (this repo). Strong, but slow and needs per-robot fine-tuning.
- **OpenVLA-OFT** (2025) — same backbone, but **parallel-decode a continuous action chunk** with an MLP
  head (L1) + optional multi-view/proprio inputs → **25–50× faster**, higher success. *If you deploy one
  thing today, it's this.* (`openvla-oft.github.io`)
- **FAST** (Physical Intelligence, 2025) — a **DCT-based action tokenizer**: compress an action *chunk*
  into few tokens → keeps the discrete-VLA recipe but ~15× fewer tokens/inference.
- **π0 / π0-FAST** (Physical Intelligence, 2024–25) — **flow-matching** action expert attached to a
  PaliGemma-class VLM; action chunking at high control rates; strong generalist. Where the frontier is.
- **GR00T N1 / N1.5** (NVIDIA, 2025) — **dual-system**: a VLM ("System 2", slow semantic) + a **diffusion
  transformer action head** ("System 1", fast), cross-embodiment. (It's cached on this very machine:
  `~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-LIBERO` — a natural next thing to compare on LIBERO.)

Non-VLA manipulation baselines worth knowing (they beat early VLAs on narrow tasks):
- **ACT** (ALOHA) — **Action Chunking Transformer** + CVAE; predicts a chunk of future actions.
- **Diffusion Policy** — conditional diffusion over action chunks from vision+proprio (no language).
  These two are *why the field adopted action chunking*; OpenVLA's lack of it is its main weakness.

---

## Mapping to what you already know

| You know | VLA analogue | Note |
|---|---|---|
| DETR object **queries** + cross-attention decoder | *absent* in OpenVLA; **present** (as learned action/readout queries + chunk decoding) in OFT/ACT/π0/GR00T | the field is re-introducing query-like chunk decoders that OpenVLA dropped |
| PETR **3D positional priors** | *absent* in OpenVLA; multi-view/proprio in OFT, explicit state in π0/GR00T | geometry has to be learned implicitly in vanilla OpenVLA |
| Qwen-VL **visual tokens prepended to LLM** | identical in OpenVLA/RT-2/π0/GR00T System-2 | your VLM knowledge transfers directly to the "backbone" of every modern VLA |
| LLM **autoregressive decoding** | OpenVLA's action decoding; replaced by parallel/flow decoding in successors | AR decoding is the speed bottleneck the successors attack |
| Diffusion models (if you know them) | Octo/DP/π0/GR00T action heads | the "continuous multimodal action" answer to discretization |

---

## So which would you actually use? (a decision guide)

- **Learning / research clarity:** OpenVLA (this repo). The discrete-token design is the most
  transparent — you can read the whole action mechanism in ~70 lines (`docs/02_ARCHITECTURE.md` §6).
- **Deploying on a real robot, single-arm, today:** OpenVLA-**OFT** (speed + success + chunking).
- **High-frequency / bimanual / dexterous:** π0-class flow policies or GR00T-style dual-system.
- **Tiny/edge, no big LLM:** Octo or Diffusion Policy.

The throughline: **start from a pretrained VLM (your comfort zone), attach an action head, and choose
your action representation.** OpenVLA picks the simplest head (LM tokens); the frontier trades that
simplicity for speed and smoothness via chunked continuous/flow decoding.

---

### Suggested next experiment on this machine
You already have GR00T-N1.7-LIBERO cached. A great comparative exercise: run GR00T on the same
`libero_spatial` suite and compare success rate + inference speed against the OpenVLA numbers in
`study_outputs/libero_eval_libero_spatial.json`. Same benchmark, opposite action-representation choice
(diffusion head vs discrete tokens) — the cleanest possible A/B for understanding the trade-off.
