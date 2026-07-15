# OpenVLA — The Full Training Recipe (Stage 0 → 3)

> How `openvla-7b` is built, from off-the-shelf networks to your task fine-tune, focused on the one thing
> that matters most: **what is frozen vs trained at each stage** (the freeze/thaw story *is* the recipe).
> Fact-checked against the OpenVLA & Prismatic papers, the **actual loaded model** (`09_MODEL_STRUCTURE.md`),
> and this repo's `vla-scripts/finetune.py`. Companion to `05_FINETUNING_3090.md` (the 24 GB memory math).

This doc was written to sanity-check a 4-stage pipeline diagram. **Verdict: the diagram is ~85% right** —
the freeze/thaw logic for Stages 0–2 is correct (including the key insight that vision is *unfrozen* in
Stage 2). Two things to fix are called out in **§"Is the diagram right?"** at the end.

---

## TL;DR — the four stages

```
Stage 0   off-the-shelf         SigLIP(0.4B) + DINOv2(0.3B) + Llama-2(6.7B)   ← just downloaded, 0 trained
   │
Stage 1   VLM pretrain          Prismatic-7B: teach the LLM to SEE
   │      (Prismatic)           vision ❄️ frozen · projector 🔥NEW · LLM 🔥full     objective: text next-token CE
   │
Stage 2   VLA pretrain          OpenVLA: teach the VLM to ACT
   │      (OpenVLA, 970K demos) EVERYTHING 🔥 (vision unfrozen too)               objective: action-token CE
   │
Stage 3   task fine-tune        specialize on your robot/task
          (LIBERO, ALOHA, …)    base ❄️ + LoRA 🔥 (~1–2%)                          objective: action-token CE
                                (OFT variant adds a continuous head + proprio + FiLM — see §3B)
```

`openvla-7b` on the Hub = **the output of Stage 2.** Stages 0–2 are already done for you; you only ever
run **Stage 3.**

---

## Master table — who trains what

| Stage | Data | Objective | Vision | Projector | LLM | New parts | Trained |
|---|---|---|:--:|:--:|:--:|---|---|
| **0** off-the-shelf | web-scale (Meta/Google) | (their own) | ❄️ ckpt | — | ❄️ ckpt | — | **0** (downloaded) |
| **1** VLM (Prismatic) | LLaVA-1.5 instruct mix | text next-token CE | ❄️ **frozen** | 🔥 **new (~71M)** | 🔥 full | projector | proj + LLM (~6.8B) |
| **2** VLA (OpenVLA) | 970K OXE robot demos | **action-token CE** | 🔥 **unfrozen** | 🔥 | 🔥 | *lm_head reused as action head (0 new)* | **all ~7.5B** |
| **3A** fine-tune *(this repo)* | your demos (LIBERO…) | **action-token CE** (discrete) | ❄️ (LoRA) | ❄️ (LoRA) | ❄️ (LoRA) | LoRA r=32 (~1–2%) | ~1–2% (LoRA) |
| **3B** OFT *(separate repo)* | LIBERO / ALOHA | **L1 regression on action chunks** | LoRA or full | | | continuous head + proprio + FiLM (new) | LoRA/full + new heads |

Legend: ❄️ frozen · 🔥 trained · 🔥new = new params, trained from scratch.

### Reading the table: why "trained" jumps 6.8B → 7.5B (Stage 1 → 2)

**The model does not grow — it is the same 7.5B params in both stages.** Only the *freeze status* changes:
the 6.8B → 7.5B jump is **exactly the vision encoder being unfrozen**, not new weights being added.

| Component | Params | Stage 1 (VLM) | Stage 2 (VLA) |
|---|--:|:--:|:--:|
| DINOv2 ViT | 0.30B | ❄️ frozen | 🔥 **trained** |
| SigLIP ViT | 0.43B | ❄️ frozen | 🔥 **trained** |
| Projector | 0.07B | 🔥 trained | 🔥 trained |
| Llama-2 | 6.74B | 🔥 trained | 🔥 trained |
| **Total model** | **7.54B** | 7.54B | 7.54B |
| **→ trained** | | **6.81B** | **7.54B** |
| **→ frozen** | | 0.73B (vision) | 0 |

- **Stage 1 trained** = projector (0.07B) + LLM (6.74B) = **6.81B** ("~6.8B"); vision (0.73B) is **frozen**.
- **Stage 2 trained** = everything = **7.54B** ("~7.5B"); nothing frozen.
- **Difference** = `7.54 − 6.81 = 0.73B` = **SigLIP (0.43B) + DINOv2 (0.30B)** = the two ViTs.

```
Stage 1:  [ vision 0.73B ❄️ ] + [ projector+LLM 6.81B 🔥 ]  = 7.5B total,  6.8B trained
Stage 2:  [ vision 0.73B 🔥 ] + [ projector+LLM 6.81B 🔥 ]  = 7.5B total,  7.5B trained
                     ↑ this 0.73B flipped ❄️ → 🔥
```

The vision backbone was *always there*; Stage 1 holds it **fixed** (frozen feature extractor — standard
VLM practice), Stage 2 **unfreezes** it so the ViTs adapt to robot images. **That flip *is* the 6.8B → 7.5B
change** — and the OpenVLA paper found this vision-unfreezing materially improved manipulation.

---

## Stage 0 — off-the-shelf ingredients
Three pretrained checkpoints, **nothing trained by the VLA team** — just downloaded:

| Component | Who trained it | How | Params |
|---|---|---|---|
| **SigLIP** SO400M/14 | Google | image–text contrastive (sigmoid) | 0.43 B |
| **DINOv2** ViT-L/14 | Meta | self-supervised (self-distillation) | 0.30 B |
| **Llama-2 7B** | Meta | web-scale text next-token | 6.74 B |

Why these: SigLIP = *semantic* eyes, DINOv2 = *geometric* eyes, Llama = the reasoning/sequence engine.
(Full "why two ViTs" story: `09_MODEL_STRUCTURE.md` §2.)

---

## Stage 1 — VLM pretraining (Prismatic-7B): teach the LLM to *see*
- **Data:** the LLaVA-1.5 instruction-tuning mixture (image–text QA/caption/instruct). **No robots yet.**
- **Objective:** ordinary **next-token cross-entropy on text** — a standard VLM.
- **Freeze/thaw:** **vision ❄️ frozen**; the **projector is new** (randomly initialized) and trained 🔥;
  the **full LLM is trained** 🔥. Single-stage (Prismatic dropped LLaVA's projector-only warmup).
- **What's created here:** the **projector** — the only from-scratch module in the whole pipeline. It maps
  fused patch features (2176-d) → Llama width (4096-d). In the actual `openvla-7b` it's a **3-layer MLP
  ≈ 71 M params** (2176→8704→4096→4096), *not* 30M — see §"Is the diagram right?".
- **Result:** Prismatic-7B, a VLM that can describe images and answer visual questions.

---

## Stage 2 — VLA pretraining (OpenVLA): teach the VLM to *act*
- **Data:** **~970K real robot demonstrations** curated from **Open X-Embodiment** (many robots, tasks,
  scenes) — the big, expensive stage.
- **Objective:** **cross-entropy on action tokens only.** Actions are discretized into 256 bins and
  written as the top-256 vocab ids (31744–31999); the prompt is masked (labels = −100). **No new action
  network** — the existing `lm_head` is reused as the "action head" (`09_MODEL_STRUCTURE.md` §5, §7.5).
- **Freeze/thaw:** **nothing frozen — all ~7.5B trained**, and crucially **the vision encoder is
  *unfrozen* too.** The OpenVLA paper reports this was **important**: fine-tuning DINOv2+SigLIP on robot
  images (vs. the usual frozen-vision VLM practice) materially improved manipulation. *This is the single
  most important line in the whole recipe, and the diagram gets it right.*
- **Cost (why you won't redo it):** ~**64 A100 GPUs × ~14 days** (~21.5K A100-hours), global batch ~2048.
- **Result:** `openvla-7b` — the checkpoint you download.

---

## Stage 3 — Task fine-tuning: two *different* recipes (don't conflate them)

### 3A. Standard OpenVLA fine-tuning — **what this repo does**
The original-paper recipe, implemented in `vla-scripts/finetune.py` and mirrored by your
`study_scripts/05_lora_finetune_3090.py`:
- **Objective:** **the *same* discrete action-token cross-entropy as Stage 2** — `loss = output.loss`
  ([finetune.py:261](../vla-scripts/finetune.py#L261)). **No regression, no new head.** (The `l1_loss` at
  [:286](../vla-scripts/finetune.py#L286) is only a *monitoring metric*, decoded from the argmax tokens —
  **not** the training loss.)
- **Freeze/thaw:** **base 7.5B frozen ❄️ + LoRA adapters 🔥** on `all-linear`, **rank 32**,
  `lora_alpha = min(32,16) = 16` ([finetune.py:172-178](../vla-scripts/finetune.py#L172-L178)) → ~1–2%
  trainable. Optionally 4-bit (**QLoRA**) to fit small GPUs (your 3090 demo).
- **Defaults:** batch 16, lr 5e-4, `image_aug=True` ([finetune.py:87-100](../vla-scripts/finetune.py#L87-L100)).
- **New parts:** **none** — just LoRA deltas (§`09` on where the 439 adapters land).

> So your QLoRA runs are **Stage 3A**, *not* OFT. Same objective and token scheme as pretraining; only the
> weights are nudged by a low-rank adapter.

### 3B. OpenVLA-OFT ("Optimized Fine-Tuning") — a **separate 2025 paper & repo**
This is what the diagram's Stage 3 actually depicts. It **changes the recipe** to boost speed & success:
- **Continuous action head + L1 regression** (replaces the 256-bin discrete tokens).
- **Action chunking** (predict ~8 steps at once) + **parallel decoding** (one forward, non-autoregressive →
  ~25–40× faster).
- **Proprioceptive state input** and **FiLM** language conditioning (esp. for ALOHA).
- **Freeze/thaw:** LoRA *or* full fine-tune of the base, **plus new small heads trained from scratch**
  (the continuous action head, proprio projection, FiLM layers).
- **Result:** LIBERO ~97% (vs ~76% base). **Lives in a separate codebase** (`moojink/openvla-oft`) — **not
  in this repo.** (What OFT does *and* doesn't fix is discussed in the chat/`07_VLA_LANDSCAPE.md`.)

---

## How the training data actually flows — single frames, shuffled (behavior cloning)

**This surprises everyone:** even though the raw demos are *trajectories* (video-like), OpenVLA does **not**
train on video. It trains on **shuffled single-frame `(image, instruction, one action)` tuples** — the
classic **behavior-cloning** setup. True for **both** Stage 2 (pretrain) and Stage 3A (fine-tune); they use
the same RLDS pipeline (`prismatic/vla/datasets/rlds/dataset.py`).

```
raw trajectories (episodes: [img₀,a₀], [img₁,a₁], …)          ← temporal, video-like
   │  chunk to windows: window_size=1, future_action_window_size=0   (dataset.py:260-261, 334-340)
   ▼          → each frame becomes a self-contained (1 image, 1 action) sample
flatten to individual frames
   │  interleave many datasets "at the Frame Level"                  (dataset.py:563-564)
   ▼
SHUFFLE at the frame level   dataset.shuffle(shuffle_buffer_size)    (dataset.py:572; buffer is "in frames", :479)
   ▼
image aug → batch → model
```
Temporal order **within an episode is discarded** at training time; consecutive samples in a batch are
unrelated frames from different episodes / robots / tasks.

**Why shuffling is not just allowed but *desirable*:** OpenVLA is a **Markovian (memoryless) policy** —
`action = π(image, instruction)`, no history. So each `(image, action)` pair is an **independent supervised
example** (like `(image, label)` in classification). Shuffling decorrelates gradients (adjacent video frames
are near-identical → correlated, high-variance updates) and mixes robots/tasks within each batch. Feeding
video in order would produce terrible, correlated mini-batches.

**Training vs inference — same mapping, different order:**

| | Training | Inference |
|---|---|---|
| unit | 1 frame → 1 action | 1 frame → 1 action |
| order | **shuffled** (order discarded) | **sequential** (closed loop) |
| nature | discrete, i.i.d.-ish samples | temporal rollout |

The model learns the frame→action mapping from a shuffled pile; at runtime you apply it step-by-step in
temporal order. **Neither uses video as input.** (Training order ≠ inference order.)

**The knobs (and how OFT differs):** two parameters *would* make it continuous, both OFF for base OpenVLA:
- `window_size = 1` → 1 image of context (`>1` = a short *history* of frames).
- `future_action_window_size = 0` → 1 action (`>0` attaches a *chunk* of future actions = OFT's
  action-chunking).

Even OFT (chunk `>0`) keeps samples **shuffled at the frame level** — the chunk is a longer *label*, not a
requirement to stream video in order.

> **One line:** raw data = trajectories, but training = **shuffled single-frame `(image, instruction,
> action)` tuples** (behavior cloning). Discrete and shuffleable *because the policy is memoryless*; the
> continuous/temporal part exists only at **inference**, as the closed loop.

---

## A concrete training example — one sample, end to end

Follow a single sample all the way to the loss. (This uses the study demo `study_scripts/05`, but the tensor
format is **identical** to real RLDS fine-tuning — same `pixel_values` / `input_ids` / `labels`.)

**1. The raw triple `(image, instruction, action)`:**
- **image:** 224×224, a **green block at bottom-center**
- **instruction:** `"pick up the green block"`
- **GT action (7-DoF):** `[-0.01, +0.34, -0.25, 0, 0, 0, +1.0]`  → *move right/down a bit · descend · close gripper*

**2. Tokenize → three tensors** (`build_example`, [05:70-85](../study_scripts/05_lora_finetune_3090.py#L70-L85)):
- `pixel_values [1,6,224,224]` — the image (DINOv2+SigLIP dual-stack) — the visual **input**.
- `input_ids [1,28]` and `labels [1,28]`:

```
pos:         0 ─────────────────── 19 │ 20     21     22    23    24    25    26  │ 27
input_ids:  <s> In:…green block?\nOut: ▁│ 31873  31829  31904 31872 31872 31872 31744│ </s>
labels:    -100 -100 ……………………… -100 -100│ 31873  31829  31904 31872 31872 31872 31744│  2
               └─── PROMPT (masked, -100) ──┘└──────── GT action tokens (supervised) ───────┘ EOS
```
- The **7 action tokens ARE the GT**, living in `labels` (positions 20–26) + EOS. Decoded back:
  `31873→dx≈-0.01 · 31829→dy≈+0.34 · 31904→dz≈-0.25 · 31872×3→rot 0 · 31744→grip +1` — i.e. exactly the action from step 1.
- **Length 28** = BOS(1) + prompt text(18) + space `▁`(1) + action(7) + EOS(1). `labels` is the *same length*
  as `input_ids` (one target per position); `-100` means "don't grade the prompt."

**3. One forward pass computes the loss** ([05:172-176](../study_scripts/05_lora_finetune_3090.py#L172-L176)):
```
pixel_values → DINOv2+SigLIP → projector → 256 visual tokens ┐
input_ids    → embed_tokens  → 28 text embeddings            ├→ [BOS, 256 visual, text] (~284) → 32 Llama layers
                                                             ┘        → lm_head → logits; + labels ⇒ out.loss
```
**Teacher-forced**, shifted cross-entropy with `-100` ignored → the loss grades **only** the 7 action tokens + EOS. It teaches:
```
"…Out: ▁"        → dx (31873)
"…Out: ▁ 31873"  → dy (31829)      … each DoF, conditioned on the GT prefix …
"…31744"         → </s>  (stop after 7 tokens)
```
(The 256 visual tokens are inserted *inside* `forward` and masked `-100` automatically — which is why
`labels` is 28, not ~284.)

**4. Backward → update** ([05:177](../study_scripts/05_lora_finetune_3090.py#L177), [190-191](../study_scripts/05_lora_finetune_3090.py#L190-L191)): gradients flow into the **~1% LoRA weights only**; `opt.step()` every N steps nudges them.

> **One line:** *(image + instruction) is the **question**, the 7 action tokens in `labels` are the **answer
> key**; one teacher-forced forward pass measures the model's surprise at that answer and updates the LoRA weights.*

---

## Is the diagram right?

**Mostly yes** — and it nails the key idea (vision unfrozen in Stage 2). Fix these:

1. ⚠️ **Projector size.** Diagram says *"Proj · 30M"*; the **actual `openvla-7b` projector is ~71 M** (a
   3-layer MLP, dumped from the real model in `09_MODEL_STRUCTURE.md` §1/§3). *30M* would be a 2-layer
   projector — not this model.
2. ⚠️ **Stage 3 blends two recipes.** *"L1 regression on action chunks · Proprio · FiLM · new heads"* is
   **OFT (3B)**, a separate paper/repo. **Standard fine-tuning (3A)** — the original paper, this repo's
   `finetune.py`, and your `study_scripts/05` — keeps **discrete action tokens + cross-entropy**, base ❄️ +
   LoRA r=32, and **no new head / no proprio / no FiLM**. Splitting Stage 3 into **3A vs 3B** makes the
   diagram correct.
3. ✅ **Everything else checks out:** off-the-shelf sizes (SigLIP 0.4B / DINOv2 0.3B / Llama 6.7B), Stage-1
   *vision frozen · projector new · full LLM trained*, Stage-2 *970K OXE · action-token CE · all unfrozen*,
   and *"LoRA ~1%"* (rank-32 all-linear ≈ 1–2%; your rank-16 run was 0.73%).

---

## What you can actually run in THIS repo
| Stage | Runnable here? | How |
|---|---|---|
| 0 | n/a | weights auto-download with `openvla-7b` |
| 1 (Prismatic VLM) | ❌ pre-done | already baked into `openvla-7b` |
| 2 (VLA pretrain, 970K) | ❌ (64×A100×14d) | you download the result, not reproduce it |
| **3A** standard fine-tune | ✅ | `vla-scripts/finetune.py` (LoRA r32; needs an RLDS dataset) **or** the 24 GB QLoRA demo `study_scripts/05` (**config 06**) |
| 3B OFT | ❌ here | separate repo `moojink/openvla-oft` |

**One line:** *off-the-shelf → make it see (Prismatic, vision frozen) → make it act (OpenVLA, everything
unfrozen on 970K demos) → make it yours (LoRA on your task, same token objective; OFT is a separate,
head-swapping variant).*
