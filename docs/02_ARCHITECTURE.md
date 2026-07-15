# OpenVLA Architecture — A Deep Dive for a DETR / PETR / VLM Person

> Audience: you already know DETR, PETR, Qwen-VL–style VLMs, and LLMs.
> This doc explains OpenVLA by *contrast* with what you already know, then goes
> component-by-component with exact tensor shapes and code pointers.
>
> All code pointers are `path:line` into this repo. The model's runtime code lives
> in two mirrored places:
> - `prismatic/extern/hf/` — the standalone HuggingFace port (what actually runs at inference).
> - `prismatic/models/`, `prismatic/vla/` — the original training codebase (Prismatic VLMs).

---

## 0. The one-paragraph mental model

**OpenVLA is a 7B VLM (Prismatic = DINOv2+SigLIP vision → MLP → Llama-2) that has been
turned into a robot policy by reframing "continuous action prediction" as "next-token
prediction over a discretized action vocabulary."** You give it one RGB image + a language
instruction; it autoregressively emits 7 tokens; each token is 1 of 256 bins; those 7 bins
decode to a 7-DoF end-effector delta `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`. That's it.
No object queries, no cross-attention adapters, no diffusion head, no action decoder network —
just a language model that outputs "action words."

If DETR turned detection into **set prediction with learned queries**, and Qwen-VL turned
perception into **text generation conditioned on visual tokens**, then OpenVLA turns control
into **text generation where the "text" is a robot action**.

---

## 1. Where VLA sits relative to what you know

| Aspect | DETR / PETR | Qwen-VL (VLM) | **OpenVLA (VLA)** |
|---|---|---|---|
| Task | 2D/3D detection | image→text | image+text→**action** |
| Vision encoder | CNN/ViT backbone + positional enc | ViT (often SigLIP/CLIP) | **fused DINOv2 + SigLIP ViT-L** |
| Query mechanism | learned object queries + cross-attn in a **decoder** | none (decoder-only LLM) | none (decoder-only LLM) |
| Fusion of vision | queries cross-attend to feature maps | visual tokens **prepended** to text tokens | visual tokens **inserted after BOS**, prefix to text |
| Head | FFN → box + class | LM head → text tokens | LM head → **action tokens** (reuse vocab) |
| Output space | continuous boxes (regressed) | discrete text tokens | **discretized** continuous actions (256 bins/dim) |
| Loss | Hungarian matching + L1/GIoU | next-token CE | **next-token CE on action tokens only** |
| Positional/geometry prior | strong (2D sine, 3D PE in PETR) | rope in LLM | rope in LLM (no explicit action/geometry prior) |

**The single most important conceptual jump:** in DETR/PETR the *output geometry is baked into
the head* (boxes, 3D positions, reference points). In OpenVLA there is **no geometric head at
all** — the action is emitted as language tokens and the "geometry" must be learned implicitly
inside the transformer. This is why VLAs are data-hungry and need per-robot fine-tuning, but
also why they inherit the LLM's semantic generalization (open-vocabulary instructions).

---

## 2. End-to-end forward pass (image + instruction → action)

```
                 ┌────────────────────────── OpenVLA-7B ──────────────────────────┐
 RGB 224×224 ──▶ │  Vision backbone (FUSED)                                        │
 (one image)     │    ├─ DINOv2 ViT-L/14  ─▶ 256 patch tokens × 1024              │
                 │    └─ SigLIP SO400M/14 ─▶ 256 patch tokens × 1152              │
                 │        concat on feature dim ─▶ 256 × 2176                      │
                 │                                                                 │
                 │  Projector (MLP: 2176→8704→4096→4096, GELU) ─▶ 256 × 4096       │
                 │                                                                 │
 "In: What       │  Tokenizer ─▶ text token embeddings (Llama-2)                   │
  action ...?    │                                                                 │
  \nOut:"        │  Sequence = [BOS] + [256 visual tokens] + [text tokens]         │
                 │                                                                 │
                 │  Llama-2-7B decoder (32 layers, d=4096) ──▶ logits (vocab)      │
                 │                                                                 │
                 │  generate() 7 steps, greedy ─▶ 7 action token ids               │
                 └─────────────────────────────────────────────────────────────────┘
                                        │
             de-tokenize: token id ─▶ bin index ─▶ bin center in [-1,1]
                                        │
             un-normalize with dataset q01/q99 ─▶ real action [Δx..Δyaw, grip]
```

Code path for the whole thing at inference:
`OpenVLAForActionPrediction.predict_action` → `.generate()` → `PrismaticForConditionalGeneration.forward`
in `prismatic/extern/hf/modeling_prismatic.py:506` and `:291`.

---

## 3. Component 1 — The fused vision backbone (DINOv2 + SigLIP)

Class: `PrismaticVisionBackbone` — `prismatic/extern/hf/modeling_prismatic.py:63`.

For `openvla-7b` the vision config is `dinosiglip-vit-so-224px`
(`prismatic/extern/hf/configuration_prismatic.py:22,36`):
- **Featurizer α** = `vit_large_patch14_reg4_dinov2.lvd142m` (DINOv2 ViT-L/14, 4 register tokens), embed 1024.
- **Featurizer β** = `vit_so400m_patch14_siglip_224` (SigLIP SO400M/14), embed 1152.

Both are created via `timm.create_model(..., num_classes=0)` and their `forward` is monkey-patched to
return **the second-to-last block's patch tokens** (not the final layer!):

```python
self.featurizer.forward = unpack_tuple(
    partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
)   # modeling_prismatic.py:85
```

Why penultimate features? This is a Prismatic finding — the last ViT block is over-specialized to
the pretraining objective; penultimate patch features transfer better to downstream fusion.
(Analogous to how detection necks often tap intermediate CNN stages rather than the classifier layer.)

**Input packing (important & unusual).** The two backbones want *different normalization*, so the
processor stacks the two differently-normalized copies of the image **on the channel axis**:
`pixel_values` has shape `[B, 6, 224, 224]` (2×3 channels). At `modeling_prismatic.py:120` it is split
back into two `[B,3,224,224]` images, each run through its own backbone, and the patch features are
concatenated on the **feature** dim:

```python
img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
return torch.cat([patches, patches_fused], dim=2)   # [B, 256, 1024+1152=2176]
```

- 224/14 = 16 → **16×16 = 256 patch tokens** from *each* backbone (they align spatially).
- Output: **256 visual tokens, each 2176-dim.**

**Contrast with DETR/PETR:** there is no FPN, no positional encoding added here, no multi-scale.
The ViT's own patch positions are the only spatial signal, and downstream the LLM's causal RoPE is
what orders the 256 tokens. There is also **no [CLS] pooling** — every patch token is passed on, like
visual tokens in a VLM.

> Aside — a HF gotcha you'll see in the code: `LayerScale` in timm stores a parameter named `gamma`,
> and HF's `from_pretrained` renames any param containing "gamma"/"beta". The code monkey-patches
> LayerScale to rename `gamma → scale_factor` (`modeling_prismatic.py:49-59`). This is why the model
> validates `timm==0.9.10..0.9.16` and **hard-crashes on timm ≥ 1.0** (`modeling_prismatic.py:221`).

---

## 4. Component 2 — The projector (vision → LLM width)

Class: `PrismaticProjector` — `prismatic/extern/hf/modeling_prismatic.py:127`.

Because the backbone is fused, the projector is a **3-layer GELU MLP** (a plain 2-layer MLP is used
for single backbones):

```
2176 ──Linear──▶ 8704 ──GELU──▶ 8704 ──Linear──▶ 4096 ──GELU──▶ 4096 ──Linear──▶ 4096
(vision_dim)     (4×vision)                        (llm_dim)                       (llm_dim)
```

Output: **256 tokens × 4096** — ready to drop into Llama-2's embedding stream. This is the entire
"connector." No perceiver resampler, no cross-attention, no query compression (contrast Flamingo /
BLIP-2 / Qwen-VL's resampler). 256 visual tokens is cheap enough to just concatenate.

---

## 5. Component 3 — The LLM and multimodal fusion (prefix insertion)

LLM: **Llama-2-7B** (`llama2-7b-pure`, `configuration_prismatic.py:48,59`), instantiated with
`AutoModelForCausalLM.from_config(text_config)` at `modeling_prismatic.py:248`. d=4096, 32 layers,
32 heads, vocab 32000 (+ pad to a multiple of 64).

**Fusion = literal concatenation in the embedding sequence**, inserted right after the BOS token
(`modeling_prismatic.py:383`):

```python
input_embeddings = self.get_input_embeddings()(input_ids)   # text embeds
multimodal_embeddings = torch.cat(
    [input_embeddings[:, :1, :],      # <BOS>
     projected_patch_embeddings,      # 256 visual tokens
     input_embeddings[:, 1:, :]],     # the rest of the prompt
    dim=1,
)
```

The attention mask and (during training) the label tensor are patched in parallel, with the 256
visual positions set to `IGNORE_INDEX = -100` so **no loss is computed on visual tokens**
(`modeling_prismatic.py:392-401`).

**Contrast with everything you know:**
- **vs Flamingo/BLIP-2:** those keep the LLM frozen and inject vision via *gated cross-attention*
  layers. OpenVLA/Prismatic instead makes vision a **prefix** and fine-tunes the whole stack. Simpler,
  but you pay full 7B gradients.
- **vs DETR decoder:** DETR's decoder queries *cross-attend* to encoder memory. Here there is only
  **self-attention** over `[BOS, vision, text]` — the "cross-modal attention" is just ordinary causal
  self-attention inside Llama. Vision tokens can be attended to by later text/action tokens because
  they're earlier in the (causal) sequence.
- **vs PETR:** PETR injects 3D position embeddings so queries can reason in 3D. OpenVLA has **no such
  geometric prior** — 3D understanding is entirely implicit/learned. This is a real limitation and a
  big reason OFT/action-chunking variants exist.

There is a subtle inference detail worth knowing: the HF `forward` special-cases three regimes —
cached single-token generation (`input_ids.shape[1]==1`), unimodal, and multimodal — see
`modeling_prismatic.py:319-431`. Only the **first** generation step is multimodal (it builds the
`[BOS,vision,text]` prefix and the KV cache); every subsequent action-token step is a cheap 1-token
decode reusing the cache. Generation is hard-restricted to **batch size 1** (`:326`, `:463`).

---

## 6. Component 4 — Action tokenization (the heart of "VLA")

This is the idea that makes a VLM into a policy. Defined in `OpenVLAForActionPrediction`
(`modeling_prismatic.py:492`).

**Discretization.** Each continuous action dimension (already normalized to ~[-1,1], see §7) is
binned into **256 uniform bins** over [-1, 1]:

```python
self.bins = np.linspace(-1, 1, config.n_action_bins)   # 256 edges  (:500)
self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
```

**Vocabulary reuse (the clever/cheap part).** Rather than add new tokens, OpenVLA **overwrites the
256 least-used tokens at the top of the Llama vocab** and treats them as action bins. At decode:

```python
predicted_action_token_ids = generated_ids[0, -action_dim:]          # last 7 ids
discretized_actions = self.vocab_size - predicted_action_token_ids    # id → bin index
discretized_actions = np.clip(discretized_actions - 1, 0, 254)
normalized_actions = self.bin_centers[discretized_actions]            # bin idx → value in [-1,1]
```

where `self.vocab_size = text_vocab_size - pad_to_multiple_of` (`:504`). So bin `b` ↔ token id
`vocab_size - b`. The mapping is a simple linear reflection of the top of the vocab.

**Action dimensionality is data-driven.** `get_action_dim(unnorm_key)` returns
`len(norm_stats[unnorm_key]["action"]["q01"])` (`:554`) — 7 for BridgeData/LIBERO single-arm robots.
So `generate(max_new_tokens=7)`.

**Contrast:** DETR regresses box coords with an L1/GIoU loss; a diffusion policy denoises continuous
actions. OpenVLA instead **classifies** each action dim into 256 buckets with a plain cross-entropy —
identical machinery to language modeling. Pros: reuse the whole LLM + tokenizer + training loop, get
multi-modality "for free" via the softmax. Cons: 256-bin quantization ceiling, one token per dim per
step (slow; no chunking), and no native notion of action smoothness. (These are exactly what the
newer **FAST** tokenizer and **OFT** recipe attack — see `docs/01_STUDY_PLAN.md` §Frontier.)

---

## 7. Normalization / un-normalization (why `unnorm_key` exists)

Robots have wildly different action scales, so during data prep every dataset's actions are normalized
per-dimension to roughly [-1,1] using **1st/99th percentiles** (robust to outliers). Those percentiles
are stored per dataset in the model config's `norm_stats`. At inference the model *reverses* it
(`modeling_prismatic.py:526-534`):

```python
action = 0.5 * (normalized + 1) * (q99 - q01) + q01     # where mask is True
```

- `unnorm_key` picks *which dataset's* stats to use (e.g. `bridge_orig`, `libero_spatial`).
- A per-dim `mask` leaves some dims (typically the **gripper**) un-normalized — the gripper is a
  near-binary open/close, handled separately (see the gripper sign-flipping dance in
  `experiments/robot/robot_utils.py:75-102`).
- Fine-tuned checkpoints ship a `dataset_statistics.json` that overrides `norm_stats`
  (`experiments/robot/openvla_utils.py:60`).

This is the robotics analogue of DETR normalizing boxes by image size — except here the "image size"
is a learned percentile range per embodiment, and picking the wrong `unnorm_key` silently produces
garbage actions.

---

## 8. The prompt and the processor

Prompt template (base OpenVLA), `experiments/robot/openvla_utils.py:163`:

```
In: What action should the robot take to {instruction}?\nOut:
```

The processor (`prismatic/extern/hf/processing_prismatic.py`) does two things: tokenizes that string
(Llama tokenizer) and runs the **dual image transform** producing the `[6,224,224]` channel-stacked
`pixel_values`. One gotcha encoded in `predict_action` (`:512`): Llama's tokenizer, after the `:` in
`Out:`, must be followed by the SentencePiece "space" token id `29871` to match training; the code
appends it if missing. Small detail, big correctness impact — a good example of how brittle
"LLM-as-policy" plumbing can be.

---

## 9. Shapes cheat-sheet (openvla-7b, one image)

| Stage | Tensor | Shape |
|---|---|---|
| Input image (per backbone) | pixel_values | `[1, 6, 224, 224]` |
| DINOv2 patches | — | `[1, 256, 1024]` |
| SigLIP patches | — | `[1, 256, 1152]` |
| Fused patches | — | `[1, 256, 2176]` |
| Projected visual tokens | — | `[1, 256, 4096]` |
| Text prompt tokens | input_ids | `[1, T]` (T≈20) |
| Fused sequence | inputs_embeds | `[1, 1 + 256 + (T-1), 4096]` |
| LM logits (step) | logits | `[1, ·, 32064]` |
| Generated action ids | — | `[1, 7]` |
| Decoded action | np.ndarray | `[7]` |

Total params ≈ **7.5B** (Llama-2-7B ≈ 6.7B + two ViT-L ≈ 0.6B + projector). In **bf16 ≈ 15 GB**,
which is why a single 24 GB RTX 3090 can do inference comfortably (see `docs/04_REPRO_LOG.md`).

---

## 10. Training objective (for completeness)

Standard causal LM cross-entropy, but **masked to the 7 action tokens only** (vision + prompt tokens
carry `IGNORE_INDEX`). So training OpenVLA = "teacher-force the 7 action tokens and minimize CE."
`action_accuracy` in logs = fraction of the 7 tokens predicted exactly right. Because it's just an LM
loss, the entire HF/PyTorch training stack applies unchanged — this is the whole point of the design.
Pretraining used 970k trajectories from Open X-Embodiment; fine-tuning (§study plan) adapts to one
embodiment with ~100–500 demos.

---

## 11. Read-the-code checklist (in this order)

1. `prismatic/extern/hf/configuration_prismatic.py` — how a string like `dinosiglip-vit-so-224px`
   expands into timm ids, resolutions, and the Llama text_config.
2. `prismatic/extern/hf/modeling_prismatic.py:63-158` — vision backbone + projector.
3. `:291-447` — the three-regime `forward` and the `[BOS,vision,text]` splice.
4. `:492-562` — action (de)tokenization + un-normalization. **This is the VLA-specific 70 lines.**
5. `experiments/robot/openvla_utils.py` — how eval actually calls the model (prompt, center-crop,
   `predict_action`).
6. `experiments/robot/libero/run_libero_eval.py` — the closed-loop control loop (see §arch of the
   eval in `docs/03_LIBERO_EVAL.md`).

---

### Appendix: glossary for the robotics-new reader
- **7-DoF action**: 6-DoF end-effector delta pose (translation xyz + rotation rpy) + 1 gripper command.
- **End-effector (EEF)**: the robot's "hand"/tool tip. Actions are usually *deltas* on the EEF pose.
- **RLDS**: Reinforcement Learning Datasets format (TFDS-based) used by Open X-Embodiment.
- **Rollout**: one closed-loop episode of the policy driving the sim/robot until success or timeout.
- **Embodiment**: a specific robot + camera + control setup; different embodiments ≈ different "languages" of action.
- **Action chunking**: predicting several future timesteps at once (OpenVLA does *not*; OFT/ACT/diffusion do).
