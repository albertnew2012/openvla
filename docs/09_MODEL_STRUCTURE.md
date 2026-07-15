# OpenVLA-7B — Concrete Model Structure

> A layer-level, numbers-are-real companion to `docs/02_ARCHITECTURE.md` (which is the conceptual
> version). Everything below was dumped from the **actual loaded model** (`OpenVLAForActionPrediction`),
> not from memory. Reproduce it yourself with the snippet in §8.

## 0. The whole model in one picture

```
OpenVLAForActionPrediction  ────────────────────────────────────  7.541 B params
├── vision_backbone : PrismaticVisionBackbone .................... 730.9 M  ( 9.7%)
│     ├── featurizer        = DINOv2 ViT-L/14  (24 blocks, dim 1024) ...... 303.2 M
│     └── fused_featurizer  = SigLIP SO400M/14 (27 blocks, dim 1152) ...... 427.7 M
├── projector      : PrismaticProjector (3-layer MLP) ............  71.4 M  ( 0.9%)
└── language_model : LlamaForCausalLM  (Llama-2-7B, 32 layers) ... 6738.9 M  (89.4%)
      └── + lm_head doubles as the "action head" (no extra params)
```

**Read this:** ~**89%** of OpenVLA is just Llama-2-7B. The "robot" parts (two vision ViTs + a tiny
projector) are only ~11%. OpenVLA is a **language model with eyes**, and the action output is produced
by the *same* `lm_head` that would produce words — there is **no separate action network**.

---

## 1. Parameter budget (real counts)

| Component | Class | Params | Share |
|---|---|---|---|
| Vision backbone (fused) | `PrismaticVisionBackbone` | 730.9 M | 9.7% |
|  ├─ DINOv2 ViT-L/14 | timm `vit_large_patch14_reg4_dinov2` | 303.2 M | |
|  └─ SigLIP SO400M/14 | timm `vit_so400m_patch14_siglip_224` | 427.7 M | |
| Projector | `PrismaticProjector` | 71.4 M | 0.9% |
| Language model | `LlamaForCausalLM` (Llama-2-7B) | 6 738.9 M | 89.4% |
| **Total** | `OpenVLAForActionPrediction` | **7.541 B** | 100% |

In bf16 that's ~**15.1 GB** on the GPU (2 bytes/param) — the number you see at load time.

---

## 2. Component 1 — Vision backbone (two ViTs, fused)

`PrismaticVisionBackbone` holds **two independent Vision Transformers** and concatenates their patch
features. Both take a 224×224 image, patch size 14 → **16×16 = 256 patch tokens** each.

| | featurizer (α) | fused_featurizer (β) |
|---|---|---|
| Model | **DINOv2** ViT-L/14 (+4 register tokens) | **SigLIP** SO400M/14 |
| timm id | `vit_large_patch14_reg4_dinov2.lvd142m` | `vit_so400m_patch14_siglip_224` |
| Transformer blocks | 24 | **27** |
| Embed dim | 1024 | **1152** |
| Patches out | 256 | 256 |
| Params | 303.2 M | 427.7 M |

- Each ViT's `forward` is monkey-patched to return the **second-to-last block's** patch tokens
  (`get_intermediate_layers(n={depth-2})`) — penultimate features transfer better.
- The two 256-token outputs are concatenated on the **feature axis**:
  `[B,256,1024] ⊕ [B,256,1152] → [B, 256, 2176]`. That **2176 = 1024 + 1152** is the backbone's
  output `embed_dim`.
- Input packing: the processor stacks two differently-normalized copies of the image on the channel
  axis → `pixel_values [B, 6, 224, 224]`; the backbone splits `[3,3]` and sends each to its ViT.

(Note: they are *not* both "ViT-L" — DINOv2 is ViT-Large (24×1024); SigLIP SO400M is a wider/deeper
"shape-optimized" ViT (27×1152). SigLIP is the bigger of the two.)

### Why *two* ViTs? (the key design choice, inherited from Prismatic VLMs)

The two backbones are trained with **different objectives**, so they capture **complementary** visual
information — and manipulation needs both. Prismatic VLMs showed that fusing them beats either alone.

| | **SigLIP** SO400M | **DINOv2** ViT-L |
|---|---|---|
| Trained with | image–**text** contrastive (sigmoid loss) | **self-supervised**, images only (self-distillation + masked modeling) |
| Features aligned with | **language / semantics** | **spatial / geometric structure** |
| Strong at | *"**what** is this?"* — objects, categories, reading, matching the instruction words | *"**where** is it, what's its shape / pose?"* — dense localization, correspondence, near-segmentation-quality patches |
| Weak at (alone) | precise spatial detail (contrastive features pool toward what's *nameable*) | grounding *language* (never saw text) |

- **Why it matters for a VLA specifically:** manipulation is overwhelmingly **spatial** — to grasp
  something you need precise **location, shape, and pose**, not just its name. SigLIP alone (great
  semantics, mushy geometry) is imprecise for localization; DINOv2 alone (great geometry) can't ground
  the instruction. Fused, the LLM gets **semantic grounding + geometric precision at every patch** —
  exactly what *"put the spoon on the towel"* (identify + localize + place) demands.
- **Why the fusion is clean:** both use patch-14 @ 224px → the **same 16×16 = 256-patch grid**, so patch
  *i* of DINOv2 and patch *i* of SigLIP are the **same spatial location**. They're concatenated
  **per-patch** on the feature axis (`1024 ⊕ 1152 = 2176`). This is *why* they must share patch
  size/resolution, and why `pixel_values` is `[1,6,224,224]` (two copies, each with its own normalization).
- **Cost/benefit:** two ViTs ≈ 731 M params + two vision passes, but vision is only ~10% of the 7.5 B
  model — cheap relative to the manipulation quality it buys.

**One line:** SigLIP = *semantic eyes* (aligned to the instruction), DINOv2 = *geometric eyes* (aligned
to space); fused per-patch so the LLM sees both "what" and "where."

---

## 3. Component 2 — Projector (vision dim → LLM dim)

`PrismaticProjector` — a 3-layer GELU MLP that maps each fused patch token (2176-d) to Llama's width
(4096-d):

```
fc1 : Linear(2176 → 8704)   # 8704 = 4 × 2176
act_fn1 : GELU
fc2 : Linear(8704 → 4096)
act_fn2 : GELU
fc3 : Linear(4096 → 4096)
```
Output: **256 tokens × 4096** — ready to drop into the LLM's embedding stream. 71.4 M params, the only
"connector." No perceiver/resampler/cross-attention (contrast Flamingo/BLIP-2/Qwen-VL).

---

## 4. Component 3 — Language model (Llama-2-7B)

`LlamaForCausalLM`, standard Llama-2-7B:

| hyperparam | value |
|---|---|
| decoder layers | 32 |
| hidden size | 4096 |
| attention heads | 32 |
| KV heads | 32 (full MHA — *not* grouped-query) |
| MLP intermediate | 11008 |
| vocab size | **32064** (32000 + 64 pad-to-multiple) |
| positional | RoPE (rotary), applied per layer |

**One decoder block** (`LlamaDecoderLayer`, ×32) — this is the repeated unit:
```
input_layernorm          : LlamaRMSNorm
self_attn                : LlamaSdpaAttention
    q_proj/k_proj/v_proj : Linear(4096 → 4096)   # + rotary_emb (RoPE)
    o_proj               : Linear(4096 → 4096)
post_attention_layernorm : LlamaRMSNorm
mlp                      : LlamaMLP
    gate_proj/up_proj    : Linear(4096 → 11008)   # SwiGLU
    down_proj            : Linear(11008 → 4096)
```
Per block ≈ 4·(4096²) attn + 3·(4096·11008) MLP ≈ **202 M params**; ×32 ≈ 6.5 B, plus the token
embedding + `lm_head` (32064×4096 each) → 6.74 B total.

The LLM operates on the **fused sequence** `[BOS, 256 visual tokens, text tokens]` — visual tokens are
spliced in as a *prefix*, and ordinary causal self-attention does all the cross-modal mixing. There is
no cross-attention module.

---

## 5. Component 4 — The "action head" (there isn't a separate one)

OpenVLA adds **zero** new parameters for actions. It reuses Llama's `lm_head : Linear(4096 → 32064)`:
- The **top 256 vocab ids (31744–31999)** are reinterpreted as **256 action bins** over [-1, 1]
  (`n_action_bins=256`, `bin_centers` has 255 entries).
- `decode_vocab = 32064 − 64 = 32000` is the reference point used to map token id ↔ bin.
- At inference, `predict_action` calls `.generate(max_new_tokens=7)` → 7 tokens → 7 bins → un-normalize.
- Those 256 ids were **real (rare) Llama tokens** (e.g. id 31744 was `'忠'`, 31999 was `'给'`) — *overwritten*
  by fine-tuning to mean action bins, **not** newly added tokens. Where they're stored and how the model
  learns to emit them: **§7.5**.

So "the head" = the language-model head + a fixed arithmetic decode. (Full story: `docs/02_ARCHITECTURE.md` §6.)

---

## 6. End-to-end tensor-shape flow (real dims, one image)

```
pixel_values                                  [1, 6, 224, 224]
  └─split[3,3]→ 2 images                       [1,3,224,224] ×2
        ├─ DINOv2  → patch tokens              [1, 256, 1024]
        └─ SigLIP  → patch tokens              [1, 256, 1152]
  └─concat(dim=2) → fused patches              [1, 256, 2176]
projector(fused)                               [1, 256, 4096]     (256 visual tokens)

input_ids (text)                               [1, T]  (T≈19)
  └─ embed_tokens                              [1, T, 4096]

splice [BOS, vision, text[1:]]                 [1, 1 + 256 + (T−1), 4096]  = [1, ~275, 4096]
  └─ 32× LlamaDecoderLayer                     [1, ~275, 4096]
  └─ lm_head                                   [1, ~275, 32064]  (logits)

generate ×7 (autoregressive, KV-cached)  →     7 action-token ids  (in 31744..31999)
decode: id → bin → bin_center[-1,1] → un-normalize(q01,q99)  →  action  [7]
```

---

## 7. Input & output tensor analysis (`generate()` I/O + the vocab layout)

### 7.0 The I/O contract at a glance

OpenVLA's entire job, in one line:

> **(1 RGB image + 1 language instruction)  →  one 7-DoF end-effector movement delta, per control step.**

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  1 RGB image  +  text instruction ("pick up the red block")      │
   └───────────────────────────────┬─────────────────────────────────┘
                                    ▼
                                OpenVLA
                                    ▼
        [ dx, dy, dz, dRoll, dPitch, dYaw, gripper ]   ← one 7-DoF delta
                                    ▼
                          execute on the robot
                                    ▼
                              new RGB image  ──┐
                                    ▲          │  repeat ~60–300× until the
                                    └──────────┘  task succeeds (closed loop)
```

- **Inputs = image *and* instruction.** The text is essential — the *same* image with a *different*
  instruction gives a *different* action. (No depth, no proprioception, no 3D — a single 2D frame; see
  `docs/02_ARCHITECTURE.md` §5.)
- **Output = a *relative* move**, not a target coordinate: a small nudge (~cm) of the gripper this step,
  plus a gripper open/close.
- **One step, looped.** Each call gives the *next* move for the *current* image; a full task is the
  emergent result of running this perception→action loop hundreds of times — OpenVLA never outputs a
  whole plan or trajectory in one shot.

The rest of §7 details the exact tensors behind this contract.

### 7.1 Inputs — what `processor(prompt, image)` produces
The model is called with a dict of exactly three tensors:

| key | shape | dtype | what it is |
|---|---|---|---|
| `input_ids` | `[1, T]` (T≈19) | long | text prompt tokens; **BOS at position 0** (the image is *not* here) |
| `attention_mask` | `[1, T]` | long | all 1s (no padding for a single prompt) |
| `pixel_values` | `[1, 6, 224, 224]` | bf16 | the dual DINOv2+SigLIP channel-stacked image |

The **256 visual tokens do not appear in the inputs** — they're created *inside* `forward()` (vision
backbone → projector) and spliced in as `[BOS, 256 visual, text]`. So the LLM's real sequence length is
`1 + 256 + (T−1) ≈ 275`, not `T`.

### 7.2 The vocabulary layout (you must know this to read the outputs)
The LM head emits **32064** logits per position, laid out as:
```
id      0 … 31743   →  normal Llama-2 word tokens
id  31744 … 31999   →  the 256 ACTION BINS         (the top 256 of the 32000 real vocab)
id  32000 … 32063   →  64 PAD tokens (pad_to_multiple_of=64) — junk, NOT actions
```
- `text_config.vocab_size = 32064` (= real 32000 + 64 pad) → the width of every logits row / `gen.scores[i]`.
- `vla.vocab_size = 32000` (the `V` used for decoding) `= 32064 − 64`.
- **⚠️ The action bins (31744–31999) are NOT the last tokens** — the 64 pad tokens sit *after* them.
  So `logits[-256:]` grabs the **wrong** ids (`31808…32063`). Always index explicitly:
  `logits[31744:32000]` (i.e. `logits[np.arange(V-256, V)]`).

### 7.3 Outputs of `generate(max_new_tokens=7, output_scores=True, return_dict_in_generate=True)`

| field | shape / type | what it is |
|---|---|---|
| `gen.sequences` | `[1, T+7]` | the full sequence; **`gen.sequences[0, -7:]` = the 7 generated action-token ids** |
| `gen.scores` | **tuple of length 7** | one entry **per autoregressive step**; `gen.scores[i]` = `[1, 32064]` **logits** (raw, *not* probabilities) for the i-th action token |

Two common misreadings:
- `gen.scores` is a **tuple of 7** (that's the proof of 7 sequential decode steps), not one big tensor.
- Each `gen.scores[i]` holds **logits** → you must `softmax` to get a distribution.

So the per-DoF **distribution over the 256 bins** (what `03_action_tokenization_explainer.py` plots) is:
```python
logits_i = gen.scores[i][0, 31744:32000]     # 256 LOGITS for DoF i   (NOT [-256:] — pad tokens!)
p_i      = torch.softmax(logits_i, dim=-1)    # 256 probabilities
```

### 7.4 Tokens → physical action (the decode, per DoF)
```
token_id ─► bin = V − token_id − 1   (V=32000)                 # e.g. 31911 → bin 88
         ─► normalized = bin_centers[bin] ∈ [−1,1]             # → −0.3059
         ─► action = 0.5·(norm+1)·(q99−q01) + q01  (unnorm_key)# → −0.0089 m
```
(The gripper has `mask=False`, so it skips the un-normalize step and passes through in [0,1].) This is
exactly the `token_id → bin_idx → norm → ACTION` table that `03` prints; the code is at
[modeling_prismatic.py:520-534](../prismatic/extern/hf/modeling_prismatic.py#L520-L534).

### 7.5 Where the output is *stored*, what those tokens *were*, and how generation lands there

Three questions people always ask about the "action head" — all have concrete answers.

**(a) Where the output physically lives — 256 rows of two existing matrices.**
An action is just a vocab id, so the *only* place the action output is "stored" is the **256 rows
indexed 31744–31999** of the LLM's two vocab-width matrices — no separate weights exist (§5):

| matrix | shape | the action rows (31744–31999) do… |
|---|---|---|
| `embed_tokens` | `Embedding(32064, 4096)` | the **input** side: the vector summed back into the sequence when a just-generated action token is fed to the next step |
| `lm_head` | `Linear(4096 → 32064)` | the **output** side: `hidden · lm_head[id]` is the logit for that action bin |

So "the action head" = **512 rows** (256 in + 256 out) of a 4096-wide matrix that already existed. Nothing
outside these rows encodes the action vocabulary.

**(b) Those 256 ids were REAL Llama tokens — OpenVLA *overwrites* them (it does not add tokens).**
The vocab is **not** resized. OpenVLA hijacks the **256 least-used existing tokens** (the RT-2 trick) and
redefines their meaning. Decode them on the base tokenizer and you see their *original* text — mostly
rare CJK / Unicode pieces (chosen *because* they're rare, so clobbering them barely dents language
ability):

| token id | original SP piece | OpenVLA now means |
|---|---|---|
| 31744 | `'忠'` | action bin 255 (≈ **+1.0**) |
| 31745 | `'错'` | action bin 254 |
| 31872 | `'Ÿ'` | action bin 127 |
| 31900 | `'书'` | action bin 99 |
| 31999 | `'给'` | action bin 0 (≈ **−1.0**) |

The string is *still attached* in the tokenizer; what changed is that **fine-tuning rewrote those 256
`embed_tokens` / `lm_head` rows** so that, in the `Out:` context, id 31744 encodes "max-positive action
bin" instead of "忠". Same slot, new meaning. (Verify: `tok.decode([31744])` → `'忠'`.)

**(c) How generation reliably lands in 31744–31999 — *learned*, not forced.**
At each of the 7 steps `lm_head` produces **all 32064 logits**; nothing hard-restricts them to the action
block. Instead, **fine-tuning reshaped** `P(next token | "…Out: ")` to put ~all its mass on 31744–31999
(that's exactly what the peaky per-DoF distributions in `03` show). So plain greedy `argmax`
(`do_sample=False`) lands in the action block every time — the model simply *learned* that "after `Out:`,
emit action words." `predict_action` applies **no logit mask** (you *could* add one to guarantee validity,
but the base model doesn't need it). The 7 tokens are produced **autoregressively**: each id is fed back
through its `embed_tokens` row to condition the next (dx → dy → dz → dR → dP → dY → gripper).

```
   final hidden state  h  ─►  lm_head (4096→32064)  ─►  32064 logits
                                                          │ argmax (learned to peak here)
                          31744 ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ 31999   ▼
                          └──── one of the 256 action ids ────┘  ─► feed back via embed_tokens ─► next DoF
```

### 7.6 The autoregressive decode — position vs value, and where the "7 loop" lives

Two things that confuse everyone at first.

**① Position ≠ value — they are two independent axes.**
- A token's **position** in the 7-token output (1st…7th) says **which DoF** it is (dx, dy, dz, dRoll,
  dPitch, dYaw, gripper) — a **fixed order** the model learned.
- A token's **id** (which of the 256 in 31744–31999) says **that DoF's magnitude** — chosen per step.
- They are **not** a counter `31744, 31745, 31746…`. Real example — action
  `[-0.90, -0.25, 0.0, 0.10, 0.0, 0.0, 1.0]` → ids `[31987, 31904, 31872, 31859, 31872, 31872, 31744]`:

| step (position) | = which DoF | value | → **token id** |
|---|---|---|---|
| 1 | dx | −0.90 | 31987 |
| 2 | dy | −0.25 | 31904 |
| 3 | dz | 0.00 | 31872 |
| 4 | dRoll | +0.10 | 31859 |
| 5 | dPitch | 0.00 | 31872 |
| 6 | dYaw | 0.00 | 31872 |
| 7 | gripper | +1.00 | 31744 |

The ids jump around, and **31872 repeats 3×** — one for each DoF whose value is `0.0`. That repeat is the
proof: **the id encodes the *value*, not a position counter.** (Direction: more-negative → higher id near
31999; more-positive → lower id near 31744, since `id = 32000 − bin`.)

**② Where the "run 7 times" loop physically is — it's in HuggingFace, not OpenVLA.**
`predict_action` has **no `for` loop**; it calls `self.generate(input_ids, max_new_tokens=7)` once, and the
loop lives inside transformers' `_greedy_search`:

```
predict_action                                   modeling_prismatic.py:518
  └─ self.generate(..., max_new_tokens=7)         transformers/generation/utils.py
       └─ mode == GREEDY_SEARCH → _greedy_search()          utils.py:2310
            └─ while _has_unfinished_sequences():           utils.py:2489   ◄── THE 7× loop
                   outputs = self(**model_inputs)            utils.py:2494   one OpenVLA forward  (= 1 DoF)
                   next_tokens = argmax(logits[:, -1, :])    utils.py:2530   greedy pick → id ∈ [31744,31999]
                   input_ids = cat([input_ids, next_tokens]) utils.py:2539   APPEND = the "feed back"
  ◄─ generated_ids[0, -7:]  → the 7 action tokens            modeling_prismatic.py:521
```
- **One loop iteration = one DoF.** The append at `utils.py:2539` is the autoregression — iteration *k+1*
  sees the token chosen at iteration *k* (the per-step cached forward is `modeling_prismatic.py:325`).
- **Why exactly 7:** `max_new_tokens = get_action_dim = 7` becomes a `MaxLengthCriteria`; action ids are
  never the EOS token, so it never stops early → exactly 7.
- **`gen.scores` is a tuple of 7** because `utils.py:2512` appends one logits tensor per iteration (that's
  the proof of 7 sequential steps).

**One line:** `generate(max_new_tokens=7)` → a `while` loop in `_greedy_search` that does *forward → argmax →
append* **7 times**; position picks the DoF, the argmax id picks the magnitude.

---

## 8. The real module tree (abridged `print(model)`)

```
OpenVLAForActionPrediction(
  (vision_backbone): PrismaticVisionBackbone(
    (featurizer):       VisionTransformer( patch_embed(14×14), blocks: 24 × Block, norm )   # DINOv2, 1024-d
    (fused_featurizer): VisionTransformer( patch_embed(14×14), blocks: 27 × Block, norm )   # SigLIP, 1152-d
  )
  (projector): PrismaticProjector(
    (fc1): Linear(2176, 8704)  (fc2): Linear(8704, 4096)  (fc3): Linear(4096, 4096)
    (act_fn1): GELU  (act_fn2): GELU
  )
  (language_model): LlamaForCausalLM(
    (model): LlamaModel(
      (embed_tokens): Embedding(32064, 4096)
      (layers): 32 × LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(q_proj,k_proj,v_proj,o_proj: Linear(4096,4096); rotary_emb)
        (mlp): LlamaMLP(gate_proj,up_proj: Linear(4096,11008); down_proj: Linear(11008,4096); act_fn=SiLU)
        (input_layernorm), (post_attention_layernorm): LlamaRMSNorm(4096)
      )
      (norm): LlamaRMSNorm(4096)
    )
    (lm_head): Linear(4096, 32064)   # ← also the action head
  )
)
```

---

## 9. Explore it yourself

```python
import torch
from transformers import AutoModelForVision2Seq
m = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)     # CPU is fine for structure

print(m)                                                    # full module tree
sum(p.numel() for p in m.parameters()) / 1e9                # 7.541  (billions)
{n: sum(p.numel() for p in c.parameters())/1e6              # per-component millions
 for n, c in m.named_children()}
m.vision_backbone.featurizer.blocks                          # the 24 DINOv2 blocks
m.language_model.model.layers[0]                             # one Llama decoder block
m.bin_centers.shape, m.config.n_action_bins                  # the action bins
```
In the debugger (launch config 01, `justMyCode:false`), pause after the model loads and type these in
the Debug Console, or step into `predict_action` to watch the shapes in §6 go by.

---

## 10. Mental-model summary
- **One sentence:** OpenVLA-7B = DINOv2+SigLIP (256+256 patch tokens) → 3-layer MLP → Llama-2-7B, with
  the LLM head doubling as a 256-bin action classifier.
- **Where the parameters are:** 89% LLM, 10% vision, 1% projector, 0% action head.
- **What's *not* there:** no object queries, no cross-attention adapters, no diffusion/regression head,
  no proprioception input — the deliberate minimalism that makes it "a VLM that speaks actions."
