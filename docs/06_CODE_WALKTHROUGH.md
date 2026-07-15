# Code Walkthrough — One `predict_action` Call, Line by Line

> Follow this with the source open and your debugger set to `justMyCode:false` (all launch configs do
> this). We trace the exact call stack for the getting-started snippet, from PIL image to 7-DoF action.
> Every reference is `file:line`. The runtime code is the standalone HF port under `prismatic/extern/hf/`.

The call we're tracing:

```python
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)   # -> np.ndarray shape (7,)
```

---

## Step 0 — Registration & load (once)
`AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", trust_remote_code=True)` pulls
`modeling_prismatic.py` + `configuration_prismatic.py` from the Hub and builds
`OpenVLAForActionPrediction` (`modeling_prismatic.py:492`). Construction (`:213-255`) instantiates:
- `PrismaticVisionBackbone` (`:63`) — two timm ViTs (DINOv2 + SigLIP), forwards monkey-patched to
  return penultimate patches (`:85`, `:99`).
- `PrismaticProjector` (`:127`) — the 3-layer MLP.
- `self.language_model = AutoModelForCausalLM.from_config(text_config)` (`:248`) — Llama-2-7B.
- `OpenVLAForActionPrediction.__init__` (`:495`) also builds the 256-bin grid (`:500`) and sets
  `self.vocab_size = text_config.vocab_size - pad_to_multiple_of` = 32000 (`:504`).

---

## Step 1 — The processor: text → tokens, image → dual-normalized channel stack
`PrismaticProcessor.__call__` (`processing_prismatic.py:187`):

1. **Image** → `PrismaticImageProcessor.apply_transform` (`:128`). For the fused backbone this runs the
   transform **twice** — once per backbone — because DINOv2 and SigLIP use *different* mean/std/
   interpolation (`:70-113`):
   - `letterbox_pad_transform` (`:23`) pads to square with a mean-colour border (default strategy
     `letterbox`, `:118-119`) — preserves aspect ratio (no cropping distortion).
   - `Resize → CenterCrop → ToTensor → Normalize` per backbone (`:136-140`), then
     `torch.vstack([dino_img, siglip_img])` → **`pixel_values` shape `[6, 224, 224]`** (`:143`).
   - This channel-stacking is the trick that lets one tensor carry two differently-normalized views;
     the model splits it back at `modeling_prismatic.py:120`.
2. **Text** → the Llama tokenizer (`:208`). `prompt = "In: What action should the robot take to
   {instruction}?\nOut:"` → `input_ids` `[1, T]` (with BOS), `attention_mask` `[1, T]`.
3. Returns `BatchFeature{input_ids, attention_mask, pixel_values}` (`:216`).

After `.to("cuda:0", dtype=bfloat16)`: `pixel_values` `[1,6,224,224]` bf16, `input_ids` `[1,T]` long.

---

## Step 2 — `predict_action`: prompt fix + generate
`OpenVLAForActionPrediction.predict_action` (`modeling_prismatic.py:506`):

- **(:512)** If the last input token isn't `29871` (Llama's leading-space token), append it. This matches
  the training format so the first generated token is a clean action token. *Tiny detail, real bugs if
  omitted.*
- **(:518)** `generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), ...)`.
  `get_action_dim` (`:554`) = `len(norm_stats[unnorm_key]["action"]["q01"])` = **7** for Bridge/LIBERO.
  So generate emits exactly 7 tokens.

`self.generate(...)` is HF `GenerationMixin.generate`, which calls `forward` once per new token.

---

## Step 3 — First forward pass (multimodal): the interesting one
`PrismaticForConditionalGeneration.forward` (`:291`), multimodal branch (`:361-415`):

1. **Vision** — `patch_features = self.vision_backbone(pixel_values)` (`:366`):
   - `torch.split(pixel_values, [3,3], dim=1)` → two `[1,3,224,224]` (`:120`).
   - Each ViT's patched `forward` returns penultimate-layer patch tokens: DINOv2 `[1,256,1024]`,
     SigLIP `[1,256,1152]`.
   - `torch.cat([...], dim=2)` → **`[1, 256, 2176]`** (`:123`).
2. **Project** — `projected_patch_embeddings = self.projector(patch_features)` (`:369`) →
   MLP `2176→8704→4096→4096` → **`[1, 256, 4096]`** (`PrismaticProjector.forward :146`).
3. **Splice** — build the multimodal sequence (`:383`):
   `multimodal_embeddings = cat([ embed(BOS), projected_patches(256), embed(rest_of_prompt) ], dim=1)`
   → **`[1, 1+256+(T-1), 4096]`**. The attention mask is extended in parallel (`:386-390`); labels
   (training only) get `IGNORE_INDEX=-100` for the 256 visual positions (`:392-401`).
4. **LLM** — `self.language_model(inputs_embeds=multimodal_embeddings, ...)` (`:404`) → Llama runs causal
   self-attention over `[BOS, vision, text]`; returns `logits` `[1, seq, 32064]`.
   *There is no cross-attention and no queries — vision is just a prefix the text/action tokens attend to.*

The next-token logits (last position) are used by `generate` to pick action token #1
(`do_sample=False` → argmax).

---

## Step 4 — Cached decode for action tokens #2..#7
For each subsequent step, `generate` calls `forward` with `input_ids.shape[1]==1` and the KV cache
(`:325-341`). This is the cheap unimodal path — no vision recompute, just one token through Llama with
`past_key_values`. Batch size is hard-limited to 1 (`:326`, `:463`; `prepare_inputs_for_generation :450`).
After 7 steps, `generated_ids` holds the prompt + **7 action token ids**.

---

## Step 5 — De-tokenize: token ids → physical action
Back in `predict_action` (`:520-536`):

```python
predicted_action_token_ids = generated_ids[0, -7:]          # the 7 action tokens
discretized_actions = self.vocab_size - predicted_action_token_ids     # id -> bin index (:522)
discretized_actions = np.clip(discretized_actions - 1, 0, 254)         # (:523)
normalized_actions = self.bin_centers[discretized_actions]             # bin -> value in [-1,1] (:524)

action_norm_stats = self.get_action_stats(unnorm_key)                  # q01/q99/mask (:527)
action = np.where(mask,
                  0.5*(normalized_actions+1)*(q99-q01)+q01,            # un-normalize (:530-534)
                  normalized_actions)                                   # (gripper dim left as-is)
return action                                                          # np.ndarray, shape (7,)
```

That's the whole VLA-specific part: **the top 256 vocab ids are action bins; 7 of them decode to a
7-DoF action, then percentile-un-normalized per dataset.**

---

## Step 6 — Shapes recap (bridge/libero, one image)
```
pixel_values [1,6,224,224] ─split→ 2×[1,3,224,224] ─2 ViTs→ [1,256,1024]+[1,256,1152]
             ─cat→ [1,256,2176] ─MLP→ [1,256,4096]
input_ids [1,T] ─embed→ [1,T,4096]
splice → [1, 1+256+(T-1), 4096] ─Llama→ logits [1,·,32064] ─generate×7→ 7 token ids → action (7,)
```

---

## Set these breakpoints to watch it happen
Run VS Code config **"01 · Minimal inference"** with breakpoints at:
1. `processing_prismatic.py:143` — see `pixel_values` become `[6,224,224]` (channel stack).
2. `modeling_prismatic.py:120` — the split back into two images.
3. `modeling_prismatic.py:123` — the fused `[1,256,2176]`.
4. `modeling_prismatic.py:383` — the `[BOS, vision, text]` splice (inspect the shapes).
5. `modeling_prismatic.py:521` — the raw 7 action token ids before decoding.
6. `modeling_prismatic.py:530` — the final un-normalized action.

Step through once and the model stops being a black box. Then compare mentally to a DETR decoder step
(queries cross-attending to memory) — the contrast is the whole lesson.

---

## Bonus: the eval hot loop (closed-loop)
For LIBERO, the same `predict_action` is called every control step inside
`run_libero_eval.py:186-246`, with the image re-fetched from the simulator after each `env.step`.
See `docs/03_LIBERO_EVAL.md` for the loop and the gripper/coordinate conventions.
