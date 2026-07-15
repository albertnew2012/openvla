# START HERE — Your OpenVLA Study Package

Hi Albert. You cloned OpenVLA to learn **VLA (Vision-Language-Action) models** coming from a strong
**DETR / PETR / Qwen-VL / LLM** background. This package sets the repo up end-to-end, reproduces the
headline results, and teaches the model — framed against what you already know. Everything is runnable
and documented. Start by reading in the order below.

> **What's the big idea?** OpenVLA = a Qwen-VL-style VLM (DINOv2+SigLIP → MLP → Llama-2) turned into a
> robot policy by predicting **actions as language tokens** (7 dims × 256 bins). No object queries, no
> diffusion head — just an LLM that outputs "action words." Full story in `02_ARCHITECTURE.md`.

---

## ✅ What was done for you (all verified on this RTX 3090)

| Thing | Result |
|---|---|
| Isolated env with OpenVLA's pinned deps (`.venv-openvla`), base container untouched | ✅ |
| `openvla-7b` inference | ✅ 15.1 GB VRAM, ~250 ms/action (SDPA) → `study_outputs/minimal_inference.png` |
| Headless MuJoCo/robosuite rendering (EGL) | ✅ `study_outputs/libero_render_smoketest.png` |
| **LIBERO reproduction — all 4 suites** | ✅ avg **74.1%** vs paper 76.5% — Spatial **81.0**, Object **80.0**, Goal **82.0**, Long **53.3** (paper 84.7/88.4/79.2/53.7). See `03_LIBERO_EVAL.md` §6 |
| **SimplerEnv WidowX-Bridge — ZERO-SHOT** | ✅ base openvla-7b drives the WidowX arm with **no fine-tuning** (SAPIEN/Vulkan). See `08_SIMPLERENV_ZEROSHOT.md` |
| Action-tokenization visualized | ✅ `study_outputs/action_tokenization.png` |
| QLoRA fine-tuning fits 24 GB | ✅ **peak 9.63 GB / 24 GB**, 0.73% params trainable (`05_FINETUNING_3090.md`) |
| ~10 real setup bugs diagnosed + fixed | ✅ logged in `04_REPRO_LOG.md` |

Rollout videos are under `rollouts/<date>/` (watch a `success=True` vs `success=False` one!).

---

## 📖 Reading order (this is your course)

1. **`00_START_HERE.md`** — you are here.
2. **`01_STUDY_PLAN.md`** — the phased plan tuned to your background; what transfers, what's new,
   the reading list, and self-test questions. *Read this second.*
3. **`02_ARCHITECTURE.md`** — the model, component by component, with exact shapes and DETR/PETR/VLM
   contrasts. *Your main text.* (You have it open.)
   - Companion: **`09_MODEL_STRUCTURE.md`** — the *concrete* module tree, real parameter counts, and
     layer dims dumped from the actual loaded model (numbers, not prose).
4. **`06_CODE_WALKTHROUGH.md`** — a line-by-line trace of one `predict_action` call through the code.
   Read alongside the source with your debugger.
5. **`03_LIBERO_EVAL.md`** — closed-loop evaluation explained (why control eval ≠ detection eval).
   - Companion: **`08_SIMPLERENV_ZEROSHOT.md`** — openvla-7b on **WidowX/Bridge zero-shot** (SAPIEN sim),
     and the real-to-sim visual-brittleness lesson (the flip side of the LIBERO fine-tuning story).
6. **`05_FINETUNING_3090.md`** — memory math + the QLoRA recipe that fits your 24 GB.
   - Companion: **`10_TRAINING_RECIPE.md`** — the full **Stage 0→3** pipeline (off-the-shelf → Prismatic
     VLM → OpenVLA VLA → your fine-tune) and **what's frozen vs trained** at each stage. Fact-checks the
     common 4-stage diagram.
7. **`07_VLA_LANDSCAPE.md`** — where OpenVLA sits vs RT-2/Octo/OFT/π0/GR00T, mapped onto your
   DETR/PETR/VLM knowledge. Read when you want the big picture / what to learn next.
8. **`04_REPRO_LOG.md`** — every environment problem + fix. Great debugging study and the source of
   truth for how the env is built.

---

## ▶️ How to run things

**One-click (recommended):** open the Run and Debug panel (Ctrl/Cmd+Shift+D) and pick a config from
`.vscode/launch.json`. They already use the isolated interpreter (`.venv-openvla/bin/python`) and the
right env vars (headless EGL, quiet TF). `justMyCode:false` lets you step into transformers/prismatic
internals — do this to learn.

Configs: `01 · Minimal inference` · `02 · LIBERO render smoke test` · `03 · Action tokenization` ·
`04 · LIBERO eval (quick 2×3)` · `04b · LIBERO eval (10×5)` · `05 · Canonical repo eval` ·
`06 · LoRA fine-tune (3090)`.

**Shell equivalents** live at the top of each `study_scripts/*.py` and in the docs. Always use the
isolated python and, for anything touching the simulator, the EGL env vars:

```bash
cd /home/albert/Desktop/openvla
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  .venv-openvla/bin/python study_scripts/04_libero_eval_subset.py --num_tasks 2 --num_trials_per_task 3
```

---

## 🗂️ Where everything lives

```
openvla/
├── docs/                         ← all the study docs (this folder)
│   ├── 00_START_HERE.md .. 06_CODE_WALKTHROUGH.md
├── study_scripts/                ← small, commented, runnable scripts
│   ├── 01_minimal_inference.py           (getting-started inference + viz)
│   ├── 02_libero_render_smoketest.py     (headless EGL render check)
│   ├── 03_action_tokenization_explainer.py (token↔action mapping + per-DoF distribution plot)
│   ├── 04_libero_eval_subset.py          (closed-loop LIBERO eval, subset + plots)
│   ├── 05_lora_finetune_3090.py          (QLoRA fine-tune demo on 24 GB)
│   ├── 06_verify_lora_adapter.py         (load saved adapter; prove it changed the policy)
│   └── simplerenv/                        (WidowX/Bridge ZERO-SHOT via SimplerEnv — uses .venv-simpler)
│       ├── openvla_policy.py                 (OpenVLA→SimplerEnv adapter)
│       ├── render_smoketest.py               (SAPIEN/Vulkan render check)
│       └── eval_widowx.py                    (zero-shot WidowX-Bridge eval)
├── study_outputs/                ← generated figures + JSON summaries
├── rollouts/<date>/              ← LIBERO episode MP4s (success/failure in the filename)
├── .vscode/launch.json           ← one-click run configs
├── .venv-openvla/                ← ISOLATED env for inference/LIBERO/fine-tune (py3.12, torch 2.2 / transformers 4.40.1 / timm 0.9.10)
├── .venv-simpler/                ← SEPARATE env for SimplerEnv (py3.10, SAPIEN 2.2.2) — WidowX/Bridge zero-shot
├── prismatic/extern/hf/          ← the model code that actually runs at inference (read this!)
├── experiments/robot/            ← evaluation code (LIBERO + Bridge)
└── vla-scripts/                  ← training / fine-tuning / deploy
```

Related repo on this machine: `/home/albert/Desktop/LIBERO` (the simulator; installed into the venv).

---

## ⚠️ Two things to know

1. **Use `.venv-openvla`, not the base env.** OpenVLA hard-requires `timm 0.9.10` / `transformers
   4.40.1`; the base container has timm 1.0 / transformers 5.x and would crash the model. The venv keeps
   both worlds intact. (Why: `02_ARCHITECTURE.md` §3 note + `04_REPRO_LOG.md` §1.)
2. **Minimal, documented edits** were made to a few repo files (guarded a lazy import, added an SDPA
   fallback, added a torch-CUDA-before-TF init, patched robosuite for mujoco 3.x). Each is commented in
   place and tabulated in `04_REPRO_LOG.md` §4. None change model behavior; they only make the code run
   on this Python-3.12 / no-flash-attn box.

Enjoy — start with `01_STUDY_PLAN.md`, keep `02_ARCHITECTURE.md` open, and run config `01` to see the
model predict an action in ~6 seconds.
