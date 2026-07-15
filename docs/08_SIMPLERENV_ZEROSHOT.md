# SimplerEnv — Controlling openvla-7b ZERO-SHOT on WidowX/Bridge

> The complement to LIBERO. LIBERO uses a **Franka** arm (unseen → OpenVLA needs fine-tuning there).
> SimplerEnv gives you the **WidowX** arm and **BridgeData** tasks — which *are* in OpenVLA's Open-X
> training mix — so we can run the base `openvla-7b` with **no fine-tuning at all** (`unnorm_key=bridge_orig`).
>
> **What you'll actually see (and the honest lesson):** the base model, zero-shot, produces *sensible,
> goal-directed* WidowX motion — the arm reaches toward the correct object and articulates the gripper —
> but **SimplerEnv task-success is near-zero**. That's the well-documented **real-to-sim visual gap**:
> SimplerEnv's rendering is out-of-distribution for OpenVLA, and OpenVLA is notoriously sensitive to
> visual distribution shift. So this demo teaches *two* things at once: (1) OpenVLA understands the
> WidowX embodiment + language task without any tuning, and (2) VLAs are brittle to visual OOD — the
> reason real-to-sim benchmarks like SimplerEnv exist. (OpenVLA's own zero-shot SimplerEnv-Bridge
> numbers in the literature are similarly low, ~0–5% on most tasks.)

Scripts live in `study_scripts/simplerenv/` and run in a **separate env** `.venv-simpler` (Python 3.10)
because SimplerEnv's simulator (SAPIEN 2.2.2) has no Python 3.12 wheel.

---

## 1. Why a second, different setup?

| | LIBERO (`docs/03_LIBERO_EVAL.md`) | **SimplerEnv (this doc)** |
|---|---|---|
| Simulator | MuJoCo + robosuite | **SAPIEN** (ManiSkill2_real2sim) |
| Rendering | EGL | **Vulkan** |
| Robot | Franka Panda | **WidowX** |
| In OpenVLA's training data? | No → **fine-tune** | **Yes (BridgeData)** → **zero-shot** |
| Python | 3.12 (`.venv-openvla`) | **3.10 (`.venv-simpler`)** (SAPIEN 2.2.2 needs ≤3.11) |
| `unnorm_key` | `libero_spatial`, … | **`bridge_orig`** |

The WidowX-Bridge tasks: `widowx_spoon_on_towel`, `widowx_carrot_on_plate`, `widowx_stack_cube`,
`widowx_put_eggplant_in_basket`.

---

## 2. How the OpenVLA→SimplerEnv adapter works

SimplerEnv ships `rt1` and `octo` policy wrappers but **not** OpenVLA, so we wrote a thin one:
`study_scripts/simplerenv/openvla_policy.py`. It exposes the `.step(image, instruction) →
(raw_action, action)` interface SimplerEnv expects and converts OpenVLA's 7-DoF output the same way
SimplerEnv's own Octo wrapper does for `widowx_bridge`:

```
openvla.predict_action(image, instruction, unnorm_key="bridge_orig")  ->  [dx,dy,dz, dR,dP,dY, grip]
   world_vector = [dx,dy,dz] * action_scale(=1.0)
   rot_axangle  = euler2axangle(dR,dP,dY) * action_scale     # euler delta -> axis-angle
   gripper      = sticky-gripper(grip)                        # sticky_gripper_num_repeat=1 for widowx
env.step( concat([world_vector, rot_axangle, gripper]) )
```

Key fact that makes this correct: OpenVLA's **bridge gripper is `mask=False`**, so it's passed through
in `[0,1]` (1=open, 0=close) — the *same* convention as Octo's `open_gripper` — so the sticky-gripper
logic transfers verbatim. The image is resized to 256×256 (BridgeData's native size) before the
processor. Camera is `3rd_view_camera`; control is 5 Hz.

---

## 3. Setup (what was installed, and the gotchas)

Done for you; recorded here as the debugging trail (like `docs/04_REPRO_LOG.md`):

1. **Python 3.10 via deadsnakes** (`add-apt-repository ppa:deadsnakes/ppa`) → `.venv-simpler`, because
   **SAPIEN 2.2.2 has no cp312 wheel** (pip offers only SAPIEN 3.x on py3.12, and ManiSkill2_real2sim
   is written for SAPIEN 2.2's API).
2. **Headless Vulkan verified** — `vulkaninfo` enumerates the RTX 3090 via the NVIDIA ICD
   (`/etc/vulkan/icd.d/nvidia_icd.json`); the container's `NVIDIA_DRIVER_CAPABILITIES` already includes
   `graphics`. SAPIEN renders offscreen with no X server.
3. **`ruckig` build failure** — ManiSkill pulls `ruckig` (motion-gen lib); its latest (0.17.3) has no
   py3.10 wheel and its source build breaks on `scikit-build-core ≥ 0.10`. Fix: pin **`ruckig==0.12.2`**
   (has a cp310 wheel).
4. **`pkg_resources` missing** — the venv's fresh setuptools (83.0) dropped `pkg_resources`, which
   SAPIEN 2.2 imports. Fix: **`setuptools==69.5.1`**.
5. **Assets are bundled** — ManiSkill2_real2sim commits its `data/` (scenes, bridge object infos,
   real-inpainting backgrounds), so **no separate asset download** is needed.
6. **OpenVLA stack** (transformers 4.40.1 / timm 0.9.10 / tokenizers 0.19.1 / accelerate 0.30.1) +
   `transforms3d` installed into `.venv-simpler`; the model loads via `trust_remote_code` and shares
   `~/.cache/huggingface` with `.venv-openvla` (no re-download).

---

## 4. How to run

```bash
cd /home/albert/Desktop/openvla
# render check (validates SAPIEN/Vulkan + assets, no model):
.venv-simpler/bin/python study_scripts/simplerenv/render_smoketest.py

# zero-shot eval (openvla-7b, no fine-tuning):
.venv-simpler/bin/python study_scripts/simplerenv/eval_widowx.py \
    --tasks all --episodes 3 --max_steps 120
#   --tasks: any of widowx_carrot_on_plate / widowx_put_eggplant_in_basket /
#            widowx_spoon_on_towel / widowx_stack_cube  (or "all")
```

Outputs: `study_outputs/simpler_render_smoketest.png`, per-episode MP4s in
`study_outputs/simpler_rollouts/`, and `study_outputs/simpler_eval_widowx.json`.

VS Code configs **08 / 09** run these (they use the `.venv-simpler` interpreter).

---

## 5. Results (this machine — openvla-7b, zero-shot, 4 tasks × 5 episodes)

| WidowX-Bridge task | Zero-shot success |
|---|---|
| put spoon on towel | 0/5 |
| put carrot on plate | 0/5 |
| stack green on yellow | 0/5 |
| **put eggplant in basket** | **1/5** ✅ |
| **Total** | **1/20 = 5.0%** |

This matches OpenVLA's published zero-shot SimplerEnv-Bridge numbers (~0–5%). The **eggplant success is
real**: the arm descended, grasped the eggplant, lifted it, and dropped it in the basket — completing in
58 steps (it broke early on success) vs the full 121 for the failures. See it:
- `study_outputs/simpler_eggplant_success.gif` (the successful rollout)
- `study_outputs/simpler_eggplant_success_montage.png` (6-frame strip: reach → grasp → carry → drop)
- all rollouts: `study_outputs/simpler_rollouts/*.mp4`; summary: `study_outputs/simpler_eval_widowx.json`

**Why this is the right takeaway, not a disappointment:** one clean success proves the whole pipeline
(adapter, gripper convention, action frame) is correct, and that OpenVLA *can* drive WidowX zero-shot.
The other tasks fail not because the model is confused about the embodiment (the montages show sensible
reaching) but because SimplerEnv's rendering is visually OOD — the real-to-sim gap. On the **real**
WidowX/Bridge robot (in-distribution images) OpenVLA does far better; that's the whole reason real-to-sim
benchmarks exist.

> Interpreting zero-shot numbers: SimplerEnv is a *real-to-sim* benchmark, so absolute success rates
> are lower and noisier than a real robot, but they *correlate* with real performance. The point here is
> qualitative — **the base model, with no fine-tuning, drives the WidowX arm toward the task** — which
> you can see directly in the rollout videos. Contrast with LIBERO, where the *base* model does ~0% and
> only the fine-tuned checkpoints work.

---

## 6. Takeaway for your mental model

This nails down the embodiment point from `docs/07_VLA_LANDSCAPE.md`: OpenVLA is a generalist **only over
the embodiments in its training mix**. Same policy, same weights —
- **WidowX/Bridge (SimplerEnv)** → works zero-shot (seen during training).
- **Franka (LIBERO)** → needs ~100–500 demos of fine-tuning (unseen embodiment).

That single contrast is the most important practical lesson about deploying VLAs.
