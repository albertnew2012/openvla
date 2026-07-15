# LIBERO Closed-Loop Evaluation — How It Works & How to Run It

> LIBERO is a MuJoCo/robosuite manipulation benchmark. Evaluating a VLA here is fundamentally
> different from evaluating a detector: it is **closed-loop** — the policy's own actions change the
> next observation, so errors compound. This is the methodological heart of "policy" evaluation.

Code: `experiments/robot/libero/run_libero_eval.py` (canonical) and
`study_scripts/04_libero_eval_subset.py` (configurable subset + plots). Both share the same utils.

---

## 1. The task suites

| Suite | Tests | Horizon (max steps) | Paper OpenVLA SR |
|---|---|---|---|
| `libero_spatial` | same objects, **different spatial layouts** | 220 | 84.7% |
| `libero_object`  | **different objects**, same layout | 280 | 88.4% |
| `libero_goal`    | same objects/layout, **different goals** | 300 | 79.2% |
| `libero_10` (long) | 10 **long-horizon** multi-step tasks | 520 | 53.7% |
| `libero_90`      | 90 short tasks (pretraining suite) | 400 | — |

Each suite has 10 tasks × 50 predefined initial states. The paper's numbers are 3 seeds × 500 rollouts.

---

## 2. The closed-loop control loop (what actually runs)

For each task → each initial state → run an episode:

```
reset env; set fixed initial state
for t in range(max_steps + 10):
    if t < 10:                       # (1) let objects settle: no-op action
        step([0,0,0,0,0,0,-1]); continue
    img  = agentview_image[::-1,::-1]           # (2) 180° flip to match training
    img  = tf.image.resize(img, (224,224), lanczos3)   # (3) same resize as training dataloader
    if center_crop: img = crop 90% area, resize back    # (4) match train-time random-crop aug
    prompt = "In: What action should the robot take to {task}?\nOut:"
    action = vla.predict_action(processor(prompt, img), unnorm_key=suite)   # (5) 7-DoF
    action = normalize_gripper(action); action = invert_gripper(action)     # (6) gripper conventions
    obs, reward, done, _ = env.step(action)     # (7) execute; observation changes
    if done: success += 1; break
save rollout mp4
```

The seven numbered steps map 1:1 to lines in `run_libero_eval.py:186-246`. Points worth internalizing:

- **(1) settle steps** — the sim drops objects on reset; the first 10 steps are no-ops. (A robotics
  quirk with no analogue in static-image tasks.)
- **(2) 180° flip & (3) lanczos resize** — *distribution matching*. The policy is brittle to
  preprocessing mismatch; the eval reproduces the exact training-time image pipeline
  (`libero_utils.py:33-58`). This is the robotics version of "use the same normalization at test time,"
  but far more consequential.
- **(4) center crop** — the LIBERO checkpoints were fine-tuned with random 90%-area crops, so at test
  time we take the **center 90% crop** (`--center_crop True`). Skipping this measurably drops success.
- **(6) gripper gymnastics** — the RLDS dataloader standardizes gripper to `[0,1]` with `0=close`;
  the sim wants `[-1,+1]` with `-1=open`. So the eval rescales and flips the sign
  (`robot_utils.py:75-102`). Getting this wrong inverts the gripper and fails silently.
- **(7) closed loop** — `env.step(action)` changes the world; the next image depends on what the
  policy just did. Compounding error is why a policy that's 95% right per-step can still fail an
  episode, and why success rate (not token accuracy) is the real metric.

**`done` == task success**: LIBERO sets `done=True` only when the task's success predicate is met, so
`total_successes / total_episodes` is the success rate.

---

## 3. Why closed-loop matters (the DETR/PETR contrast)

- Detection/PETR eval is **open-loop**: a fixed dataset, each image scored independently, metric = mAP.
  One bad prediction doesn't corrupt the next input.
- Policy eval is **closed-loop**: the model is in a feedback loop with the environment. A single early
  mistake (e.g., knocking the bowl) shifts every future observation off-distribution (covariate shift),
  and the policy may never recover. This is why imitation-learned policies need either lots of data,
  action chunking, or DAgger-style corrections.
- Practical consequence: **variance is high**. A 30-rollout subset gives a noisy estimate; the paper
  averages 1500 rollouts (3×500). Treat your subset number as ±several percent.

---

## 4. Running it

One-click: use `.vscode/launch.json` → "04 · LIBERO eval — spatial (quick 2×3)" or
"05 · Canonical repo eval". From a shell (note the EGL env vars):

```bash
# quick subset (our driver: adds JSON summary + action-trajectory plot)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  .venv-openvla/bin/python study_scripts/04_libero_eval_subset.py \
  --task_suite_name libero_spatial --num_tasks 2 --num_trials_per_task 3

# canonical full reproduction (10 tasks × 50 = 500 rollouts, hours on a 3090)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  .venv-openvla/bin/python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial --center_crop True
```

Runtime math on a 3090 with SDPA: ~250 ms/action × up to 220 steps ≈ up to ~55 s/rollout (successes
end early). 30 rollouts ≈ 10–15 min; the full 500 ≈ several hours. To scale toward the paper number,
increase `--num_trials_per_task` (our driver) or run the canonical script overnight.

Outputs:
- `rollouts/<date>/...--success=<bool>--task=...mp4` — watch these! Success and failure look obviously
  different and teach you the model's behaviour.
- `study_outputs/libero_eval_<suite>.json` — machine-readable per-episode results + success rate.
- `study_outputs/libero_action_trajectory_<suite>.png` — the 7 action dims over one rollout.

---

## 5. Reading the results / debugging low success

- **Watch a failure video** next to a success. Typical VLA failure modes:
  - *Getting "stuck"* (repeats tiny/idle actions) — OpenVLA has no action chunking and is sensitive to
    idle actions (see README "VLA Performance Troubleshooting").
  - *Gripper mistiming* — closes too early/late; check the gripper sign handling.
  - *Depth/precision misses* — no explicit 3D prior (see `docs/02_ARCHITECTURE.md` §5), so fine
    alignment is hard from a single view.
- **Sanity before blaming the model** (from the README): replay a demo's actions in the env (should
  succeed → data/env pipeline OK); feed training images into your inference pipeline and check token
  accuracy (→ inference pipeline OK). Only then suspect the policy/data.
- **`unnorm_key`** must match the checkpoint's stats key (here `libero_spatial`). Wrong key →
  mis-scaled actions → silent failure. Our driver auto-tries the `_no_noops` suffix.

---

## 6. Our reproduced numbers (this machine — RTX 3090, SDPA, mujoco 3.x)

**All four LIBERO suites reproduced** (each with its fine-tuned checkpoint, `center_crop=True`):

| Suite | Ours | Paper | Rollouts (tasks × trials) |
|---|---|---|---|
| LIBERO-Spatial | **81.0%** | 84.7% | 100 (10 × 10) |
| LIBERO-Object | **80.0%** | 88.4% | 50 (10 × 5) |
| LIBERO-Goal | **82.0%** | 79.2% | 50 (10 × 5) |
| LIBERO-Long (10) | **53.3%** | 53.7% | 30 (10 × 3) |
| **Average** | **74.1%** | **76.5%** | 230 total |

Reads:
- **LIBERO-Long matches almost exactly** (53.3 vs 53.7) and **Goal exceeds the paper** (82.0 vs 79.2).
  Spatial/Object sit a few points under — smaller samples + 3090/mujoco-3.x/SDPA vs A100/mujoco-2.3.x/
  flash-attn. The **average (74.1% vs 76.5%)** is a convincing whole-benchmark reproduction.
- **Variance lesson** (Spatial sample-size sweep): 4 rollouts → 100%, 30 → 70.0%, **100 → 81.0%**.
  Closed-loop success rate is high-variance; small subsets under-report (this is §3 in action).
- Per-suite/per-task breakdowns and all rollout videos: `study_outputs/libero_eval_libero_*.json`
  and `rollouts/<date>/`. Trajectory plots: `study_outputs/libero_action_trajectory_*.png`.
