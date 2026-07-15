"""
04_libero_eval_subset.py  —  Closed-loop LIBERO evaluation (configurable subset).

Faithfully mirrors experiments/robot/libero/run_libero_eval.py (same preprocessing, gripper
handling, wait-steps, per-suite max_steps) but adds:
  * --num_tasks / --num_trials_per_task to run a quick subset (the stock script always does 10x50),
  * a JSON results summary,
  * an action-trajectory plot for the first rollout (7 DoF over time).

Use it to reproduce a LIBERO success rate quickly, then scale up toward the paper's 500-trial number.

Run (spatial, 2 tasks x 3 trials):
    MUJOCO_GL=egl .venv-openvla/bin/python study_scripts/04_libero_eval_subset.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --num_tasks 2 --num_trials_per_task 3
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

# IMPORTANT: initialize torch's CUDA context BEFORE tensorflow is imported (libero_utils/openvla_utils
# import TF at module load). On this stack, importing TF first and then triggering torch's lazy CUDA
# init segfaults at `.to("cuda")` — which is why the model loads but then the process dies silently
# with no rollout/video. Touching CUDA here establishes the context first and avoids the crash.
import torch as _torch

if _torch.cuda.is_available():
    _torch.zeros(1, device="cuda:0")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle, save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action, get_image_resize_size, get_model, invert_gripper_action,
    normalize_gripper_action, set_seed_everywhere,
)

MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


def resolve_checkpoint(path_or_id):
    """If given an HF repo id, return the local cached snapshot dir (so dataset_statistics.json loads)."""
    if os.path.isdir(path_or_id):
        return path_or_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(path_or_id, local_files_only=True)
    except Exception:
        return path_or_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--task_suite_name", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=2, help="-1 = all tasks in suite")
    ap.add_argument("--num_trials_per_task", type=int, default=3)
    ap.add_argument("--center_crop", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    ckpt = resolve_checkpoint(args.pretrained_checkpoint)
    print(f"[*] checkpoint resolved to: {ckpt}")

    cfg = SimpleNamespace(
        model_family="openvla", pretrained_checkpoint=ckpt,
        load_in_8bit=False, load_in_4bit=False, center_crop=args.center_crop,
        unnorm_key=args.task_suite_name,
    )

    model = get_model(cfg)
    # match run_libero_eval's unnorm_key fallback (datasets saved with a "_no_noops" suffix)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, (
        f"unnorm_key {cfg.unnorm_key} not in {list(model.norm_stats)}")
    print(f"[*] using unnorm_key={cfg.unnorm_key}")
    processor = get_processor(cfg)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    resize_size = get_image_resize_size(cfg)
    n_tasks = suite.n_tasks if args.num_tasks < 0 else min(args.num_tasks, suite.n_tasks)
    max_steps = MAX_STEPS[args.task_suite_name]

    results, first_traj = [], None
    total, successes = 0, 0
    t_start = time.time()
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        print(f"\n=== task {task_id}: {task_description!r} ===")
        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            replay, traj = [], []
            t, done = 0, False
            while t < max_steps + 10:
                try:
                    if t < 10:  # wait for objects to settle
                        obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    img = get_libero_image(obs, resize_size)
                    replay.append(img)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate((obs["robot0_eef_pos"],
                                                 quat2axisangle(obs["robot0_eef_quat"]),
                                                 obs["robot0_gripper_qpos"])),
                    }
                    action = get_action(cfg, model, observation, task_description, processor=processor)
                    traj.append(action.copy())
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                    t += 1
                except Exception as e:
                    print(f"  [!] exception: {e}")
                    break
            total += 1
            successes += int(done)
            results.append({"task_id": task_id, "task": task_description,
                            "episode": ep, "success": bool(done), "steps": len(replay)})
            mp4 = save_rollout_video(replay, total, success=done, task_description=task_description)
            print(f"  ep {ep}: success={done} steps={len(replay)}  running={successes}/{total} "
                  f"({100*successes/total:.1f}%)  video={os.path.basename(mp4)}")
            if first_traj is None and len(traj) > 3:
                first_traj = (np.array(traj), task_description, bool(done))
        env.close()

    elapsed = time.time() - t_start
    sr = successes / max(total, 1)
    summary = {
        "checkpoint": args.pretrained_checkpoint, "task_suite": args.task_suite_name,
        "num_tasks": n_tasks, "num_trials_per_task": args.num_trials_per_task,
        "center_crop": args.center_crop, "total_episodes": total, "successes": successes,
        "success_rate": sr, "elapsed_sec": round(elapsed, 1), "per_episode": results,
    }
    os.makedirs("study_outputs", exist_ok=True)
    out_json = f"study_outputs/libero_eval_{args.task_suite_name}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[RESULT] {args.task_suite_name}: {successes}/{total} = {100*sr:.1f}% "
          f"success over {n_tasks} tasks x {args.num_trials_per_task} trials  ({elapsed/60:.1f} min)")
    print(f"[RESULT] summary -> {out_json}")

    # action-trajectory plot for the first rollout
    if first_traj is not None:
        traj, desc, ok = first_traj
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for d in range(traj.shape[1]):
                ax.plot(traj[:, d], label=labels[d], lw=1.3)
            ax.set_title(f"OpenVLA action trajectory (task: {desc[:60]!r}, success={ok})")
            ax.set_xlabel("control step"); ax.set_ylabel("un-normalized action")
            ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            fig.tight_layout()
            p = f"study_outputs/libero_action_trajectory_{args.task_suite_name}.png"
            fig.savefig(p, dpi=130, bbox_inches="tight")
            print(f"[RESULT] action trajectory plot -> {p}")
        except Exception as e:
            print(f"[!] trajectory plot skipped: {e}")


if __name__ == "__main__":
    main()
