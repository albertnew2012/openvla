"""
eval_widowx.py  —  Run openvla-7b ZERO-SHOT on SimplerEnv WidowX/Bridge tasks (no fine-tuning).

This is the payoff: openvla-7b was trained on BridgeData (WidowX) as part of Open-X, so it controls
these tasks out of the box — unlike LIBERO (Franka), which needed fine-tuning. Closed-loop, headless.

Run (in .venv-simpler):
    .venv-simpler/bin/python study_scripts/simplerenv/eval_widowx.py \
        --tasks widowx_carrot_on_plate widowx_put_eggplant_in_basket --episodes 3
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# init torch CUDA before anything that might import TF-ish libs (defensive; SimplerEnv widowx path has no TF)
import torch  # noqa: E402

if torch.cuda.is_available():
    torch.zeros(1, device="cuda:0")

import imageio  # noqa: E402
import simpler_env  # noqa: E402
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402

from openvla_policy import OpenVLABridgeInference  # noqa: E402

WIDOWX_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def run_episode(env, model, instruction, max_steps, save_path):
    obs, _ = env.reset()
    instruction = env.get_language_instruction()
    model.reset(instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    frames = [image]
    success, truncated, done, t = False, False, False, 0
    while not (truncated or done) and t < max_steps:
        _, action = model.step(image, instruction)
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        success = success or bool(info.get("success", False))
        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image)
        t += 1
        if success:
            break
    if save_path:
        imageio.mimsave(save_path, frames, fps=10)
    return success, len(frames), instruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--unnorm_key", default="bridge_orig")
    ap.add_argument("--tasks", nargs="+", default=["widowx_carrot_on_plate"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    if args.tasks == ["all"]:
        args.tasks = WIDOWX_TASKS

    model = OpenVLABridgeInference(saved_model_path=args.model, unnorm_key=args.unnorm_key,
                                   policy_setup="widowx_bridge", image_size=args.image_size)

    os.makedirs("study_outputs/simpler_rollouts", exist_ok=True)
    results, total, succ = [], 0, 0
    for task in args.tasks:
        env = simpler_env.make(task)
        for ep in range(args.episodes):
            vid = f"study_outputs/simpler_rollouts/{task}--ep{ep}.mp4"
            ok, nsteps, instr = run_episode(env, model, task, args.max_steps, vid)
            total += 1
            succ += int(ok)
            results.append({"task": task, "episode": ep, "success": bool(ok), "steps": nsteps,
                            "instruction": instr})
            print(f"  {task} ep{ep}: success={ok} steps={nsteps} running={succ}/{total} "
                  f"({100*succ/total:.0f}%)  instr={instr!r}")
        env.close()

    sr = succ / max(total, 1)
    summary = {"model": args.model, "unnorm_key": args.unnorm_key, "total": total, "successes": succ,
               "success_rate": sr, "per_episode": results}
    with open("study_outputs/simpler_eval_widowx.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[RESULT] WidowX-Bridge ZERO-SHOT: {succ}/{total} = {100*sr:.1f}% "
          f"over tasks {args.tasks} x {args.episodes} eps")
    print(f"[RESULT] summary -> study_outputs/simpler_eval_widowx.json ; videos -> study_outputs/simpler_rollouts/")


if __name__ == "__main__":
    main()
