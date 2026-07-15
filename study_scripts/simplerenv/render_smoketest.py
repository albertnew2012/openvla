"""
render_smoketest.py  —  Prove SAPIEN/Vulkan headless rendering + bundled assets work in SimplerEnv,
BEFORE loading the 15 GB policy. Creates one WidowX-Bridge task, renders the 3rd-person camera, dumps a PNG.

Run (in .venv-simpler):
    .venv-simpler/bin/python study_scripts/simplerenv/render_smoketest.py
"""

import os

import numpy as np
from PIL import Image

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def main():
    task = os.environ.get("SIMPLER_TASK", "widowx_carrot_on_plate")
    print(f"[*] creating SimplerEnv task: {task}")
    env = simpler_env.make(task)
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print(f"[*] robot_uid: {env.robot_uid}")
    print(f"[*] language instruction: {instruction!r}")

    image = get_image_from_maniskill2_obs_dict(env, obs)  # (H,W,3) uint8
    os.makedirs("study_outputs", exist_ok=True)
    out = "study_outputs/simpler_render_smoketest.png"
    Image.fromarray(image).save(out)
    print(f"[RESULT] rendered {image.shape} frame -> {out}")
    print(f"[RESULT] pixel range [{image.min()},{image.max()}] mean {image.mean():.1f} "
          f"({'NON-BLANK, Vulkan render OK' if image.std() > 1 else 'BLANK! render FAILED'})")
    env.close()


if __name__ == "__main__":
    main()
