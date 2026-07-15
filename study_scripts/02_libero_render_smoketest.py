"""
02_libero_render_smoketest.py  —  Prove headless MuJoCo/robosuite rendering works BEFORE
we spend 15 GB downloading a policy. Isolates the #1 failure mode (offscreen GL) from the model.

It spins up ONE LIBERO task, steps a few no-op actions, and dumps the agentview frame to a PNG.

Run:
    MUJOCO_GL=egl .venv-openvla/bin/python study_scripts/02_libero_render_smoketest.py
"""

import os

# EGL = headless GPU rendering (we verified libEGL_nvidia + nvidia glvnd vendor exist).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    print(f"[*] MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"[*] libero bddl path: {get_libero_path('bddl_files')}")

    suite = "libero_spatial"
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(0)
    print(f"[*] suite={suite}  n_tasks={bench.n_tasks}")
    print(f"[*] task 0 language: {task.language!r}")

    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    init_states = bench.get_task_init_states(0)
    env.set_init_state(init_states[0])

    obs = None
    for _ in range(10):  # let objects settle
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    img = obs["agentview_image"][::-1, ::-1]  # same 180° flip the eval uses
    os.makedirs("study_outputs", exist_ok=True)
    out = "study_outputs/libero_render_smoketest.png"
    Image.fromarray(img).save(out)
    print(f"[RESULT] rendered agentview frame {img.shape} -> {out}")
    print(f"[RESULT] pixel range [{img.min()},{img.max()}] mean {img.mean():.1f} "
          f"({'NON-BLANK, render OK' if img.std() > 1 else 'BLANK! render FAILED'})")
    env.close()


if __name__ == "__main__":
    main()
