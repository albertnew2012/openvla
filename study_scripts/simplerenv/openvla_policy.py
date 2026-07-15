"""
openvla_policy.py  —  OpenVLA policy adapter for SimplerEnv (WidowX / Bridge).

SimplerEnv ships rt1 + octo policy wrappers but NOT openvla, so this is a thin adapter that gives
openvla-7b the same `.step(image, instruction) -> (raw_action, action)` interface SimplerEnv expects.
The action conversion (world_vector, euler->axis-angle rotation, sticky gripper) mirrors SimplerEnv's
own Octo wrapper for `policy_setup="widowx_bridge"` (action_scale=1.0, sticky_gripper_num_repeat=1).

OpenVLA's bridge gripper is mask=False → passed through in [0,1] (1=open, 0=close), i.e. the SAME
convention as Octo's `open_gripper`, so the sticky-gripper logic transfers verbatim.

Runs in .venv-simpler (py3.10). Uses SDPA if flash-attn isn't present.
"""

import importlib.util

import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor


class OpenVLABridgeInference:
    def __init__(
        self,
        saved_model_path: str = "openvla/openvla-7b",
        unnorm_key: str = "bridge_orig",
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        image_size: int = 256,  # resize sim frame to this square before the processor (bridge native)
    ):
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
        print(f"[OpenVLA] loading {saved_model_path} (attn={attn}, bf16) ...")
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.vla = (
            AutoModelForVision2Seq.from_pretrained(
                saved_model_path, attn_implementation=attn, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, trust_remote_code=True,
            ).to("cuda:0").eval()
        )
        self.unnorm_key = unnorm_key
        self.action_scale = action_scale
        self.image_size = image_size
        if policy_setup != "widowx_bridge":
            raise NotImplementedError("This adapter is set up for widowx_bridge.")
        self.sticky_gripper_num_repeat = 1  # widowx_bridge (Octo uses 15 for google_robot)
        self.reset("")

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    @torch.no_grad()
    def step(self, image: np.ndarray, task_description=None):
        """image: (H,W,3) uint8. Returns (raw_action, action) where action feeds SimplerEnv:
        np.concatenate([action['world_vector'], action['rot_axangle'], action['gripper']])."""
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        pil = Image.fromarray(image).convert("RGB")
        if self.image_size:
            pil = pil.resize((self.image_size, self.image_size), Image.LANCZOS)
        prompt = f"In: What action should the robot take to {self.task_description.lower()}?\nOut:"
        inputs = self.processor(prompt, pil).to("cuda:0", dtype=torch.bfloat16)
        raw = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)  # (7,)

        raw_action = {
            "world_vector": np.asarray(raw[:3], dtype=np.float64),
            "rotation_delta": np.asarray(raw[3:6], dtype=np.float64),
            "open_gripper": np.asarray(raw[6:7], dtype=np.float64),  # [0,1]; 1=open, 0=close
        }
        action = {"world_vector": raw_action["world_vector"] * self.action_scale}
        roll, pitch, yaw = raw_action["rotation_delta"]
        ax, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = (ax * angle * self.action_scale).astype(np.float64)

        # --- sticky gripper (mirrors SimplerEnv Octo widowx_bridge) ---
        current = raw_action["open_gripper"]
        if self.previous_gripper_action is None:
            relative = np.array([0.0])
        else:
            relative = self.previous_gripper_action - current
        self.previous_gripper_action = current
        if np.abs(relative) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        action["gripper"] = np.asarray(relative, dtype=np.float64).reshape(1)
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
