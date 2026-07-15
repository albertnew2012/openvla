# OpenVLA Docker

A reproducible GPU container with **everything the project needs** — inference, LIBERO simulation
eval (headless EGL rendering), and LoRA/QLoRA **plus** full RLDS fine-tuning. Structure mirrors
`~/Desktop/ubuntu2404/docker` (Dockerfile + build/create scripts, host-UID user mapping, home mount).

## Quick start

```bash
cd ~/Desktop/openvla
bash docker/build_image.sh          # builds image  openvla:<your-username>   (one-time, ~15-40 min)
bash docker/create_container.sh     # creates + enters container  openvla_<your-username>
```

Inside the container (you land in the repo, `openvla`/`prismatic` already installed):

```bash
# 1) inference (downloads openvla-7b to ~/.cache/huggingface the first time; reused from the mount)
python study_scripts/01_minimal_inference.py

# 2) headless render check, then a LIBERO eval + a QLoRA fine-tune demo
python study_scripts/02_libero_render_smoketest.py
python study_scripts/04_libero_eval_subset.py --num_tasks 2 --num_trials_per_task 3
python study_scripts/05_lora_finetune_3090.py --max_steps 60 --batch_size 2

# 3) the canonical eval / training entrypoints also work as-is:
python experiments/robot/libero/run_libero_eval.py --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial --center_crop True
```

## What's baked in

- **CUDA 12.1 + cuDNN**, Ubuntu 22.04, **Python 3.10** (OpenVLA's tested interpreter).
- **PyTorch 2.2.0 (cu121)** + torchvision/torchaudio.
- **OpenVLA stack pinned to the paper**: transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10,
  peft 0.11.1, bitsandbytes 0.43.1, accelerate 0.30.1, draccus 0.8.0, numpy 1.26.4, scipy 1.13.1.
- **RLDS training pipeline**: tensorflow-cpu 2.15, tensorflow_datasets 4.9.3, dlimp.
- **LIBERO sim**: robosuite 1.4.1 + **mujoco 2.3.7** + LIBERO (installed), and the **EGL/GLVND**
  system libs (`libegl1`, `libglvnd0`, …) that make MuJoCo render headlessly on the GPU.
- **Flash-Attention 2.5.5** (optional; `--build-arg INSTALL_FLASH_ATTN=false` to skip — SDPA is the
  automatic fallback).

## Why Python 3.10 (not 24.04/py3.12 like `ubuntu2404`)

OpenVLA's training deps (`tensorflow==2.15`, `tensorflow_datasets`, `dlimp`, `mujoco 2.3.x`) install
cleanly only on Python 3.10, and mujoco 2.3.x keeps the `mj_fullM` API robosuite 1.4.1 expects. On
24.04/py3.12 you hit a chain of incompatibilities (protobuf/tfds, `np.long`, mujoco-3 API, …) that my
host venv worked around with ~10 patches — see [../docs/04_REPRO_LOG.md](../docs/04_REPRO_LOG.md).
This image sidesteps all of them by using OpenVLA's supported interpreter. To force a different
base: `docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04 ...`.

## Notes

- **GPU + rendering**: needs the NVIDIA Container Toolkit on the host (`--gpus all`). EGL rendering
  works because `NVIDIA_DRIVER_CAPABILITIES` includes `graphics` and the GLVND libs are installed.
- **Model weights are not baked in** (they're ~15 GB each). They download to `~/.cache/huggingface`,
  which is shared via the home mount — so anything already downloaded on the host is reused.
- **Persistence**: `~/.local` is a named volume, so the editable `openvla` install (and any
  `pip install --user` packages) persist across `create_container.sh` runs.
- Re-running `create_container.sh` re-attaches to the existing container instead of erroring.
