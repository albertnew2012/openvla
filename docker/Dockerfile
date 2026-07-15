# OpenVLA — reproducible GPU container (mirrors ../../ubuntu2404/docker structure & conventions).
#
# WHY Ubuntu 22.04 / Python 3.10 here (ubuntu2404 uses 24.04/py3.12):
#   OpenVLA is pinned to Python 3.10 / torch 2.2 / transformers 4.40.1 / timm 0.9.10, and its
#   training data pipeline (tensorflow==2.15, tensorflow_datasets, dlimp, mujoco 2.3.x) only installs
#   cleanly on Python 3.10. This base IS OpenVLA's officially tested environment, so the whole
#   project runs out of the box — inference, LIBERO sim eval, and LoRA/QLoRA *and* full RLDS
#   fine-tuning — with none of the py3.12 workarounds a 24.04 image would need (see docs/04_REPRO_LOG.md).
#   Override with --build-arg CUDA_IMAGE=... if you must match a specific driver/toolkit.
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USER_NAME=defaultuser
ARG USER_ID=1000
ARG GROUP_ID=1000

# --- PyTorch stack, pinned to OpenVLA's tested versions (cu121 wheels; they run on newer drivers
#     via CUDA minor-version compatibility, exactly as verified on the host RTX 3090). ---
ARG TORCH_CUDA=cu121
ARG TORCH_VERSION=2.2.0
ARG TORCHVISION_VERSION=0.17.0
ARG TORCHAUDIO_VERSION=2.2.0
# Build Flash-Attention 2 (slow, ~20-30 min, needs the devel/nvcc base). Set "false" to skip;
# the code falls back to PyTorch SDPA (numerically equivalent for correctness on Ampere).
ARG INSTALL_FLASH_ATTN=true

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video

# Headless MuJoCo/robosuite rendering on the GPU via EGL (no X server). The `graphics` capability
# above + the libglvnd/libEGL packages below are what make LIBERO render inside a container.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

# Install into system site-packages (no venv), like ubuntu2404.
ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# --- System packages: build tools (cmake/ninja for egl_probe & flash-attn), codecs, and the
#     EGL / OpenGL (GLVND) stack — the piece the OpenVLA README omits. MuJoCo needs the
#     vendor-neutral libEGL.so.1 dispatch (libglvnd0/libegl1), not just the NVIDIA vendor lib. ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config patchelf \
    ca-certificates curl wget git git-lfs gnupg locales sudo lsb-release \
    ffmpeg vim htop unzip zip rsync net-tools iputils-ping \
    libffi-dev libssl-dev libglib2.0-0 libsndfile1 \
    python3 python3-dev python3-pip python-is-python3 python3-venv python3-tk \
    libegl1 libgles2 libglvnd0 libgl1 libglx0 libopengl0 libgl1-mesa-dri libglfw3 libosmesa6 && \
    git lfs install --system && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip/setuptools from the Debian package lack a RECORD file, so an in-place upgrade fails;
# --ignore-installed installs fresh copies into /usr/local that shadow the Debian ones.
RUN python3 -m pip install --upgrade --ignore-installed pip setuptools wheel ninja packaging

# --- PyTorch (CUDA-matched index) ---
RUN python3 -m pip install \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

# --- OpenVLA core stack (pinned to the paper). numpy<2 (torch 2.2 predates the numpy 2.0 ABI);
#     accelerate 0.30.1 is the version whose 4-bit dispatch works with transformers 4.40.1
#     (newer accelerate calls .to() on 4-bit models and breaks QLoRA — see docs/04_REPRO_LOG.md). ---
RUN python3 -m pip install \
        "numpy==1.26.4" "scipy==1.13.1" \
        "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
        "accelerate==0.30.1" "peft==0.11.1" "bitsandbytes==0.43.1" \
        "draccus==0.8.0" "sentencepiece==0.1.99" \
        json-numpy jsonlines einops rich matplotlib protobuf \
        imageio imageio-ffmpeg huggingface_hub wandb

# --- RLDS / Open-X training data pipeline (installs cleanly on py3.10). tensorflow-cpu so TF's
#     image ops don't contend with PyTorch for VRAM at eval; the tf.data pipeline is CPU-bound so
#     this also serves training. dlimp with --no-deps to skip its exact tensorflow pin. ---
RUN python3 -m pip install "tensorflow-cpu==2.15.0" "tensorflow_datasets==4.9.3" && \
    (python3 -m pip install "tensorflow_graphics==2021.12.3" || \
        echo "[warn] tensorflow_graphics skipped (only needed for DROID transforms) — continuing") && \
    python3 -m pip install --no-deps "git+https://github.com/moojink/dlimp_openvla"

# --- LIBERO simulation benchmark. Pin mujoco==2.3.7 (has py3.10 wheels AND keeps the mj_fullM
#     signature robosuite 1.4.1 expects → no source patch needed, unlike mujoco 3.x on py3.12).
#     LIBERO is installed non-editable into site-packages so `import libero` + its bundled
#     bddl/asset data work self-contained. ---
RUN python3 -m pip install \
        "mujoco==2.3.7" "robosuite==1.4.1" bddl easydict cloudpickle "gym==0.25.2" "imageio[ffmpeg]" && \
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO && \
    python3 -m pip install /opt/LIBERO --no-deps && \
    python3 -m pip install "numpy==1.26.4"   # reassert (robosuite/mujoco may bump numpy)

# --- (optional) Flash-Attention 2 — faster training/inference; SDPA is used automatically if absent. ---
RUN if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
        python3 -m pip install "flash-attn==2.5.5" --no-build-isolation || \
        echo "[warn] flash-attn build failed — code falls back to attn_implementation='sdpa'"; \
    fi

# --- Non-root user mapped to the host UID/GID (so bind-mounted files keep correct ownership). ---
RUN existing_user="$(getent passwd "${USER_ID}" | cut -d: -f1 || true)" && \
    existing_group="$(getent group "${GROUP_ID}" | cut -d: -f1 || true)" && \
    if [ -n "${existing_group}" ] && [ "${existing_group}" != "${USER_NAME}" ]; then \
      groupmod -n "${USER_NAME}" "${existing_group}"; \
    elif [ -z "${existing_group}" ]; then \
      groupadd -g "${GROUP_ID}" "${USER_NAME}"; \
    fi && \
    if [ -n "${existing_user}" ] && [ "${existing_user}" != "${USER_NAME}" ]; then \
      usermod -l "${USER_NAME}" -d "/home/${USER_NAME}" -m -s /bin/bash "${existing_user}"; \
    elif [ -z "${existing_user}" ]; then \
      useradd -l -u "${USER_ID}" -g "${GROUP_ID}" -m -s /bin/bash "${USER_NAME}"; \
    fi && \
    usermod -u "${USER_ID}" -g "${GROUP_ID}" -s /bin/bash "${USER_NAME}" && \
    usermod -aG sudo "${USER_NAME}" && \
    mkdir -p /home/${USER_NAME}/.local && \
    chown -R "${USER_ID}:${GROUP_ID}" /home/${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME}

# --- Build-time verification: fail early if the core stack is broken. (Sim imports are wrapped
#     because docker build has no GPU/EGL device — verify those at runtime with study_scripts/02.) ---
RUN python3 --version && (nvcc --version | tail -1 || true) && \
    python3 -c "import torch, torchvision; print('torch', torch.__version__, '| torch-cuda', torch.version.cuda)" && \
    python3 -c "import transformers, timm, tokenizers, peft, bitsandbytes, accelerate; \
print('transformers', transformers.__version__, '| timm', timm.__version__, '| tokenizers', tokenizers.__version__, \
'| peft', peft.__version__, '| bnb', bitsandbytes.__version__, '| accelerate', accelerate.__version__)" && \
    python3 -c "import tensorflow as tf, tensorflow_datasets; print('tensorflow-cpu', tf.__version__, '+ tfds OK')" && \
    python3 -c "import draccus, dlimp, json_numpy; print('draccus + dlimp + json_numpy OK')" && \
    (python3 -c "import robosuite, mujoco, libero; print('robosuite', robosuite.__version__, '| mujoco', mujoco.__version__, '| LIBERO import OK')" || \
        echo "[warn] sim import needs EGL/GPU at runtime; verify inside the container with study_scripts/02_libero_render_smoketest.py")

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD ["bash"]
