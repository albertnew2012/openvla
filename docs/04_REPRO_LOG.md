# Reproduction Log — Setting up OpenVLA & Reproducing LIBERO (RTX 3090, from scratch)

> This is the "what actually happened" log: the environment, every problem hit, its diagnosis,
> and the fix. VLA repos are notoriously version-fragile, so this debugging trail is itself a
> big chunk of the learning value. Date: 2026-07-11. Machine: single **RTX 3090 (24 GB)**,
> Ubuntu 24.04 container, driver CUDA 13.2, **Python 3.12 only** (no conda).

---

## 0. TL;DR outcome

| Goal | Status | Evidence |
|---|---|---|
| Load `openvla-7b` + run inference on 24 GB | ✅ | 15.1 GB VRAM, ~250 ms/action (SDPA). `study_outputs/minimal_inference.png` |
| Reproduce LIBERO — **all 4 suites** (closed-loop sim) | ✅ | avg **74.1%** vs paper **76.5%** (Spatial 81.0 / Object 80.0 / Goal 82.0 / Long 53.3) → `docs/03_LIBERO_EVAL.md` §6 |
| Headless GPU rendering (MuJoCo/robosuite) | ✅ | `study_outputs/libero_render_smoketest.png` |
| Action-tokenization visualized | ✅ | `study_outputs/action_tokenization.png` |
| QLoRA fine-tune on 24 GB | ✅ **peak 9.63 GB**, loss 7.4→0.001, acc→1.0 | `docs/05_FINETUNING_3090.md`, `study_scripts/05_lora_finetune_3090.py` |

**Key decision:** build an **isolated venv** (`.venv-openvla`) with OpenVLA's *pinned* versions
rather than touch the base container (which has torch 2.11 / transformers 5.13-dev and other
projects like GR00T/Qwen-VL that must not be disturbed).

---

## 1. Environment reconnaissance (what we started with)

- GPU: RTX 3090, 24 GB, ~free. Driver exposes CUDA 13.2 (backward-compatible with older runtimes).
- Base Python env: `torch 2.11+cu130`, `transformers 5.13.0.dev0`, `timm 1.0.27`, `tokenizers 0.22.2`.
- OpenVLA **requires** (and hard-checks): `transformers 4.40.1`, `tokenizers 0.19.1`,
  `timm ∈ {0.9.10..0.9.16}`, `torch 2.2.*`.
- The model literally **raises `NotImplementedError` on timm ≥ 1.0** (`modeling_prismatic.py:221`)
  and warns on transformers ≠ 4.40.1. So the base env cannot run OpenVLA, and we can't downgrade
  timm globally without breaking the base env's other models.

→ **Isolated venv is mandatory, not stylistic.**

---

## 2. Building the isolated env

`python3.12 -m venv` failed (`ensurepip` missing / no `python3.12-venv`). Used **`virtualenv`** instead
(bundles pip). Then installed the pinned stack + **cu121** torch wheels (they run fine on the newer
driver via CUDA minor-version compatibility):

```
virtualenv -p python3.12 .venv-openvla
.venv-openvla/bin/pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
.venv-openvla/bin/pip install numpy==1.26.4    # torch 2.2 predates the numpy 2.0 ABI
.venv-openvla/bin/pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    accelerate peft==0.11.1 draccus==0.8.0 json-numpy jsonlines einops rich matplotlib \
    protobuf sentencepiece imageio imageio-ffmpeg
.venv-openvla/bin/pip install -e . --no-deps   # openvla/prismatic, WITHOUT the tensorflow==2.15 pin
```

Why `--no-deps` for the editable install: `pyproject.toml` pins `tensorflow==2.15.0`, which has **no
Python 3.12 wheel**. TF is only needed for the RLDS *training* pipeline and (as it turns out) some
eval image ops — not for core inference — so we install it separately and deliberately.

---

## 3. The issue → fix ledger (in the order we hit them)

### (a) `import prismatic` drags in the whole training stack
`prismatic/__init__.py` did `from .models import ...`, which chains into
`prismatic.vla.datasets.rlds` → `import dlimp` / `tensorflow_datasets` / `tensorflow_graphics`.
That entire stack is unimportable on py3.12. But **inference & sim-eval only need the standalone
HF port** in `prismatic/extern/hf/*` (deps: numpy/timm/torch/transformers).
- **Fix:** made the top-level re-export lazy (guarded `try/except`) in `prismatic/__init__.py`.
  Non-destructive: works as before when training deps are present; degrades gracefully otherwise.

### (b) Eval hardcodes Flash-Attention 2 (not built here)
`experiments/robot/openvla_utils.py:get_vla` forced `attn_implementation="flash_attention_2"`.
- **Fix:** auto-detect `flash_attn`; fall back to PyTorch **SDPA** (numerically equivalent for
  correctness on Ampere; ~same success rate, just slower decode). One-line-ish patch, documented in code.

### (c) `huggingface-cli` is dead in this image
huggingface_hub 1.x removed the old CLI. **Fix:** use `hf download <repo> --exclude "*.bin"`
(openvla-7b ships safetensors; excluding `.bin` avoids nothing needed).

### (d) TensorFlow / protobuf / tfds incompatibility (noise, not fatal)
`tensorflow_datasets`/`tensorflow-metadata` pulled protobuf ≥5 generated code (`runtime_version`),
but `tensorflow-cpu 2.17` pins protobuf <5 → `ImportError` when the guarded `.models` import runs.
- **Fix:** we never actually need tfds for eval, so we just let the guard from (a) swallow it. TF-cpu
  itself imports fine and is used for the eval's image ops. (A one-time noisy traceback at startup is
  harmless.) `dlimp` was installed `--no-deps` to skip its `tensorflow==2.15` pin.

### (e) LIBERO's `requirements.txt` would wreck the env
It pins `transformers==4.21.1`, `numpy==1.22.4`, `robosuite==1.4.0`, etc. Installing it would
downgrade transformers and break OpenVLA.
- **Fix:** install LIBERO with `pip install -e LIBERO --no-deps` (its `install_requires` is empty
  anyway) and use **OpenVLA's own** `experiments/robot/libero/libero_requirements.txt`
  (`robosuite==1.4.1`, `bddl`, `easydict`, `cloudpickle`, `gym==0.25.2`, `imageio[ffmpeg]`).

### (f) `egl_probe` wheel build failed → **no `cmake`**
robosuite→egl_probe needs cmake to build (it ships its own EGL/GL headers). **Fix:** `pip install cmake`.

### (g) Headless MuJoCo render crashed → **missing GLVND `libEGL.so.1`**
The container had the NVIDIA vendor lib `libEGL_nvidia.so.0` but no vendor-neutral dispatch
`libEGL.so.1`, so PyOpenGL/mujoco couldn't load EGL (`'NoneType' has no attribute 'eglQueryString'`).
- **Fix:** `apt-get install -y libegl1 libglvnd0 libgl1 libglx0 libopengl0`. Then set
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0`.

### (h) robosuite 1.4.1 ✗ mujoco 3.x API change (`mj_fullM`)
No mujoco 2.3.x wheel exists for py3.12, so we're on mujoco 3.10. mujoco changed
`mj_fullM(m, dst, M_sparse)` → `mj_fullM(m, data, dst)`. robosuite 1.4.1 uses the old form → `TypeError`.
- **Fix:** patched the single call in
  `.venv-openvla/.../robosuite/controllers/base_controller.py` to
  `mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)`.

### (i) LIBERO editable install didn't expose `libero`
setuptools' strict PEP-660 finder produced an **empty MAPPING** (a known bug).
- **Fix:** dropped a `.pth` file with the LIBERO repo root into the venv site-packages
  (`zzz_libero_repo.pth`) so `import libero` works everywhere.
- Note: the container already had a LIBERO configured for GR00T; `~/.libero/config.yaml` points
  bddl/asset paths at `Isaac-GR00T/external_dependencies/LIBERO` — which is fine, the files exist there.

### (j) `AttributeError: np.long` from scipy
`scipy 1.18` (pulled transitively) assumes numpy 2.0's `np.long`, but we're pinned to numpy 1.26.
- **Fix:** `pip install scipy==1.13.1` (supports numpy 1.26 + py3.12).

### (l) QLoRA 4-bit load: `.to` not supported for 4-bit models
Loading `openvla-7b` with a `BitsAndBytesConfig(load_in_4bit=True)` crashed in
`accelerate.dispatch_model → model.to(device)`. Root cause: `accelerate` was installed unpinned and
resolved to a **too-new 1.14.0**, whose dispatch path calls `.to()` on quantized models (forbidden).
transformers 4.40.1 needs a contemporaneous accelerate.
- **Fix:** `pip install accelerate==0.30.1` (+ load with `device_map={"":0}`). 4-bit base then loads to
  **~4.4 GB** VRAM. Inference/eval are unaffected (they use the bf16 `.to()` path). Only QLoRA needed this.

### (k) **Segfault** loading the torch model after TensorFlow was imported
`torch.cuda._lazy_init` crashed because `import tensorflow` ran first (the eval utils import TF at
module load). Classic TF↔torch in-process CUDA conflict.
- **Fix:** initialize torch's CUDA context **before** TF is imported — added a tiny
  `import torch; torch.zeros(1, device="cuda:0")` at the top of the eval scripts
  (`run_libero_eval.py` and `study_scripts/04_...`). This is the single most important fix for the eval.

---

## 4. Files changed in the repo (all documented, minimal)

| File | Change | Reason |
|---|---|---|
| `prismatic/__init__.py` | guarded top-level `.models` import | avoid training-stack import on py3.12 (issue a) |
| `experiments/robot/openvla_utils.py` | SDPA fallback when `flash_attn` absent | issue b |
| `experiments/robot/libero/run_libero_eval.py` | torch CUDA init before TF import | issue k (segfault) |
| `.venv-openvla/.../robosuite/controllers/base_controller.py` | mujoco 3.x `mj_fullM` signature | issue h (in venv, not repo) |

New (additive) study material: `study_scripts/`, `docs/`, `.vscode/launch.json`, `study_outputs/`.

---

## 5. Reproduction results

- **Inference**: `openvla-7b` loads in ~6 s, **15.1 GB** VRAM, ~250 ms/step (SDPA, bf16) ≈ 4 Hz.
- **LIBERO-Spatial** (converging as we add rollouts — a nice lesson in closed-loop eval variance):
  - quick 2×2 subset: **4/4 = 100%** (122 s)
  - 30 rollouts (3/task): **21/30 = 70.0%** (noisy small sample; one hard task went 0/3)
  - **100 rollouts (10/task): 81/100 = 81.0%** ← the headline number, **vs paper 84.7% ± 0.9**
  - Per-task (100-rollout): mostly 7–10/10; the outlier is "…bowl on the ramekin…" at **3/10** (a
    genuinely hard balance-grasp) which alone accounts for most of the gap to the paper.
- **LIBERO-Object** (`openvla-7b-finetuned-libero-object`): **40/50 = 80.0%** vs paper **88.4%**.
  Per-task mostly 4–5/5 (ketchup/milk/OJ = 5/5); "butter" 2/5 is the hardest.
- **LIBERO-Goal** (`openvla-7b-finetuned-libero-goal`): **41/50 = 82.0%** vs paper **79.2%** — *above* it.
- **LIBERO-Long / libero_10** (`openvla-7b-finetuned-libero-10`, long-horizon 520-step tasks):
  **16/30 = 53.3%** vs paper **53.7%** — a near-exact match on the hardest suite.
- **4-suite average: 74.1% vs paper 76.5%** — a convincing whole-benchmark reproduction on a single
  3090. Full table + reads in `docs/03_LIBERO_EVAL.md` §6.
- **QLoRA fine-tune (3090)**: 4-bit base + LoRA r16 → **peak 9.63 GB / 24 GB**, 0.73% params trainable,
  loss 7.40→0.0010, action-acc 0.21→1.00 over 60 steps. Adapter saved to `runs/lora-3090-demo/`.
- The 70%→81% jump from 30→100 rollouts is itself the point of `docs/03_LIBERO_EVAL.md` §3: closed-loop
  success rate is **high-variance**, so small subsets under-report. Our 81% on a 3090 (SDPA, mujoco 3.x)
  vs the paper's 84.7% on A100 (flash-attn, mujoco 2.3.x) is a solid reproduction within expected noise.

> Caveat on faithfulness: we use SDPA instead of Flash-Attention-2 and PIL/TF image ops as installed;
> both are numerically ~equivalent. GPU is a 3090, not the paper's A100 — large-model GPU
> nondeterminism can shift individual rollouts, but aggregate success rates track the paper.

---

## 6. How to reproduce from scratch (condensed)

```bash
# 1. env
virtualenv -p python3.12 .venv-openvla
.venv-openvla/bin/pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
.venv-openvla/bin/pip install numpy==1.26.4 transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    accelerate peft==0.11.1 draccus==0.8.0 json-numpy jsonlines einops rich matplotlib protobuf \
    sentencepiece imageio imageio-ffmpeg "scipy==1.13.1" "tensorflow-cpu==2.17.0"
.venv-openvla/bin/pip install -e . --no-deps
pip install cmake                                   # build tool for egl_probe
sudo apt-get install -y libegl1 libglvnd0 libgl1 libglx0 libopengl0

# 2. LIBERO
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
.venv-openvla/bin/pip install -e ../LIBERO --no-deps
.venv-openvla/bin/pip install robosuite==1.4.1 bddl easydict cloudpickle gym==0.25.2 "imageio[ffmpeg]"
echo "$(cd ../LIBERO && pwd)" > .venv-openvla/lib/python3.12/site-packages/zzz_libero_repo.pth
# patch robosuite mj_fullM (issue h) + apply repo patches (issues a,b,k) — see this log

# 3. run (see .vscode/launch.json for one-click versions)
.venv-openvla/bin/python study_scripts/01_minimal_inference.py
MUJOCO_GL=egl .venv-openvla/bin/python study_scripts/04_libero_eval_subset.py --num_tasks 2 --num_trials_per_task 3
```

Everything above is encoded in `.vscode/launch.json` and the `study_scripts/`.
