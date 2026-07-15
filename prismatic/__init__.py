# NOTE (study setup, 2026-07): The original single line here was
#     from .models import available_model_names, available_models, get_model_description, load
# That eagerly pulls in the full training stack (prismatic.models -> prismatic.vla ->
# RLDS/TFDS/dlimp data pipeline), which cannot import on Python 3.12 due to a
# tensorflow_datasets / protobuf incompatibility. Inference & simulation eval only need
# the standalone HF port under `prismatic.extern.hf.*` (deps: numpy/timm/torch/transformers),
# so we make the convenience re-export lazy: it still works when the training deps are
# present, and degrades gracefully otherwise. See docs/04_REPRO_LOG.md ("Setup surgery").
try:
    from .models import available_model_names, available_models, get_model_description, load  # noqa: F401
except Exception as _e:  # pragma: no cover
    import logging as _logging

    _logging.getLogger(__name__).debug(f"[prismatic] Skipping eager `.models` import (training deps absent): {_e}")
