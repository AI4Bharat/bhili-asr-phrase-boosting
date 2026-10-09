import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _model_available() -> bool:
    ckpt = os.environ.get("BHILI_ASR_MODEL")
    if not ckpt or not Path(ckpt).is_file():
        return False
    try:
        from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
            EncDecHybridRNNTCTCBPEModelWithPrompt as cls,
        )
    except Exception:
        return False
    return hasattr(cls, "set_inference_language")


def pytest_collection_modifyitems(config, items):
    if _model_available():
        return
    skip = pytest.mark.skip(reason="set BHILI_ASR_MODEL and install the NeMo multisoftmax build to run model tests")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
