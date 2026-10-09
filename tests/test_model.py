"""End-to-end tests. They run only when BHILI_ASR_MODEL points at the checkpoint and the
NeMo multisoftmax build is installed; otherwise they are skipped."""

import os
from pathlib import Path

import pytest

SAMPLE = Path(os.environ.get("BHILI_ASR_SAMPLE", Path(os.environ.get("BHILI_ASR_MODEL", "x")).with_name("sample_audio.wav")))
SAMPLE_TEXT = "प्रत्येक वस्तू ईसताना यादी मेने वगळी टाकणं"


@pytest.fixture(scope="module")
def model():
    from bhili_boost.model import load_model
    return load_model()


@pytest.mark.model
@pytest.mark.parametrize("beam", [1, 4])
def test_sample_without_boosting(model, beam):
    from bhili_boost.model import DecodeConfig, transcribe
    assert transcribe(model, [str(SAMPLE)], DecodeConfig(beam=beam)) == [SAMPLE_TEXT]


@pytest.mark.model
@pytest.mark.parametrize("beam", [1, 4])
def test_unrelated_terms_leave_sample_unchanged(model, beam):
    from bhili_boost.model import DecodeConfig, transcribe
    out = transcribe(model, [str(SAMPLE)], DecodeConfig(terms=["एच एन पी व्ही", "ग्लिरीसिडिया"], beam=beam))
    assert out == [SAMPLE_TEXT]


@pytest.mark.model
def test_boosting_can_switch_back_off(model):
    from bhili_boost.model import DecodeConfig, transcribe
    transcribe(model, [str(SAMPLE)], DecodeConfig(terms=["भात"]))
    assert transcribe(model, [str(SAMPLE)], DecodeConfig()) == [SAMPLE_TEXT]
