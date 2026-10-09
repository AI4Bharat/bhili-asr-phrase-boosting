import json

import numpy as np
import soundfile as sf

from bhili_boost.audio import prepare, read_manifest, write_model_manifest


def test_prepare_keeps_16k_mono_and_converts_others(tmp_path):
    mono = tmp_path / "mono16k.wav"
    sf.write(mono, np.zeros(16000, dtype="float32"), 16000)
    path, dur = prepare(mono, tmp_path)
    assert path == str(mono) and dur == 1.0

    stereo = tmp_path / "stereo8k.wav"
    sf.write(stereo, np.zeros((8000, 2), dtype="float32"), 8000)
    path, dur = prepare(stereo, tmp_path)
    info = sf.info(path)
    assert info.samplerate == 16000 and info.channels == 1 and abs(dur - 1.0) < 1e-3


def test_manifest_relative_paths_and_model_manifest(tmp_path):
    a = tmp_path / "a.wav"
    sf.write(a, np.zeros(16000, dtype="float32"), 16000)
    man = tmp_path / "m.jsonl"
    man.write_text(json.dumps({"audio_filepath": "a.wav", "text": "भात"}) + "\n", encoding="utf-8")
    rows = read_manifest(man)
    assert rows[0]["audio_filepath"] == str(a)
    out = write_model_manifest([rows[0]["audio_filepath"]], tmp_path, "bhb")
    entry = json.loads(open(out, encoding="utf-8").read().splitlines()[0])
    assert entry["lang"] == "bhb" and entry["duration"] == 1.0
