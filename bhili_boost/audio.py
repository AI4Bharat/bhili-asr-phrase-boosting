"""Audio preparation: the model expects 16 kHz mono input."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def _resample(wave: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wave
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.as_tensor(wave), sr, TARGET_SR).numpy()


def prepare(path: str | Path, workdir: str | Path) -> tuple[str, float]:
    """Return a path to a 16 kHz mono copy of ``path`` (the original if already so) and its duration."""
    info = sf.info(str(path))
    if info.samplerate == TARGET_SR and info.channels == 1:
        return str(path), info.frames / TARGET_SR
    wave, sr = sf.read(str(path), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    wave = _resample(wave, sr)
    out = Path(workdir) / (Path(path).stem + "_16k.wav")
    sf.write(str(out), wave, TARGET_SR)
    return str(out), len(wave) / TARGET_SR


def read_manifest(path: str | Path) -> list[dict]:
    """Read a NeMo-style JSONL manifest. Relative audio paths are resolved against its folder."""
    base = Path(path).resolve().parent
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not os.path.isabs(row["audio_filepath"]):
                row["audio_filepath"] = str(base / row["audio_filepath"])
            rows.append(row)
    return rows


def write_model_manifest(files: list[str], workdir: str | Path, lang: str) -> str:
    """Write the manifest the model reads: 16 kHz mono audio, one entry per input file, same order."""
    out = Path(workdir) / "model_manifest.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for f in files:
            audio, dur = prepare(f, workdir)
            fh.write(json.dumps({"audio_filepath": audio, "duration": dur, "lang": lang, "text": ""}) + "\n")
    return str(out)
