"""Loading the Bhili Nemotron model and decoding with (or without) phrase boosting.

The checkpoint is a multisoftmax model: one output head per language. It needs AI4Bharat's
NeMo 3.1.0 multisoftmax build (see the README) and must always run with the Bhili head
(``bhb``) selected. This module also works around two issues in that build:

* beam search with a boosting tree segfaults during CUDA-graph capture, so CUDA graphs are
  turned off for beam decoding;
* the batched beam decoder checks the vocabulary size of the full multi-language head, not the
  Bhili slice in use, so the joint network's class count is aligned to the slice.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import dataclass, field

import torch

LANG = "bhb"
HF_REPO = "ai4bharat/bhili-asr-nemotron-600m"
HF_FILE = "indic_nemotron_bhili_sft_lr1-averaged.nemo"


@dataclass
class DecodeConfig:
    """Decoding settings. Defaults are the tested setting (beam 4, boosting strength 0.5)."""

    terms: list[str] = field(default_factory=list)
    alpha: float = 0.5
    beam: int = 4
    batch_size: int = 16


def _quiet():
    """Silence NeMo's console output unless BHILI_BOOST_VERBOSE=1."""
    if os.environ.get("BHILI_BOOST_VERBOSE") == "1":
        return contextlib.nullcontext()
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(nemo_logging.ERROR)
    except Exception:
        pass
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    return stack


def resolve_checkpoint(path: str | None) -> str:
    """Use ``path``, else $BHILI_ASR_MODEL, else download the checkpoint from Hugging Face."""
    path = path or os.environ.get("BHILI_ASR_MODEL")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model checkpoint not found: {path}")
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, HF_FILE)


def _check_nemo():
    try:
        from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
            EncDecHybridRNNTCTCBPEModelWithPrompt as cls,
        )
    except ImportError as exc:
        raise ImportError("NeMo is not installed. See the README section 'Install'.") from exc
    if not hasattr(cls, "set_inference_language"):
        raise ImportError(
            "The installed NeMo has no multisoftmax support (set_inference_language is missing). "
            "This model needs AI4Bharat's NeMo 3.1.0 multisoftmax build; see the README section 'Install'."
        )


def load_model(checkpoint: str | None = None, device: str | None = None):
    """Load the model with the Bhili head selected."""
    _check_nemo()
    from nemo.collections.asr.data import audio_to_text_lhotse_prompt_index as prompt_index
    from nemo.collections.asr.models import ASRModel

    # Every utterance gets the Bhili prompt. Without this the dataloader picks the 'auto' or
    # 'bhb' prompt at random per utterance, which costs about 17 WER points.
    prompt_index.LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut = (
        lambda self, cut: self._get_prompt_index(LANG)
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_checkpoint(checkpoint)
    with _quiet():
        model = ASRModel.restore_from(path, map_location=device).eval()
        model.set_inference_language(LANG)
    return model


def configure(model, cfg: DecodeConfig) -> None:
    """Set the decoding strategy and boosting tree on ``model``."""
    from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
    from omegaconf import OmegaConf, open_dict

    tree = None
    if cfg.terms:
        tree = OmegaConf.structured(
            BoostingTreeModelConfig(key_phrases_list=list(cfg.terms), context_score=1.0, depth_scaling=2.0, use_triton=False)
        )
    alpha = cfg.alpha if cfg.terms else 0.0
    with open_dict(model.cfg.decoding):
        dec = model.cfg.decoding
        greedy = dec.greedy
        greedy.boosting_tree, greedy.boosting_tree_alpha = None, 0.0
        greedy.ngram_lm_model, greedy.ngram_lm_alpha = None, 0.0
        if cfg.beam > 1:
            dec.strategy = "malsd_batch"
            if dec.get("beam") is None:
                dec.beam = {}
            beam = dec.beam
            beam.beam_size = cfg.beam
            beam.return_best_hypothesis = True
            beam.allow_cuda_graphs = False
            beam.boosting_tree, beam.boosting_tree_alpha = tree, alpha
            beam.ngram_lm_model, beam.ngram_lm_alpha = None, 0.0
        else:
            dec.strategy = "greedy_batch"
            greedy.boosting_tree, greedy.boosting_tree_alpha = tree, alpha
    bhb_vocab = model.tokenizer.tokenizers_dict[LANG].vocab_size
    model.joint._num_classes = bhb_vocab + 1 + model.joint._num_extra_outputs
    model._build_decoding_objects()


def transcribe(model, files: list[str], cfg: DecodeConfig) -> list[str]:
    """Transcribe ``files`` (any sample rate, mono or stereo) with the settings in ``cfg``."""
    from bhili_boost.audio import write_model_manifest

    with _quiet():
        configure(model, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = write_model_manifest(files, tmp, LANG)
        with torch.no_grad(), _quiet():
            hyps = model.transcribe(manifest, batch_size=cfg.batch_size, target_lang=LANG, num_workers=0, verbose=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    return [h if isinstance(h, str) else h.text for h in hyps]
