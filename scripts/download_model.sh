#!/usr/bin/env bash
# Download the gated Bhili ASR checkpoint and its sample clip.
# Request access on https://huggingface.co/ai4bharat/bhili-asr-nemotron-600m first, then
# authenticate once: `hf auth login` (or set HF_TOKEN in the environment).
#
#   bash scripts/download_model.sh [destination folder, default: models]
set -euo pipefail
DEST="${1:-models}"
mkdir -p "$DEST"
python - "$DEST" <<'PY'
import sys
from huggingface_hub import hf_hub_download

dest = sys.argv[1]
for name in ("indic_nemotron_bhili_sft_lr1-averaged.nemo", "sample_audio.wav"):
    print("downloaded", hf_hub_download("ai4bharat/bhili-asr-nemotron-600m", name, local_dir=dest))
PY
echo
echo "Point the scripts at the checkpoint with:"
echo "  export BHILI_ASR_MODEL=$(cd "$DEST" && pwd)/indic_nemotron_bhili_sft_lr1-averaged.nemo"
