# Bhili ASR phrase boosting

Tools to improve recognition of domain terms (schemes, crops, pests, inputs, places) in
AI4Bharat's Bhili speech recognition model,
[`ai4bharat/bhili-asr-nemotron-600m`](https://huggingface.co/ai4bharat/bhili-asr-nemotron-600m),
without retraining it.

You give the decoder a list of terms. Whenever its hypothesis starts to match one, it gets a
small bonus, so a term the model nearly heard comes out spelled right:

```
without boosting: एचएम बद्दल माहिती द्या
with boosting   : एच एन पी व्ही बदल माहिती द्या
```

The repo has four tools:

| Script | What it does | Needs the model |
|---|---|---|
| `scripts/check_terms.py` | Checks a term list for mistakes (Latin letters, digits, everyday words, duplicates) | no |
| `scripts/transcribe.py` | Transcribes audio, with or without boosting; `--compare` shows both | yes |
| `scripts/evaluate.py` | Measures a term list on audio with reference transcripts: WER, and a KEEP / CHECK / DROP verdict per term | yes |
| `scripts/download_model.sh` | Downloads the gated checkpoint and a sample clip from Hugging Face | no |

## What boosting can and cannot fix

Measured on the Bhili test set and real field calls (details in [docs/RESULTS.md](docs/RESULTS.md)):

- **It fixes** technical terms the model nearly hears: HNPV, Gliricidia, गिरीपुष्प. Cost on the
  whole test set: WER 32.95 → 33.06 with the default settings.
- **It does not fix** words the model does not hear at all. PM-Kisan was never recovered at a
  usable strength. Casual conversational speech is also out of reach. Those need training data.
- **A bad list makes things worse.** 211 words mined automatically raised WER from 33.2 to 35.0.
  Keep the list short and specific; see [docs/WRITING_TERMS.md](docs/WRITING_TERMS.md).

## Install

### 1. NeMo (multisoftmax build)

The checkpoint has one output head per language ("multisoftmax"). Loading it and selecting the
Bhili head needs **AI4Bharat's NeMo 3.1.0 multisoftmax build**. It is not on PyPI, and neither
`pip install nemo_toolkit[asr]` (3.0.0) nor NVIDIA's main branch has it: with those, loading
fails or the scripts stop with "no multisoftmax support".

On the IITM cluster the source is at
`/projects/data/llmteam/sanjay/demos/bhili-asr/nemo-multisoftmax`. Copy it and install the copy
(installing builds files inside the source folder, so do not install it in place):

```bash
conda create -n bhili-boost python=3.11 -y
conda activate bhili-boost
cp -r /projects/data/llmteam/sanjay/demos/bhili-asr/nemo-multisoftmax "$HOME/nemo-multisoftmax"
pip install -e "$HOME/nemo-multisoftmax[asr]"
```

Elsewhere, ask the AI4Bharat Bhili ASR maintainers for the build.

### 2. This repo

```bash
git clone git@github.com:AI4Bharat/bhili-asr-phrase-boosting.git
cd bhili-asr-phrase-boosting
pip install -r requirements.txt
```

### 3. The model

The checkpoint is gated. Request access on the
[model page](https://huggingface.co/ai4bharat/bhili-asr-nemotron-600m), then:

```bash
hf auth login                # or: export HF_TOKEN=<your token>
bash scripts/download_model.sh models
export BHILI_ASR_MODEL=$PWD/models/indic_nemotron_bhili_sft_lr1-averaged.nemo
```

Every script takes `--model <path>`; without it they use `$BHILI_ASR_MODEL`, and without that
they download the checkpoint from Hugging Face.

### 4. Check the install

```bash
python -m pytest            # unit tests; model tests run too when BHILI_ASR_MODEL is set
python scripts/transcribe.py models/sample_audio.wav
# sample_audio.wav    प्रत्येक वस्तू ईसताना यादी मेने वगळी टाकणं
```

## Usage

### Write a term list

Start from [`terms/TEMPLATE.txt`](terms/TEMPLATE.txt): one term per line, Devanagari, spelled
the way farmers say it in Bhili, acronyms letter by letter (`एच एन पी व्ही`). Read
[docs/WRITING_TERMS.md](docs/WRITING_TERMS.md) first. [`terms/example_terms.txt`](terms/example_terms.txt)
is the 16-term list used in the measurements.

```bash
python scripts/check_terms.py terms/my_terms.txt
```

### Transcribe

```bash
# files
python scripts/transcribe.py --terms terms/my_terms.txt call1.wav call2.wav

# show what boosting changes
python scripts/transcribe.py --terms terms/my_terms.txt --compare call1.wav

# a manifest, results to a file
python scripts/transcribe.py --terms terms/my_terms.txt --manifest calls.jsonl --out preds.jsonl
```

Audio can be any sample rate, mono or stereo; it is converted to 16 kHz mono. A manifest is
JSONL with an `audio_filepath` per line (relative paths are resolved against the manifest's
folder). Output lines are `{"audio_filepath", "pred_text"}`, plus `"pred_text_no_boost"` with
`--compare`.

| Option | Default | Meaning |
|---|---|---|
| `--terms` | none | term list; without it, no boosting |
| `--alpha` | `0.5` | boosting strength. Higher values insert terms where they were not said |
| `--beam` | `4` | beam width; `1` = greedy decoding |
| `--compare` | off | also decode without boosting and print both |
| `--model` | `$BHILI_ASR_MODEL` | `.nemo` checkpoint path |
| `--device` | cuda if available | `cuda` or `cpu` |
| `--batch-size` | `16` | lower it if the GPU runs out of memory |

### Measure a list before deploying it

```bash
python scripts/evaluate.py --terms terms/my_terms.txt --manifest test.jsonl --report report.md
```

The manifest needs a reference `text` on every line. The report gives WER without and with
boosting and a row per term: how often it appears in the references, how often each run gets
it right, how often each run writes it where it was not said, and a verdict:

- **KEEP**: boosting helps it.
- **DROP**: boosting inserts it more often than it fixes it. Remove or respell it.
- **CHECK**: no measurable effect here.

Use `--limit 200` for a quick first look.

### From Python

```python
from bhili_boost.model import DecodeConfig, load_model, transcribe
from bhili_boost.terms import load_terms

model = load_model()                      # uses $BHILI_ASR_MODEL
cfg = DecodeConfig(terms=load_terms("terms/my_terms.txt"), alpha=0.5, beam=4)
print(transcribe(model, ["call1.wav"], cfg))
```

## Things to know

- **The Bhili head must be selected.** The scripts do this. If you call the model yourself,
  pass `target_lang="bhb"` and pin the `bhb` prompt (`load_model` shows how). The model's
  dataloader otherwise picks the `auto` or `bhb` prompt at random per utterance, which costs
  about 17 WER points.
- **Two workarounds for the NeMo build** are applied in `bhili_boost/model.py`:
  - beam search with boosting segfaults during CUDA-graph capture, so CUDA graphs are off for
    beam decoding (decoding is still fast on a GPU: the 4,130-utterance test set takes about
    three minutes on one H100);
  - the batched beam decoder checks the vocabulary size of the full multi-language head
    instead of the Bhili head in use, so that size is aligned before decoding.
- **Quiet by default.** NeMo's own console output is hidden; set `BHILI_BOOST_VERBOSE=1` to see it.
- **Offline only.** These scripts decode whole files. The model also supports cache-aware
  streaming; boosting in streaming mode has not been tested.

## Repository layout

```
bhili_boost/           the library
  model.py             load the model, configure decoding and boosting, transcribe
  terms.py             read and check term lists
  audio.py             16 kHz mono conversion, manifests
  text.py              normalisation and WER
scripts/               command-line tools (see the table above)
terms/                 TEMPLATE.txt to fill in; example_terms.txt (the evaluated list)
examples/              a sample manifest
docs/
  WRITING_TERMS.md     how to write a term list that helps
  RESULTS.md           what was measured, and the numbers
tests/                 unit tests; model tests run when BHILI_ASR_MODEL is set
```

## Tests

```bash
python -m pytest -v
```

Unit tests need no model or GPU. The end-to-end tests in `tests/test_model.py` run only when
`BHILI_ASR_MODEL` points at the checkpoint and the NeMo multisoftmax build is installed; they
check the sample clip decodes correctly with greedy and beam decoding, that unrelated terms do
not change it, and that boosting can be switched off again.
