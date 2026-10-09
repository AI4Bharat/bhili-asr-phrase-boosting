#!/usr/bin/env python3
"""Transcribe Bhili audio, optionally with phrase boosting from a term list.

Examples
  python scripts/transcribe.py audio1.wav audio2.wav
  python scripts/transcribe.py --terms terms/my_terms.txt --compare audio1.wav
  python scripts/transcribe.py --terms terms/my_terms.txt --manifest calls.jsonl --out preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bhili_boost.audio import read_manifest  # noqa: E402
from bhili_boost.model import DecodeConfig, load_model, transcribe  # noqa: E402
from bhili_boost.terms import load_terms, validate_terms  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="*", help="audio files (any sample rate; stereo is mixed down)")
    p.add_argument("--manifest", help="JSONL manifest with an 'audio_filepath' per line")
    p.add_argument("--terms", help="term list, one term per line (omit to decode without boosting)")
    p.add_argument("--alpha", type=float, default=0.5, help="boosting strength (default 0.5, the tested value)")
    p.add_argument("--beam", type=int, default=4, help="beam width; 1 = greedy (default 4)")
    p.add_argument("--compare", action="store_true", help="also decode without boosting and show both")
    p.add_argument("--out", help="write results as JSONL to this file")
    p.add_argument("--model", help="path to the .nemo checkpoint (default: $BHILI_ASR_MODEL or Hugging Face download)")
    p.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    files = list(a.audio)
    if a.manifest:
        files += [r["audio_filepath"] for r in read_manifest(a.manifest)]
    if not files:
        print("error: give audio files or --manifest", file=sys.stderr)
        return 2
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        print(f"error: audio not found: {', '.join(missing[:5])}", file=sys.stderr)
        return 2

    terms: list[str] = []
    if a.terms:
        issues = validate_terms(a.terms)
        for issue in issues:
            print(f"[terms] {issue}", file=sys.stderr)
        if any(i.level == "error" for i in issues):
            print("error: fix the term list errors above first (scripts/check_terms.py)", file=sys.stderr)
            return 2
        terms = load_terms(a.terms)
        print(f"[terms] {len(terms)} terms loaded from {a.terms}", file=sys.stderr)

    model = load_model(a.model, a.device)
    base = None
    if a.compare and terms:
        base = transcribe(model, files, DecodeConfig(terms=[], beam=a.beam, batch_size=a.batch_size))
    hyps = transcribe(model, files, DecodeConfig(terms=terms, alpha=a.alpha, beam=a.beam, batch_size=a.batch_size))

    rows = []
    for i, f in enumerate(files):
        row = {"audio_filepath": f, "pred_text": hyps[i]}
        if base is not None:
            row["pred_text_no_boost"] = base[i]
        rows.append(row)
        name = Path(f).name
        if base is not None and base[i] != hyps[i]:
            print(f"{name}\n    without boosting: {base[i]}\n    with boosting   : {hyps[i]}")
        else:
            print(f"{name}\t{hyps[i]}")
    if base is not None:
        changed = sum(b != h for b, h in zip(base, hyps))
        print(f"[compare] boosting changed {changed} of {len(files)} transcripts", file=sys.stderr)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[out] wrote {len(rows)} rows to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
