#!/usr/bin/env python3
"""Measure what a term list does on audio that has reference transcripts.

  python scripts/evaluate.py --terms terms/my_terms.txt --manifest test.jsonl --report report.md

The manifest needs 'audio_filepath' and 'text' (the reference) on every line. The script decodes
everything twice, without and with boosting, and reports:

  * WER without and with boosting;
  * for every term: how often it is in the references, how often each run gets it right,
    and how often each run writes it where the reference does not have it (false insertions);
  * a verdict per term: KEEP (helps), CHECK (no effect) or DROP (inserts more than it fixes).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bhili_boost.audio import read_manifest  # noqa: E402
from bhili_boost.model import DecodeConfig, load_model, transcribe  # noqa: E402
from bhili_boost.terms import load_terms, validate_terms  # noqa: E402
from bhili_boost.text import normalize, wer  # noqa: E402


def count_phrase(words: list[str], phrase: list[str]) -> int:
    n = len(phrase)
    return sum(words[i : i + n] == phrase for i in range(len(words) - n + 1))


def term_stats(terms, refs, base, boost):
    rows = []
    for term in terms:
        ph = normalize(term)
        in_ref = hit_b = hit_x = fa_b = fa_x = 0
        for r, b, x in zip(refs, base, boost):
            cr, cb, cx = (count_phrase(normalize(t), ph) for t in (r, b, x))
            in_ref += cr
            hit_b += min(cr, cb)
            hit_x += min(cr, cx)
            fa_b += max(cb - cr, 0)
            fa_x += max(cx - cr, 0)
        gained, new_fa = hit_x - hit_b, fa_x - fa_b
        if new_fa > max(gained, 0):
            verdict = "DROP"
        elif gained > 0:
            verdict = "KEEP"
        else:
            verdict = "CHECK"
        rows.append(dict(term=term, in_ref=in_ref, hits_without=hit_b, hits_with=hit_x,
                         false_ins_without=fa_b, false_ins_with=fa_x, verdict=verdict))
    return rows


def render(args, n, w_base, w_boost, stats) -> str:
    lines = [
        "# Phrase-boosting evaluation", "",
        f"- term list: `{args.terms}` ({len(stats)} terms)",
        f"- manifest: `{args.manifest}` ({n} utterances)",
        f"- decoding: beam {args.beam}, boosting strength {args.alpha}", "",
        "| | WER |", "|---|---|",
        f"| without boosting | {w_base:.2f} |", f"| with boosting | {w_boost:.2f} |",
        f"| change | {w_boost - w_base:+.2f} |", "",
        "| term | in references | correct without | correct with | false insertions without | false insertions with | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {"DROP": 0, "KEEP": 1, "CHECK": 2}
    for s in sorted(stats, key=lambda s: (order[s["verdict"]], -s["in_ref"])):
        lines.append(f"| {s['term']} | {s['in_ref']} | {s['hits_without']} | {s['hits_with']} | "
                     f"{s['false_ins_without']} | {s['false_ins_with']} | {s['verdict']} |")
    lines += ["", "KEEP = more correct with boosting and no extra false insertions beyond the gain. "
              "DROP = boosting adds more false insertions than correct words: remove the term or respell it. "
              "CHECK = no measurable effect here, often because the term is absent from these references."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terms", required=True)
    p.add_argument("--manifest", required=True, help="JSONL with 'audio_filepath' and reference 'text'")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--limit", type=int, help="use only the first N utterances (quick check)")
    p.add_argument("--report", help="write a Markdown report here")
    p.add_argument("--out", help="write per-utterance results as JSONL here")
    p.add_argument("--model"); p.add_argument("--device"); p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args(argv)

    issues = validate_terms(a.terms)
    if any(i.level == "error" for i in issues):
        for i in issues:
            print(f"[terms] {i}", file=sys.stderr)
        return 2
    terms = load_terms(a.terms)
    rows = read_manifest(a.manifest)[: a.limit] if a.limit else read_manifest(a.manifest)
    if not rows or any("text" not in r for r in rows):
        print("error: every manifest line needs a reference 'text'", file=sys.stderr)
        return 2
    files = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]

    model = load_model(a.model, a.device)
    base = transcribe(model, files, DecodeConfig(terms=[], beam=a.beam, batch_size=a.batch_size))
    boost = transcribe(model, files, DecodeConfig(terms=terms, alpha=a.alpha, beam=a.beam, batch_size=a.batch_size))

    stats = term_stats(terms, refs, base, boost)
    report = render(a, len(rows), wer(refs, base), wer(refs, boost), stats)
    print(report)
    if a.report:
        Path(a.report).write_text(report, encoding="utf-8")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            for f, r, b, x in zip(files, refs, base, boost):
                fh.write(json.dumps({"audio_filepath": f, "text": r, "pred_text_no_boost": b, "pred_text": x}, ensure_ascii=False) + "\n")
    verdicts = Counter(s["verdict"] for s in stats)
    print(f"[summary] KEEP {verdicts['KEEP']}  CHECK {verdicts['CHECK']}  DROP {verdicts['DROP']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
