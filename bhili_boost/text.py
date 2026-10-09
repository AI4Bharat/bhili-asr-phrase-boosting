"""Text normalisation and word error rate, with no dependencies beyond the standard library."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_PUNCT = re.compile(r"[^\w\sऀ-ॿ]")
_DANDA = re.compile(r"[।॥]")


def normalize(text: str | None) -> list[str]:
    """NFC-normalise, drop punctuation (including danda) and split into words."""
    text = unicodedata.normalize("NFC", text or "")
    text = _DANDA.sub(" ", _PUNCT.sub(" ", text))
    return text.split()


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance between two word lists (substitutions + insertions + deletions)."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, start=1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    """Corpus word error rate in percent."""
    errors = words = 0
    for r, h in zip(refs, hyps, strict=True):
        rw = normalize(r)
        errors += word_errors(rw, normalize(h))
        words += len(rw)
    return 100.0 * errors / max(words, 1)
