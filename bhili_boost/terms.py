"""Reading and checking a phrase-boosting term list.

A term list is a UTF-8 text file with one term or phrase per line. Blank lines and lines
starting with ``#`` are ignored. Terms should be Devanagari, spelled the way farmers say them
in Bhili, with acronyms written out letter by letter (``एच एन पी व्ही`` rather than ``HNPV``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Everyday words that appear in most turns. Boosting them inserts them everywhere.
GENERIC_WORDS = frozenset(
    {
        "हो", "हा", "ह", "हां", "हाँ", "हम्म", "नाही", "नाहा", "माहिती", "पाणी", "पाय", "काय",
        "आहे", "हाय", "सांगा", "आन", "आणि", "का", "की", "मी", "आय", "तुम्ही", "आपु", "बस",
    }
)

_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9०-९]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass(frozen=True)
class TermIssue:
    """A problem found in a term list. ``level`` is ``"error"`` or ``"warning"``."""

    line: int
    term: str
    level: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.level.upper()} '{self.term}': {self.message}"


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", line)).strip()


def load_terms(path: str | Path) -> list[str]:
    """Return the unique terms in ``path``, in file order, NFC-normalised."""
    terms: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        term = _clean(raw)
        if term and not term.startswith("#") and term not in terms:
            terms.append(term)
    return terms


def validate_terms(path: str | Path) -> list[TermIssue]:
    """Check a term list and return every issue found.

    Errors make the list unusable as written. Warnings point at terms that are likely to be
    inserted where nobody said them.
    """
    issues: list[TermIssue] = []
    seen: dict[str, int] = {}
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        term = _clean(raw)
        if not term or term.startswith("#"):
            continue
        if term in seen:
            issues.append(TermIssue(n, term, "warning", f"duplicate of line {seen[term]}; ignored"))
            continue
        seen[term] = n
        if _LATIN.search(term):
            issues.append(TermIssue(n, term, "error", "contains Latin letters; write it in Devanagari as spoken (HNPV -> एच एन पी व्ही)"))
        elif not _DEVANAGARI.search(term):
            issues.append(TermIssue(n, term, "error", "contains no Devanagari"))
        if _DIGIT.search(term):
            issues.append(TermIssue(n, term, "error", "contains digits; write numbers as words, the way they are spoken"))
        if term in GENERIC_WORDS:
            issues.append(TermIssue(n, term, "warning", "everyday word; boosting it inserts it into unrelated turns"))
        elif len(term.replace(" ", "")) < 3:
            issues.append(TermIssue(n, term, "warning", "very short; likely to be inserted into unrelated turns"))
    if not seen:
        issues.append(TermIssue(0, "", "error", "the file has no terms"))
    return issues
