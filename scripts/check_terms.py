#!/usr/bin/env python3
"""Check a term list before using it. Needs no model and no GPU.

  python scripts/check_terms.py terms/my_terms.txt

Exit code 0 = usable (warnings may remain), 1 = errors to fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bhili_boost.terms import load_terms, validate_terms  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = argv[0]
    if not Path(path).is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    issues = validate_terms(path)
    terms = load_terms(path)
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    for issue in issues:
        print(issue)
    print(f"\n{len(terms)} terms, {len(errors)} errors, {len(warnings)} warnings")
    if not errors:
        print("OK: the list can be used. Next: measure it with scripts/evaluate.py on audio with reference transcripts.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
