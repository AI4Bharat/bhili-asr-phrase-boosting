from bhili_boost.terms import load_terms, validate_terms


def _write(tmp_path, text):
    p = tmp_path / "terms.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_skips_comments_blanks_and_duplicates(tmp_path):
    p = _write(tmp_path, "# header\n\nभात\n  एच   एन पी व्ही  \nभात\n")
    assert load_terms(p) == ["भात", "एच एन पी व्ही"]


def test_load_applies_nfc(tmp_path):
    import unicodedata
    raw = "\N{DEVANAGARI LETTER QA}\N{DEVANAGARI VOWEL SIGN AA}"  # precomposed qa, decomposed by NFC
    p = _write(tmp_path, raw + "\n")
    (term,) = load_terms(p)
    assert term == unicodedata.normalize("NFC", raw) and term != raw


def test_example_and_template_lists_have_no_errors():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "terms"
    ex = validate_terms(root / "example_terms.txt")
    assert not [i for i in ex if i.level == "error"]
    assert len(load_terms(root / "example_terms.txt")) == 16
    tpl = validate_terms(root / "TEMPLATE.txt")
    assert [i.message for i in tpl] == ["the file has no terms"]


def test_latin_digits_and_generic_are_flagged(tmp_path):
    p = _write(tmp_path, "HNPV\nदोन 2 टक्के\nमाहिती\nहो\nभात\n")
    issues = {(i.term, i.level) for i in validate_terms(p)}
    assert ("HNPV", "error") in issues
    assert ("दोन 2 टक्के", "error") in issues
    assert ("माहिती", "warning") in issues
    assert ("हो", "warning") in issues
    assert not any(t == "भात" for t, _ in issues)


def test_duplicate_is_a_warning_with_line_number(tmp_path):
    p = _write(tmp_path, "भात\nभेंडी\nभात\n")
    (issue,) = validate_terms(p)
    assert issue.level == "warning" and issue.line == 3 and "line 1" in issue.message
