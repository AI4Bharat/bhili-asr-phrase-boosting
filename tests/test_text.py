import pytest

from bhili_boost.text import normalize, wer, word_errors


def test_normalize_strips_punctuation_and_danda():
    assert normalize("भात, पिकाम। माहिती?") == ["भात", "पिकाम", "माहिती"]


def test_word_errors():
    assert word_errors(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert word_errors(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert word_errors(["a", "b"], []) == 2
    assert word_errors([], ["a"]) == 1


def test_wer_corpus_level():
    assert wer(["भात पिकाम माहिती"], ["भारत पिकाम माहिती"]) == pytest.approx(100 / 3)
    assert wer(["a b", "c d"], ["a b", "c d"]) == 0.0


def test_wer_requires_equal_lengths():
    with pytest.raises(ValueError):
        wer(["a"], ["a", "b"])
