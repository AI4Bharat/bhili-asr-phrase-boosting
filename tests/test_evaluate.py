import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_count_phrase():
    assert evaluate.count_phrase(["एच", "एन", "पी", "व्ही", "एच", "एन"], ["एच", "एन"]) == 2


def test_term_verdicts():
    refs = ["एच एन पी व्ही बद्दल", "भात पिकाम", "माती"]
    base = ["एचएम बद्दल", "भात पिकाम", "माती"]
    boost = ["एच एन पी व्ही बद्दल", "भात पिकाम", "माती ठिबक"]
    stats = {s["term"]: s for s in evaluate.term_stats(["एच एन पी व्ही", "भात", "ठिबक"], refs, base, boost)}
    assert stats["एच एन पी व्ही"]["verdict"] == "KEEP"
    assert stats["भात"]["verdict"] == "CHECK"
    assert stats["ठिबक"]["verdict"] == "DROP" and stats["ठिबक"]["false_ins_with"] == 1
