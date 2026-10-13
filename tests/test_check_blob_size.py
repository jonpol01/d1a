import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_blob_size import LIMIT, violations  # noqa: E402

BIG = LIMIT + 1


def test_new_large_file_is_refused_unless_allowlisted():
    changes = [("A", "weights/model.safetensors"), ("A", "evals/new-suite/test.jsonl"), ("A", "small.txt")]
    sizes = {"weights/model.safetensors": BIG, "evals/new-suite/test.jsonl": BIG, "small.txt": 10}
    assert violations(changes, sizes, binary=set(), patterns=["evals/*.jsonl"]) == [("weights/model.safetensors", BIG, "new file")]


def test_edit_of_large_text_passes_but_a_changed_large_binary_does_not():
    changes = [("M", "evals/hard-v1/test.jsonl"), ("M", "docs/demo.gif")]
    sizes = {"evals/hard-v1/test.jsonl": BIG, "docs/demo.gif": BIG}
    assert violations(changes, sizes, binary={"docs/demo.gif"}, patterns=[]) == [("docs/demo.gif", BIG, "changed binary")]
