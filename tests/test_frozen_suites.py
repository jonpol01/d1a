"""The frozen suites under evals/ as data, without their builders (retired: docs/removed-tools.md). Every partition kept in
git still has the sha256 and record count its manifest pins; the eval-only suites built from Kev's builders keep their
contracts; devtools-v1's evaluation partitions share exactly the ids its manifest documents. No weights, no network."""
import json
from collections import Counter
from pathlib import Path

import pytest

from d1a.suite import PRIVATE_DATASET, SERVING_CONTEXT, digest, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = sorted((ROOT / "evals").glob("*/manifest.json"))


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m.parent.name for m in MANIFESTS])
def test_every_partition_in_git_matches_its_manifest(manifest):
    files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
    for name, entry in files.items():
        path = manifest.parent / name
        if not path.exists(): continue   # a large or private partition: fetched and hash-checked by d1a.suite on use
        assert digest(path) == entry["sha256"], f"{path} changed after the freeze"
        if "records" in entry: assert sum(1 for _ in read_jsonl(path)) == entry["records"], path


def test_kev_built_eval_suites_keep_their_contracts():
    for name in ("breadth-v1", "longdoc-v1"):
        m = json.loads((ROOT / "evals" / name / "manifest.json").read_text(encoding="utf-8"))
        assert m["eval_only"] is True and m["trainable_sources"] == [] and m["locked"] == ["test"], name
        assert m["mirror"]["dataset"] == PRIVATE_DATASET and len(m["mirror"]["revision"]) == 40, name
    breadth = json.loads((ROOT / "evals/breadth-v1/manifest.json").read_text(encoding="utf-8"))
    assert {d for a in breadth["areas"].values() for d in a["datasets"]} == set(breadth["datasets"])
    overlap = json.loads((ROOT / "evals/breadth-v1/overlap.json").read_text(encoding="utf-8"))
    assert overlap["offending_records"] == 0 and overlap["records"] == sum(f["records"] for f in breadth["files"].values())
    longdoc = json.loads((ROOT / "evals/longdoc-v1/manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in longdoc["context"].items() if k != "note"} == SERVING_CONTEXT
    assert set(longdoc["files"]) == {"development.jsonl", "test.jsonl"}


def test_devtools_v1_evaluation_partitions_share_only_the_documented_ids():
    """Two codereviewer items appear twice in one partition each (a known defect of the frozen build, kept so the hashes
    hold); nothing else may repeat."""
    for split, expected in (("development", {"codereviewer/cls-test/13657"}), ("test", {"codereviewer/cls-test/19245"})):
        ids = Counter(r["_meta"]["id"] for r in read_jsonl(ROOT / "evals/devtools-v1" / f"{split}.jsonl"))
        assert {i for i, n in ids.items() if n > 1} == expected
