"""d1a.eval.suite: frozen partitions read only as pinned (hash, count, the locked test, removed suites, the Hub mirrors), the
file conventions (UTF-8 whatever the locale, LF-only JSONL, atomic writes), and the training-source guard."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

from d1a.eval import suite as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = {"state": "Zwölf Boxkämpfer: ‘quotes’ and é", "questions": {}, "_meta": {"id": "r0"}}


def frozen(directory, partitions):
    """A suite directory whose manifest pins `partitions` ({split: records}) as written."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, records in partitions.items():
        S.write_jsonl(directory / f"{split}.jsonl", records)
        files[f"{split}.jsonl"] = {"sha256": S.digest(directory / f"{split}.jsonl"), "records": len(records)}
    S.write_json(directory / "manifest.json", {"files": files})
    return directory


# --- reading frozen partitions ------------------------------------------------------------------------------------------

def test_partitions_are_read_only_as_pinned(tmp_path):
    suite = frozen(tmp_path / "s", {"development": [RECORD], "test": [RECORD, RECORD]})
    assert S.load_split(suite, "development") == [RECORD]
    with pytest.raises(ValueError, match="locked test"):
        S.load_split(suite, "test")
    assert len(S.load_split(suite, "test", allow_test=True)) == 2
    with pytest.raises(ValueError, match="unknown split"):
        S.load_split(suite, "validation")
    S.write_jsonl(suite / "development.jsonl", [{}])                                    # edited after freezing
    with pytest.raises(ValueError, match="checksum"):
        S.load_split(suite, "development")
    manifest = S.read_json(suite / "manifest.json")
    manifest["files"]["test.jsonl"]["records"] = 3
    S.write_json(suite / "manifest.json", manifest)
    with pytest.raises(ValueError, match="record count"):
        S.load_split(suite, "test", allow_test=True)


def test_a_removed_suite_is_refused_with_its_reason(tmp_path):
    removed = tmp_path / "evals" / "external" / "scienthoon-v1"
    frozen(removed, {"development": [RECORD]})
    for path in (removed, "evals/external/scienthoon-v1", "/root/kev/evals/external/scienthoon-v1"):
        with pytest.raises(S.RemovedSuite, match="removed on 2026-09-27: unsound as a gate"):
            S.load_split(path, "development")
        with pytest.raises(S.RemovedSuite, match="Do not read it again"):
            S.read_manifest(path)
    assert S.suite_key("/root/kev/evals/v7/decision-v7") == "evals/v7/decision-v7" and S.suite_key("x/y") is None
    assert S.removed_suite("evals/v7/decision-v7") is None


def test_a_missing_partition_is_fetched_and_verified(tmp_path, monkeypatch):
    payload = json.dumps(RECORD, ensure_ascii=False).encode() + b"\n"
    suite = tmp_path / "evals" / "x" / "decision-x"
    suite.mkdir(parents=True)
    S.write_json(suite / "manifest.json", {"files": {"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served, calls = tmp_path / "served.jsonl", []
    served.write_bytes(payload)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, path, repo_type, revision: (calls.append((repo, path, repo_type, revision)), str(served))[1])
    assert S.load_split(suite, "train") == [RECORD]
    assert calls == [(S.SUITES_DATASET, "x/decision-x/train.jsonl", "dataset", S.SUITES_REVISION)]
    (suite / "train.jsonl").unlink()
    served.write_bytes(b'{"tampered": 1}\n')                                             # a mirror that changed is caught
    with pytest.raises(ValueError, match="checksum"):
        S.load_split(suite, "train")
    outside = frozen(tmp_path / "loose", {})
    S.write_json(outside / "manifest.json", {"files": {"train.jsonl": {"sha256": "0", "records": 1}}})
    with pytest.raises(FileNotFoundError, match="not under an evals/ tree"):
        S.load_split(outside, "train")


def test_a_private_suite_fetches_from_its_own_mirror(tmp_path, monkeypatch):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    suite = tmp_path / "evals" / "held" / "docs-x"
    suite.mkdir(parents=True)
    S.write_json(suite / "manifest.json", {"mirror": {"dataset": S.PRIVATE_DATASET, "revision": "abc123"},
                                           "files": {"test.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served, calls = tmp_path / "served.jsonl", []
    served.write_bytes(payload)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, path, repo_type, revision: (calls.append((repo, path, repo_type, revision)), str(served))[1])
    assert len(S.load_split(suite, "test", allow_test=True)) == 1
    assert calls == [(S.PRIVATE_DATASET, "held/docs-x/test.jsonl", "dataset", "abc123")]   # its own dataset and revision
    (suite / "test.jsonl").unlink()

    def denied(*args, **kwargs):   # without access, the Hub answers "not found"
        raise RepositoryNotFoundError("404 Client Error", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", denied)
    with pytest.raises(PermissionError, match="kev-private-evals, which is missing or private"):
        S.load_split(suite, "test", allow_test=True)


# --- file conventions ---------------------------------------------------------------------------------------------------

def test_partitions_load_under_any_locale(tmp_path):
    """Issue #12: frozen partitions hold non-ASCII text and are hash-checked byte for byte, so they are read as UTF-8 under
    any locale (cp936 on Windows in the report; the ASCII C locale here), and git keeps their LF line endings."""
    suite = frozen(tmp_path / "evals" / "x" / "decision-x", {"development": [RECORD]})
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"}   # stdout only
    code = f"import locale; from d1a.eval.suite import load_split; print(locale.getpreferredencoding(False), load_split({str(suite)!r}, 'development')[0]['state'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith(RECORD["state"]) and "UTF-8" not in out.stdout.split()[0].upper(), out.stdout
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.jsonl text eol=lf" in attributes and "*.json text eol=lf" in attributes


def test_jsonl_lines_end_at_newline_only(tmp_path):
    records = [{"state": "line one line two"}, {"state": "next\x85record end"}, {"state": "plain"}]
    S.write_jsonl(tmp_path / "x.jsonl", records)
    assert S.read_jsonl(tmp_path / "x.jsonl") == records
    assert (tmp_path / "x.jsonl").read_bytes().count(b"\n") == 3


def test_json_files(tmp_path):
    S.write_json(tmp_path / "a.json", {"é": [1, 2]})
    assert (tmp_path / "a.json").read_bytes() == '{\n  "é": [\n    1,\n    2\n  ]\n}\n'.encode()   # indented, UTF-8, final newline
    with pytest.raises(ValueError):
        S.write_json(tmp_path / "b.json", {"x": float("nan")})
    S.write_json(tmp_path / "a.json", {"v": 2}, atomic=True)
    assert S.read_json(tmp_path / "a.json") == {"v": 2} and not (tmp_path / ".a.json.tmp").exists()
    with S.file_lock(tmp_path / "locks" / "run.lock"):
        assert (tmp_path / "locks" / "run.lock").exists()


def test_hashes():
    assert S.record_digest({"b": "é", "a": [1]}) == hashlib.sha256('{"b":"é","a":[1]}'.encode()).hexdigest()
    assert S.normalise_text("  Straße\tIS\n  here ") == "strasse is here"
    assert S.text_digest("A  b") == S.text_digest("a b") == hashlib.sha256(b"a b").hexdigest()


def test_digest_reads_large_files_whole(tmp_path):
    blob = os.urandom(3 * (1 << 20) + 5)
    (tmp_path / "blob").write_bytes(blob)
    assert S.digest(tmp_path / "blob") == hashlib.sha256(blob).hexdigest()


# --- training sources ---------------------------------------------------------------------------------------------------

def test_training_refuses_eval_only_undeclared_and_empty_sources():
    manifest = {"trainable_sources": ["agnews"], "holdout_sources": ["mnli"], "eval_only_sources": ["pr-labels"]}
    S.validate_training([{"_meta": {"source": "agnews"}}], manifest)
    for source in ("mmlu", "mnli", "pr-labels", "boolq"):                               # eval-only by policy, held out, declared, undeclared
        with pytest.raises(ValueError, match=f"training source: {source}"):
            S.validate_training([{"_meta": {"source": source}}], manifest)
    S.validate_training([{"_meta": {"source": "boolq"}}], {})                           # no trainable list: anything not refused
    with pytest.raises(ValueError, match="empty"):
        S.validate_training([], {})


def test_training_refuses_held_out_compositional_structures():
    from d1a.training.composition import DEV_SHAPES, SHAPES, canonical, push_negation, structure_keys
    allowed = {"trainable_sources": ["compositional"]}
    S.validate_training([{"_meta": {"source": "compositional", "family": "nested_and"}}], allowed)
    S.validate_training([{"_meta": {"source": "compositional", "family": "rand:7", "structure": "and(0,1)"}}], allowed)
    for family in ("held_and_or", "final_exception", "made_up"):                         # dev, test, and not a training shape
        with pytest.raises(ValueError, match="held-out"):
            S.validate_training([{"_meta": {"source": "compositional", "family": family}}], allowed)
    # a random tree is refused when its structure, up to operand order, numbering and De Morgan, is a held-out shape's
    assert canonical(("or", ("not", 0), ("and", 1, 2))) == canonical(SHAPES["held_or_not"])
    assert canonical(("not", ("and", 0, 1))) != canonical(("or", ("not", 0), ("not", 1)))
    assert canonical(push_negation(("not", ("and", 0, 1)))) == canonical(("or", ("not", 0), ("not", 1)))
    negated = canonical(push_negation(SHAPES["final_negation"]))                         # held out only through De Morgan
    assert negated != canonical(SHAPES["final_negation"]) and negated in structure_keys(SHAPES["final_negation"])
    for held in (canonical(SHAPES[DEV_SHAPES[0]]), negated):
        with pytest.raises(ValueError, match="held-out"):
            S.validate_training([{"_meta": {"source": "compositional", "family": "rand:7", "structure": held}}], allowed)
