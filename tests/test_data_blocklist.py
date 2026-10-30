"""D1A_DATA_BLOCKLIST: names D1A's data must never hold (an organisation, its products, people) stay in a private file;
freeze will not pin data that holds one, resolve will not load it, and the trainer will not train on it."""
import json
from types import SimpleNamespace

import huggingface_hub
import pytest

from d1a.eval import suites


def blocklist_env(tmp_path, *entries):
    path, lines = tmp_path / "blocklist.txt", ["# private names", *entries]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return {"D1A_DATA_BLOCKLIST": str(path)}


def test_blocklist_matches_whole_names_and_never_repeats_them(tmp_path):
    env = blocklist_env(tmp_path, "Northwind", "ノースウィンド", "pat.example  # a comment after an entry")
    pattern, entries = suites.blocklist(env)
    assert entries == 3
    hits = ["Ask NORTHWIND ops", "repos/northwind/wms", "cd ~/northwind-kit", "社内のノースウィンド運用ページ", "mail pat.example today"]
    misses = ["northwindy", "a northwind_kit id", "pat.examples"]
    assert all(pattern.search(s) for s in hits) and not any(pattern.search(s) for s in misses)
    with pytest.raises(ValueError) as refused:
        suites.refuse_blocked(["clean line", "the Northwind board"], "mix.jsonl", env)
    assert "lines [2]" in str(refused.value) and "northwind" not in str(refused.value).lower()   # the log never repeats the name
    assert suites.refuse_blocked(["the Northwind board"], "mix.jsonl", env={}) is None   # unset: no check


def test_freeze_and_resolve_refuse_data_holding_a_blocked_name(tmp_path, monkeypatch):
    hub = tmp_path / "hub"; hub.mkdir()
    (hub / "train.jsonl").write_text('{"state": "route this Northwind ticket"}\n', encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, repo_type, revision: str(hub / path))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=lambda d, revision=None: SimpleNamespace(sha="abc123")))
    suite = tmp_path / "evals/d1a/toy"
    monkeypatch.setenv("D1A_DATA_BLOCKLIST", blocklist_env(tmp_path, "Northwind")["D1A_DATA_BLOCKLIST"])
    with pytest.raises(ValueError, match="general wording"):
        suites.freeze(suite, "me/toy", {"train": ("train.jsonl", "train")})
    assert not (suite / "manifest.json").exists()   # nothing pinned
    monkeypatch.delenv("D1A_DATA_BLOCKLIST")
    suites.freeze(suite, "me/toy", {"train": ("train.jsonl", "train")})   # without the list it pins as before
    monkeypatch.setenv("D1A_DATA_BLOCKLIST", str(tmp_path / "blocklist.txt"))
    with pytest.raises(ValueError, match="general wording"):
        suites.resolve(f"{suite}:train", purpose="train")
    (hub / "train.jsonl").write_text('{"state": "route this ticket"}\n', encoding="utf-8")
    m = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    m["partitions"]["train"]["sha256"] = suites.sha256(hub / "train.jsonl")
    (suite / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert suites.resolve(f"{suite}:train", purpose="train") == str(hub / "train.jsonl")   # clean data loads
