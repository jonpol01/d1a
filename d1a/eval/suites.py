"""D1A's own suites: our labelled datasets (System One requests with labels, as d1a.training.data.load_records reads them) frozen
at a pinned Hugging Face revision, so a training mix or a score names exactly the data it used.

A suite is a folder under evals/d1a/ holding one manifest.json; the text stays in the (private) dataset and git holds
only its hashes:

    {"format": "d1a-suite", "version": 1, "dataset": "JohnP1/d1a-pr-labels", "revision": "<commit>",
     "partitions": {"train": {"path": "train.jsonl", "role": "train", "sha256": "...", "records": 4210},
                    "development": {"path": "development.jsonl", "role": "eval", ...}}}

`evals/d1a/<suite>:<partition>` is accepted wherever a labelled file is (d1a.training.train --data, d1a.eval.benchmark --data): the
partition is fetched at the pinned revision and checked against its sha256 and record count before use. d1a.training.train
refuses an `eval` partition, so a held-out set cannot leak into training.

D1A_DATA_BLOCKLIST names a private file of names this data must never hold (an organisation, its products, people). When
it is set, freeze refuses to pin a partition that holds one, resolve refuses to load one, and d1a.training.train refuses to
train on one; the names stay in that file, never in the repository.

    uv run python -m d1a.eval.suites freeze evals/d1a/pr-labels --dataset JohnP1/d1a-pr-labels \\
        --partition train=train.jsonl:train --partition development=development.jsonl:eval
    uv run python -m d1a.eval.suites verify evals/d1a/pr-labels      # fetch every partition and check it
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from d1a.core.settings import get

FORMAT, VERSION = "d1a-suite", 1
ROLES = ("train", "eval")
MANIFEST = "manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def records(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def is_reference(arg):
    """Whether `arg` names a suite partition (`<suite dir>:<partition>`, the dir holding a d1a-suite manifest)."""
    suite, sep, _ = str(arg).rpartition(":")
    return bool(sep) and (Path(suite) / MANIFEST).is_file()


def manifest(suite):
    m = json.loads((Path(suite) / MANIFEST).read_text(encoding="utf-8"))
    if (m.get("format"), m.get("version")) != (FORMAT, VERSION):
        raise ValueError(f"{suite}/{MANIFEST} is not a {FORMAT} v{VERSION} manifest")
    return m


def blocklist(env=None):
    """(pattern, entries) for D1A_DATA_BLOCKLIST, or None when it is unset. One entry per line, `#` starts a comment, case
    ignored. An entry's ASCII letter or digit ends match only at a word edge, so a short name does not hit inside a longer
    word; other characters (CJK) match anywhere."""
    path = get("D1A_DATA_BLOCKLIST", env)
    if not path:
        return None
    entries = [e for e in (line.split("#", 1)[0].strip() for line in Path(path).read_text(encoding="utf-8").splitlines()) if e]
    if not entries:
        raise ValueError("D1A_DATA_BLOCKLIST names a file with no entries")
    word = "[A-Za-z0-9_]"
    parts = [(f"(?<!{word})" if re.match(word, e) else "") + re.escape(e) + (f"(?!{word})" if re.search(f"{word}$", e) else "")
             for e in entries]
    return re.compile("|".join(parts), re.IGNORECASE), len(entries)


def refuse_blocked(lines, where, env=None):
    """Raise when any of `lines` (a file's path, or strings) holds a D1A_DATA_BLOCKLIST entry. The message gives the line
    numbers, never the entry, so a refusal does not repeat the name in a log. -> the number of entries checked (None: unset)."""
    rule = blocklist(env)
    if rule is None:
        return None
    pattern, n = rule
    if isinstance(lines, (str, Path)):
        lines = Path(lines).read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(lines, 1) if pattern.search(line)]
    if hits:
        raise ValueError(f"{where}: {len(hits)} line(s) hold a D1A_DATA_BLOCKLIST entry (lines {hits[:5]}); "
                         "rewrite them in general wording before this data is used")
    return n


def resolve(arg, purpose="eval"):
    """`<suite dir>:<partition>` -> the verified local file of that partition (fetched at the pinned revision into the
    Hugging Face cache); any other argument is returned unchanged. purpose "train" refuses a partition whose role is eval."""
    if not is_reference(arg): return arg
    suite, _, name = str(arg).rpartition(":")
    m = manifest(suite)
    if name not in m["partitions"]:
        raise ValueError(f"{suite} has no partition {name!r} (it has {sorted(m['partitions'])})")
    part = m["partitions"][name]
    if purpose == "train" and part["role"] != "train":
        raise ValueError(f"{arg} is an {part['role']} partition; training on it would leak it into the model it scores")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(m["dataset"], part["path"], repo_type="dataset", revision=m["revision"])
    if sha256(path) != part["sha256"] or records(path) != part["records"]:
        raise ValueError(f"{arg}: {m['dataset']}@{m['revision'][:10]}/{part['path']} does not match the manifest")
    refuse_blocked(path, arg)
    return path


SKILLS = ("evals/hard-v1", "evals/devtools-v1", "evals/documents-v1")   # Kev's frozen suites with a train partition (decision-v7's is replayed by d1a.training.train itself)


def train_sources(root=Path(__file__).resolve().parents[2]):
    """Every training source a fine-tune replays so it keeps what earlier stages taught it (#167): the frozen skills
    suites' train partitions, then every D1A suite partition whose role is train, as evals/d1a/<suite>:<partition>."""
    out = list(SKILLS)
    for m in sorted((Path(root) / "evals/d1a").glob(f"*/{MANIFEST}")):
        parts = json.loads(m.read_text(encoding="utf-8"))["partitions"]
        out += [f"evals/d1a/{m.parent.name}:{name}" for name, p in parts.items() if p["role"] == "train"]
    return out


def freeze(suite, dataset, partitions, revision=None, provenance=None):
    """Write `suite`/manifest.json for `partitions` {name: (path in the dataset, role)} of `dataset` at `revision` (default:
    its current commit), hashing each file as it is there. `provenance`: where the data came from, kept in the manifest."""
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().dataset_info(dataset, revision=revision).sha
    parts = {}
    for name, (path, role) in partitions.items():
        if role not in ROLES: raise ValueError(f"partition {name}: role must be one of {ROLES}, not {role!r}")
        local = hf_hub_download(dataset, path, repo_type="dataset", revision=revision)
        refuse_blocked(local, f"{dataset}@{revision[:10]}/{path}")   # a pin never names what the data must not hold
        parts[name] = {"path": path, "role": role, "sha256": sha256(local), "records": records(local)}
    m = {"format": FORMAT, "version": VERSION, "dataset": dataset, "revision": revision, "partitions": parts}
    if provenance: m["provenance"] = provenance
    Path(suite).mkdir(parents=True, exist_ok=True)
    (Path(suite) / MANIFEST).write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="pin a dataset's partitions into <suite>/manifest.json")
    f.add_argument("suite"); f.add_argument("--dataset", required=True); f.add_argument("--revision")
    f.add_argument("--partition", action="append", required=True, metavar="NAME=PATH:ROLE")
    f.add_argument("--provenance", help="where the data came from (kept in the manifest)")
    v = sub.add_parser("verify", help="fetch every partition of a suite and check it against the manifest")
    v.add_argument("suite")
    a = ap.parse_args()
    if a.command == "freeze":
        partitions = {}
        for spec in a.partition:
            name, _, rest = spec.partition("="); path, _, role = rest.rpartition(":")
            partitions[name] = (path, role)
        m = freeze(a.suite, a.dataset, partitions, a.revision, a.provenance)
        print(f"{a.suite}: {a.dataset}@{m['revision'][:10]}, " + ", ".join(f"{n} {p['records']}" for n, p in m["partitions"].items()))
    else:
        for name in manifest(a.suite)["partitions"]:
            print(f"{a.suite}:{name} ok ({resolve(f'{a.suite}:{name}')})")


if __name__ == "__main__":
    main()
