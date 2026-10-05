# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); the suites mirror pinned at kev-suites cc4bac8 as in upstream Kev 6b1da9d (adds the hard-v1 and documents-v1 train partitions); the suite freezer (freeze, main) and the helpers only it used removed.
import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from d1a.composition import DEV_SHAPES, HELD_OUT_KEYS, TEST_SHAPES, TRAIN_SHAPES
from d1a.data import EVAL_ONLY
from d1a.model import SERVE_MAX_BRANCH, SERVE_MAX_BRANCH_8K, SERVE_MAX_PACKED, SERVE_MAX_STATE, SERVE_MAX_STATE_8K, training_context

SPLITS = ("train", "calibration", "development", "test")
# the encoder limits every frozen record satisfies, as written into manifests ("context")
CONTEXT = {**training_context(), "truncate": False}
# what a manifest records for an eval-only suite frozen as published rather than admitted to the training context
SERVING_CONTEXT = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH, "max_packed": SERVE_MAX_PACKED, "truncate": False}
# the serving context before 64k states: what the suites frozen until then record and were admitted under; their builders
# (hard-v1, devtools-v1, breadth-v1, long states, semif/typesafe, documents-v1) keep it and d1a.model.MAX_TRAIN_STATE_8K,
# so they still rebuild byte for byte
SERVING_CONTEXT_8K = {"max_state": SERVE_MAX_STATE_8K, "max_branch": SERVE_MAX_BRANCH_8K, "max_packed": SERVE_MAX_STATE_8K + SERVE_MAX_BRANCH_8K, "truncate": False}
# Clean records are admitted with this many branch tokens to spare, so the variants that add an option (the suites'
# none-of-these variants, training-time none/distractor augmentation) still encode under MAX_BRANCH.
ADMISSION_BRANCH_HEADROOM = 64
# Frozen suites are mirrored on the Hub. Manifests (with the sha256 of every partition) and the development/test
# partitions live in git; large training partitions are fetched from this dataset on first use and verified against
# the manifest, so the suite hash and every provenance record stay unchanged. A suite whose partitions must never be
# public (a held-out test set) names its own mirror in the manifest, {"mirror": {"dataset": ..., "revision": ...}},
# usually the private PRIVATE_DATASET; only its manifest is in git, which publishes the hashes but not the text.
SUITES_DATASET = "jaredpalmer/kev-suites"
PRIVATE_DATASET = "jaredpalmer/kev-private-evals"
SUITES_REVISION = "cc4bac803e73112689ec327ffa481c519cbc7a05"
# partitions larger than this stay out of git (gitignored; the manifest's sha256 still pins them)
GIT_LIMIT = 10 * 1024 * 1024
# the pinned tokenizer suites built for the Qwen3.5 family are admitted and length-counted under (hard-v1, devtools-v1, long states)
ADMISSION_TOKENIZER = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
# programmatic policy sources in the frozen suites (legacy policy, compositional and contrastive records); the trainer's mix ablations treat them as one group
SYNTHETIC_SOURCES = ("legacy_policy", "compositional", "contrastive")
# Suites deleted from the repo because they are unsound as a gate: {suite dir: why, when, the last round that read it}.
# load_split / read_manifest refuse them with the reason (their committed rows under runs/ are the record of the rounds
# that read them, up to `last_round`).
REMOVED_SUITES = {
    "evals/external/scienthoon-v1": {
        "removed": "2026-09-27",
        "last_round": 22,
        "reason": ("unsound as a gate: 291 templated synthetic support tickets x 3 questions; `queue` is saturated (0.948-0.952 "
                   "for every 27B), `priority` is unlearnable by construction (its own manifest: the label follows an org rule "
                   "absent from the text), and `angry` has 15 of 291 gold labels that contradict the text and turns on ~12 stock "
                   "closing phrases with disputed conventions"),
        "record": "PLAN.md (Standing rules; 2026-09-27 note); committed rows under runs/ (e.g. runs/r20-scienthoon)",
    },
}


class RemovedSuite(ValueError):
    """A read of a suite in REMOVED_SUITES."""


def suite_key(path):
    """'evals/...' for a suite path given relative, absolute or as a container saw it (/root/kev/evals/x); None if none."""
    parts = Path(str(path)).parts
    return str(Path(*parts[parts.index("evals"):])) if "evals" in parts else None


def removed_suite(path):
    """The REMOVED_SUITES entry of a suite path (any form suite_key accepts), or None."""
    key = suite_key(path)
    return REMOVED_SUITES.get(key) if key else None


def refuse_removed(path):
    if entry := removed_suite(path):
        raise RemovedSuite(f"{suite_key(path)} was removed on {entry['removed']}: {entry['reason']}. Rounds up to "
                           f"{entry['last_round']} read it; their committed rows are the record ({entry['record']}). Do not read it again.")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record_digest(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalise_text(text):
    """Casefolded, whitespace runs collapsed to one space: the text two states are compared on for exact deduplication."""
    return " ".join(text.casefold().split())


def text_digest(text):
    """sha256 of normalise_text(text): the `text_sha256` of suite builders (d1a.data computes the same inline; it cannot
    import this module). scripts/screen_overlap.py tokenises differently on purpose (words only, for n-gram overlap)."""
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()


# Every JSON/JSONL file this repo writes is UTF-8 with LF line endings, whatever the platform's locale says (issue #12:
# frozen partitions are sha256-checked byte for byte, and they contain non-ASCII text). Read them the same way.
ENCODING = "utf-8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding=ENCODING))


def write_json(path, value, atomic=False):
    """atomic: write a sibling temp file and os.replace it over `path`, so a reader (or a restart after a crash mid-write)
    sees the old file or the new one, never a torn one. For state files rewritten in place."""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if not atomic:
        Path(path).write_text(text, encoding=ENCODING); return
    tmp = Path(path).with_name(f".{Path(path).name}.tmp")
    tmp.write_text(text, encoding=ENCODING)
    os.replace(tmp, path)


@contextmanager
def file_lock(path):
    """Hold an exclusive advisory lock on `path` (created if absent) for the block; a second holder waits. Local
    orchestration only (one writer of a shared run directory at a time)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding=ENCODING) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_jsonl(path):
    # split on "\n" only: str.splitlines() also breaks on U+2028, U+2029 and U+0085, which write_jsonl leaves unescaped
    return [json.loads(line) for line in Path(path).read_text(encoding=ENCODING).split("\n") if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding=ENCODING)


def validate_training(records, manifest):
    """Every training record must come from a source the manifest declares trainable, never from an eval-only or held-out
    one, and compositional records must not use a held-out rule structure."""
    allowed = set(manifest.get("trainable_sources", []))
    forbidden = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) | set(manifest.get("holdout_sources", []))
    for r in records:
        m = r["_meta"]
        if m["source"] in forbidden or (allowed and m["source"] not in allowed):
            raise ValueError(f"eval-only or undeclared training source: {m['source']}")
        if m["source"] == "compositional":
            held_shape = m["family"] in DEV_SHAPES + TEST_SHAPES
            held_structure = m.get("structure") in HELD_OUT_KEYS
            if held_shape or held_structure or (m["family"] not in TRAIN_SHAPES and not m["family"].startswith("rand:")):
                raise ValueError("held-out compositional structure in training")
    if not records:
        raise ValueError("empty training partition")


def read_manifest(directory):
    refuse_removed(directory)
    return read_json(Path(directory) / "manifest.json")


def load_split(directory, split, allow_test=False):
    refuse_removed(directory)   # a removed suite is refused with its reason, not a missing-file error
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test; never use it for search")
    directory = Path(directory)
    manifest = read_manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists():
        fetch_partition(directory, path.name)
    if digest(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != manifest["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records


def fetch_partition(directory, filename):
    """Download one partition of a frozen suite from its Hub mirror into place: the manifest's own "mirror" if it names
    one, else SUITES_DATASET@SUITES_REVISION. The caller verifies the sha256."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    directory = Path(directory).resolve()
    evals_root = next((p for p in directory.parents if p.name == "evals"), None)
    if evals_root is None:
        raise FileNotFoundError(f"{directory / filename} is missing and is not under an evals/ tree")
    relative = directory.relative_to(evals_root) / filename
    mirror = read_manifest(directory).get("mirror")
    repo, revision = (mirror["dataset"], mirror["revision"]) if mirror else (SUITES_DATASET, SUITES_REVISION)   # a named mirror pins its own revision
    try:
        cached = hf_hub_download(repo, str(relative), repo_type="dataset", revision=revision)
    except (RepositoryNotFoundError, GatedRepoError) as e:   # a private mirror answers "not found" to anyone without access
        raise PermissionError(f"{relative} is only in {repo}, which is missing or private to this account; `hf auth login` "
                              "(or HF_TOKEN) with access to it, or ask for it") from e
    shutil.copyfile(cached, directory / filename)
    print(f"fetched {relative} from {repo}@{revision[:10]}", flush=True)
