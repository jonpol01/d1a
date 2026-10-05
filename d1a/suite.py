"""Frozen evaluation suites: reading them as pinned data, and the file conventions everything in D1A writes with.

A suite is a directory under evals/ with a manifest.json and up to four partitions (SPLITS) as JSONL. The manifest pins
each partition's sha256 and record count, and load_split refuses a partition that does not match. Large partitions are
not in git: they download on first use from the suite's Hub mirror (fetch_partition) and are checked the same way. D1A
never rebuilds these suites; docs/removed-tools.md says which commit can.

The manifests record the encoder limits their records were admitted under (CONTEXT, SERVING_CONTEXT,
SERVING_CONTEXT_8K), and validate_training keeps eval-only and held-out data out of training.
"""
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

# The encoder limits a manifest records as its "context". CONTEXT: the training context, which every admitted record fits.
# SERVING_CONTEXT: an eval-only suite frozen as published. SERVING_CONTEXT_8K: the serving limits before 64k states, which
# the suites frozen until then record and were admitted under (hard-v1, devtools-v1, breadth-v1, the long states,
# semif/typesafe, documents-v1), so their builders still reproduce them byte for byte.
CONTEXT = {**training_context(), "truncate": False}
SERVING_CONTEXT = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH, "max_packed": SERVE_MAX_PACKED, "truncate": False}
SERVING_CONTEXT_8K = {"max_state": SERVE_MAX_STATE_8K, "max_branch": SERVE_MAX_BRANCH_8K, "max_packed": SERVE_MAX_STATE_8K + SERVE_MAX_BRANCH_8K, "truncate": False}
# Branch tokens a clean record was admitted with to spare, so the variants that add an option (the suites' none-of-these
# variants, and d1a.train's none and distractor augmentation) still encode under MAX_BRANCH.
ADMISSION_BRANCH_HEADROOM = 64

# The Hub mirrors. A manifest and the development and test partitions are in git; a large training partition downloads
# from SUITES_DATASET at SUITES_REVISION and is checked against the manifest, so suite hashes and provenance never change.
# A suite whose partitions must stay private (a held-out test set) names its own mirror in the manifest,
# {"mirror": {"dataset": ..., "revision": ...}}, usually PRIVATE_DATASET: git then holds its hashes but not its text.
SUITES_DATASET = "jaredpalmer/kev-suites"
PRIVATE_DATASET = "jaredpalmer/kev-private-evals"
SUITES_REVISION = "cc4bac803e73112689ec327ffa481c519cbc7a05"
GIT_LIMIT = 10 * 1024 * 1024   # a partition larger than this stays out of git (gitignored; the manifest still pins it)
# the pinned tokenizer that the suites built for the Qwen3.5 family were admitted and counted under (hard-v1, devtools-v1, long states)
ADMISSION_TOKENIZER = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
# the programmatic policy sources in the frozen suites (legacy policy, compositional and contrastive records), which the
# trainer's mix options treat as one group
SYNTHETIC_SOURCES = ("legacy_policy", "compositional", "contrastive")

# Suites deleted because they are unsound as a gate, with why, when, and the last round that read them. Reading one is
# refused with that reason; the rows those rounds committed under runs/ remain the record.
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
    """A suite path from its "evals" component on ("evals/..."), however it was given (relative, absolute, or as a container
    saw it, /root/kev/evals/x); None for a path with no "evals" component."""
    parts = Path(str(path)).parts
    if "evals" not in parts:
        return None
    return str(Path(*parts[parts.index("evals"):]))


def removed_suite(path):
    """The REMOVED_SUITES entry for a suite path in any form suite_key reads, or None."""
    key = suite_key(path)
    return REMOVED_SUITES.get(key) if key else None


def refuse_removed(path):
    entry = removed_suite(path)
    if entry:
        raise RemovedSuite(f"{suite_key(path)} was removed on {entry['removed']}: {entry['reason']}. Rounds up to "
                           f"{entry['last_round']} read it; their committed rows are the record ({entry['record']}). Do not read it again.")


# --- hashes -------------------------------------------------------------------------------------------------------------

def digest(path):
    """sha256 of a file's bytes, read a megabyte at a time."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            sha.update(block)
    return sha.hexdigest()


def record_digest(record):
    """sha256 of a JSON value in its compact form, non-ASCII kept as UTF-8."""
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalise_text(text):
    """The text two states are compared on for exact deduplication: casefolded, each run of whitespace one space."""
    return " ".join(text.casefold().split())


def text_digest(text):
    """sha256 of normalise_text(text): a record's `text_sha256` (d1a.data computes it inline, since it cannot import this
    module)."""
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()


# --- files --------------------------------------------------------------------------------------------------------------

# Every JSON and JSONL file is UTF-8, whatever the platform's locale: frozen partitions are hash-checked byte for byte and
# hold non-ASCII text (issue #12).
ENCODING = "utf-8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding=ENCODING))


def write_json(path, value, atomic=False):
    """Indented JSON, UTF-8, with a final newline; NaN and infinity are refused. atomic: write a hidden sibling and rename
    it over `path`, so a reader, or a restart after a crash mid-write, sees the old file or the new one, never half of one."""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target = Path(path)
    if atomic:
        staged = target.with_name(f".{target.name}.tmp")
        staged.write_text(text, encoding=ENCODING)
        os.replace(staged, path)
    else:
        target.write_text(text, encoding=ENCODING)


@contextmanager
def file_lock(path):
    """An exclusive advisory lock on `path` (created if absent) for the block: a second holder waits. For one writer of a
    shared run directory at a time, on one machine."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding=ENCODING) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def read_jsonl(path):
    """One JSON value per non-blank line. Lines end at "\\n" only: str.splitlines() would also break at U+2028, U+2029 and
    U+0085, which write_jsonl leaves as they are."""
    lines = Path(path).read_text(encoding=ENCODING).split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding=ENCODING)


# --- suites -------------------------------------------------------------------------------------------------------------

def held_out_compositional(meta):
    """Whether a compositional record was built on a held-out rule shape or structure (or a shape that is not a training
    one, unless it is a random tree, "rand:...")."""
    family = meta["family"]
    return (family in DEV_SHAPES + TEST_SHAPES or meta.get("structure") in HELD_OUT_KEYS
            or (family not in TRAIN_SHAPES and not family.startswith("rand:")))


def validate_training(records, manifest):
    """Refuse training records from a source the manifest does not declare trainable, or one that is eval-only or held out,
    and compositional records on a held-out rule structure. An empty partition is refused too."""
    trainable = set(manifest.get("trainable_sources", []))
    refused = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) | set(manifest.get("holdout_sources", []))
    for record in records:
        meta = record["_meta"]
        source = meta["source"]
        if source in refused or (trainable and source not in trainable):
            raise ValueError(f"eval-only or undeclared training source: {source}")
        if source == "compositional" and held_out_compositional(meta):
            raise ValueError("held-out compositional structure in training")
    if not records:
        raise ValueError("empty training partition")


def read_manifest(directory):
    refuse_removed(directory)
    return read_json(Path(directory) / "manifest.json")


def load_split(directory, split, allow_test=False):
    """A partition's records, after checking its sha256 and record count against the manifest (downloading it first if it
    is not here). The test partition needs allow_test: it is locked, never for search."""
    refuse_removed(directory)   # with its reason, rather than a missing-file error
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test; never use it for search")
    directory = Path(directory)
    manifest = read_manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists():
        fetch_partition(directory, path.name)
    pinned = manifest["files"][path.name]
    if digest(path) != pinned["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != pinned["records"]:
        raise ValueError("suite record count mismatch")
    return records


def fetch_partition(directory, filename):
    """Download one partition into the suite directory from its mirror: the manifest's own "mirror" when it names one
    (which pins its own revision), else SUITES_DATASET at SUITES_REVISION. The caller checks the sha256."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    directory = Path(directory).resolve()
    evals = next((parent for parent in directory.parents if parent.name == "evals"), None)
    if evals is None:
        raise FileNotFoundError(f"{directory / filename} is missing and is not under an evals/ tree")
    relative = directory.relative_to(evals) / filename
    mirror = read_manifest(directory).get("mirror")
    repo, revision = (mirror["dataset"], mirror["revision"]) if mirror else (SUITES_DATASET, SUITES_REVISION)
    try:
        cached = hf_hub_download(repo, str(relative), repo_type="dataset", revision=revision)
    except (RepositoryNotFoundError, GatedRepoError) as error:   # a private mirror answers "not found" to anyone without access
        raise PermissionError(f"{relative} is only in {repo}, which is missing or private to this account; `hf auth login` "
                              "(or HF_TOKEN) with access to it, or ask for it") from error
    shutil.copyfile(cached, directory / filename)
    print(f"fetched {relative} from {repo}@{revision[:10]}", flush=True)
