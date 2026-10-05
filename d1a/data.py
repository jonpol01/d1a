"""Labelled requests: built from public datasets (build), read from your own JSONL (load_records), varied for training
(augment, none_pair), and turned into the records the model reads (materialize) through the same path a
/v1/systemone request takes (d1a.api.to_record), so training sees exactly what serving does.

A labelled request is a System One request with a label on every question:

    {"state": <JSON content>, "questions": {id: {"type", "instructions", "criteria", "label", "src"}}, "_meta": {...}}

    label: the option key (choice), true or false (noul), the level index (score)

materialize() returns the internal record {"state": str, "questions": [{"instr", "options", "label": int, "src", ...}]}.

Every random choice comes from a seeded generator, and each converter draws in a fixed order, so a given seed always
yields the same records: the frozen suites and training runs depend on it.
"""
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

from .api import SystemOneRequest, to_record

REPOS = {"banking77": "legacy-datasets/banking77", "boolq": "google/boolq", "agnews": "fancyzhx/ag_news",
         "mnli": "nyu-mll/multi_nli", "sst5": "SetFit/sst5", "yelp": "Yelp/yelp_review_full"}


def source_seed(seed, source):
    """A 64-bit seed for one source (or any other label) under a run's seed, independent of every other source's."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def normalised_sha256(text):
    """sha256 of the casefolded text with whitespace runs collapsed (d1a.suite.text_digest; this module cannot import it)."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


class Source(random.Random):
    """One source's sampling: the seeded generator its converter draws everything from, the dataset revision to pin, and
    the provenance of each row drawn (`origins`, which sample() fills and build() attaches to the records)."""

    def __init__(self, seed, revision=None):
        super().__init__(seed)
        self.revision = revision
        self.origins = []


def _dataset(repo, split, src):
    """A dataset split; `repo` is "owner/name" or "owner/name:config", and the revision is the Source's."""
    name, _, config = repo.partition(":")
    return load_dataset(name, config or None, split=split, revision=src.revision)


# A "none of the above" option has to appear both as the right answer and as a wrong one, in varied wording, or the model
# learns to pick the wording (it did, in the first training run).
NONE = "None of the above"
NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
                ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
                ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
                ("other", "An answer not covered by the other options"), ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}

# The label texts of the public sources.
AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance", "scitech": "Science and technology: research, gadgets, software, space"}
MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise", "contradiction": "The hypothesis contradicts the premise"}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP = ["1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
BANK_TEMPLATES = ["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"]
FOCUS = ["Use only the information given.", "Pick the single best fit.", "Consider the whole message."]
CHANNELS = ["email", "chat", "web form"]


# --- varying the request shape (so the model sees every JSON form a caller may send) ----------------------------------

def _wrap_state(text, rng):
    """The state as plain text (68%), a {"document"} object (15%), a ticket object (10%) or a one-message chat (7%)."""
    draw = rng.random()
    if draw < 0.15:
        return {"document": text}
    if draw < 0.25:
        return {"ticket": {"channel": rng.choice(CHANNELS), "body": text}}
    if draw < 0.32:
        return [{"role": "customer", "content": text}]
    return text


def _instr(text, rng):
    """The instructions as text, or 15% of the time as a {"question", "focus"} object."""
    if rng.random() < 0.15:
        return {"question": text, "focus": rng.choice(FOCUS)}
    return text


def _desc(desc, rng, p_null=0.3, p_struct=0.1):
    """An option's description, sometimes dropped (None) or given as a {"what"} object."""
    draw = rng.random()
    if draw < p_null:
        return None
    if draw < p_null + p_struct:
        return {"what": desc}
    return desc


TEXT_FIELDS = ("text", "premise", "passage", "content", "question", "sentence")


def _sample(ds, n, src):
    """Up to n rows drawn without replacement (unlabelled rows, label -1, skipped), recording each kept row's provenance:
    its index, the hash of its normalised text and the hash of the whole row."""
    kept, src.origins = [], []
    for i in src.sample(range(len(ds)), min(n, len(ds))):
        row = ds[i]
        if row.get("label", 0) == -1:
            continue
        text = next((row[field] for field in TEXT_FIELDS if isinstance(row.get(field), str)), json.dumps(row, sort_keys=True))
        src.origins.append({"row": i, "text_sha256": normalised_sha256(text),
                            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
        kept.append(row)
    return kept


# --- the public sources (each draws from its Source in a fixed order) --------------------------------------------------

def request(state, **questions):
    return {"state": state, "questions": questions}


def _banking(split, n, rng):
    ds = _dataset("legacy-datasets/banking77", split=split, src=rng)
    intents = ds.features["label"].names
    out = []
    for ex in _sample(ds, n, rng):
        template = rng.choice(BANK_TEMPLATES)
        criteria = {k: _desc(template.format(k.replace("_", " ")), rng, p_null=0.5, p_struct=0.0) for k in intents}
        state = _wrap_state(ex["text"], rng)
        out.append(request(state, intent={"type": "choice", "instructions": _instr("Which banking intent best describes this customer message?", rng),
                                          "criteria": criteria, "label": intents[ex["label"]], "src": "banking77"}))
    return out


def _boolq(split, n, rng):
    ds = _dataset("google/boolq", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        q = {"type": "noul", "instructions": _instr(ex["question"].strip().rstrip("?") + "?", rng), "label": bool(ex["answer"]), "src": "boolq"}
        if rng.random() < 0.4:
            q["criteria"] = {"true": "The passage supports a yes answer", "false": "The passage supports a no answer or does not say"}
        out.append(request(_wrap_state(ex["passage"], rng), answer=q))
    return out


def _agnews(split, n, rng):
    ds = _dataset("fancyzhx/ag_news", split=split, src=rng)
    topics = list(AG)
    out = []
    for ex in _sample(ds, n, rng):
        topic = topics[ex["label"]]
        questions = {"topic": {"type": "choice", "instructions": _instr("What is the topic of this article?", rng),
                               "criteria": {k: _desc(v, rng) for k, v in AG.items()}, "label": topic, "src": "agnews"}}
        for k in rng.sample(topics, 2):   # two yes/no topic questions, one of which may be the answer
            questions[f"is_{k}"] = {"type": "noul", "instructions": f"Is this article about {AG[k].split(':')[0].lower()}?",
                                    "label": k == topic, "src": "agnews_yn"}
        out.append(request(_wrap_state(ex["text"], rng), **questions))
    return out


def _mnli(split, n, rng):
    ds = _dataset("nyu-mll/multi_nli", split=split, src=rng)
    relations = list(MNLI)
    out = []
    for ex in _sample(ds, n, rng):
        if ex["label"] < 0:
            continue
        state = _wrap_state(ex["premise"], rng)
        instructions = _instr(f'Hypothesis: "{ex["hypothesis"]}" How does it relate to the premise?', rng)
        criteria = {k: _desc(v, rng) for k, v in MNLI.items()}
        out.append(request(state, relation={"type": "choice", "instructions": instructions, "criteria": criteria, "label": relations[ex["label"]], "src": "mnli"}))
    return out


def _sst5(split, n, rng):
    ds = _dataset("SetFit/sst5", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        state = _wrap_state(ex["text"], rng)
        out.append(request(state, sentiment={"type": "score", "instructions": _instr("What is the sentiment of this review sentence?", rng),
                                             "criteria": list(SST5), "label": ex["label"], "src": "sst5"}))
    return out


def _yelp(split, n, rng):
    ds = _dataset("Yelp/yelp_review_full", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        text = " ".join(ex["text"].split()[:220])   # the first 220 words
        rating = {"type": "score", "instructions": _instr("How many stars did this reviewer give?", rng), "criteria": list(YELP), "label": ex["label"], "src": "yelp"}
        recommend = {"type": "noul", "instructions": "Would this reviewer recommend the business?",
                     "criteria": {"true": "Clearly positive overall", "false": "Negative or mixed"}, "label": ex["label"] >= 3, "src": "yelp_yn"}
        out.append(request(_wrap_state(text, rng), rating=rating, recommend=recommend))
    return out


# source -> (converter, training split, evaluation split)
SOURCES = {"banking77": (_banking, "train", "test"), "boolq": (_boolq, "train", "validation"), "agnews": (_agnews, "train", "test"),
           "mnli": (_mnli, "train", "validation_matched"), "sst5": (_sst5, "train", "test"), "yelp": (_yelp, "train", "test")}

# Sources training refuses (d1a.train, d1a.suite.validate_training): the frozen suites hold records from these, kept for
# evaluation only. MMLU is a knowledge probe; Emotion/TweetEval are noisy-label honesty checks; QNLI/PAWS/SciQ measure
# reading transfer.
EVAL_ONLY = ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq")


def build(n_per_source, split="train", seed=0, exclude=(), only=(), revisions=None, sources=None, repos=None):
    """Up to n_per_source labelled requests from each source (SOURCES unless given; `only` and `exclude` pick), from its
    training split or ("test") its evaluation split, each with its provenance in "_meta", shuffled together. Each source
    draws from its own seed (source_seed), so adding or dropping a source never changes another's records."""
    sources = SOURCES if sources is None else sources
    repos = REPOS if repos is None else repos
    unknown = (set(exclude) | set(only)) - sources.keys()
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    requests = []
    for name, (convert, train_split, eval_split) in sources.items():
        if name in exclude or (only and name not in only):
            continue
        src = Source(source_seed(seed, name), (revisions or {}).get(repos[name]))
        source_split = train_split if split == "train" else eval_split
        records = convert(source_split, n_per_source, src)
        if len(records) != len(src.origins):
            raise ValueError(f"provenance mismatch for {name}")
        for record, origin in zip(records, src.origins):
            record["_meta"] = {**origin, "source": name, "repo": repos[name], "revision": src.revision,
                               "split": source_split, "id": f"{name}/{source_split}/{origin['row']}"}
        requests += records
    random.Random(seed).shuffle(requests)
    return requests


# --- training-time variation --------------------------------------------------------------------------------------------

def shuffled(criteria, keys, rng):
    rng.shuffle(keys)
    return {k: criteria[k] for k in keys}


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """The request with each Choice question's options shuffled, and with one of three changes drawn per question: a
    none-of-the-above option replaces the right one and becomes the label (p_none, 3+ options), a none option is added as
    a wrong alternative (p_none_distract), or an unrelated distractor is added (p_distract). Questions with a soft target
    are only shuffled: adding or removing options would change what the target means. Other types are left as they are."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        criteria, label = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:
            out["questions"][qid] = {**q, "criteria": shuffled(criteria, list(criteria), rng)}
            continue
        draw = rng.random()
        unused_none = [(k, v) for k, v in NONE_OPTIONS if k not in criteria]
        unused_distractors = [k for k in DISTRACTORS if k not in criteria]
        if draw < p_none and len(criteria) > 2 and unused_none:
            key, text = rng.choice(unused_none)
            criteria.pop(label); criteria[key] = text; label = key
        elif p_none <= draw < p_none + p_none_distract and len(criteria) < 255 and unused_none:
            key, text = rng.choice(unused_none)
            criteria[key] = text
        elif p_none + p_none_distract <= draw < p_none + p_none_distract + p_distract and len(criteria) < 255 and unused_distractors:
            key = rng.choice(unused_distractors)
            criteria[key] = DISTRACTORS[key]
        out["questions"][qid] = {**q, "criteria": shuffled(criteria, list(criteria), rng), "label": label}
    return out


def none_pair(req, rng):
    """A minimal pair for the none-of-the-above shortcut: one Choice question rendered twice, once with its right option
    present (an added none option is wrong) and once with it removed (the same none option is right), in the same option
    order, so only whether the evidence matches an option differs. [] when no Choice qualifies (3+ options, no soft
    target, as in augment)."""
    eligible = [(qid, q) for qid, q in req["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3 and q.get("target") is None]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    none_key, none_text = rng.choice([o for o in NONE_OPTIONS if o[0] not in q["criteria"]] or [("none_of_these", None)])
    order = list(q["criteria"]) + [none_key]
    rng.shuffle(order)
    present = {**q, "criteria": {k: none_text if k == none_key else q["criteria"][k] for k in order}}
    absent = {**present, "criteria": {k: v for k, v in present["criteria"].items() if k != q["label"]}, "label": none_key}
    return [{"state": req["state"], "questions": {qid: present}}, {"state": req["state"], "questions": {qid: absent}}]


# --- your own data, and the model's records -----------------------------------------------------------------------------

def load_records(path, source="custom"):
    """Labelled requests from a JSONL file, one per line, in the API's request shape with a `label` on every question:

        {"state": "...", "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "other": null}, "label": "billing"},
                                       "urgent": {"type": "noul", "instructions": "...", "label": true},
                                       "priority": {"type": "score", "instructions": "...", "criteria": ["low", "medium", "high"], "label": 2}}}

    Labels: the option name for choice, true/false for noul, the level index (from 0) for score. Each record gets a `_meta`
    and each question a `src` (unless it has them), so the records behave like a frozen suite's: source = `source`, id =
    the line number."""
    records = []
    with Path(path).open(encoding="utf-8") as lines:
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            record = json.loads(line)
            if "state" not in record or not isinstance(record.get("questions"), dict) or not record["questions"]:
                raise ValueError(f"{path}:{n + 1}: a record needs a state and a non-empty questions object")
            for qid, q in record["questions"].items():
                if "label" not in q:
                    raise ValueError(f"{path}:{n + 1}: question {qid!r} has no label")
                q.setdefault("src", f"{source}_{q['type']}")
            state = record["state"]
            text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
            defaults = {"source": source, "variant": "clean", "id": f"{source}/{n}", "group_id": f"{source}/{n}", "row": n, "split": "custom",
                        "text_sha256": normalised_sha256(text)}
            record["_meta"] = {**defaults, **record.get("_meta", {})}
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def api_request(record):
    """The /v1/systemone body for a labelled record: the state and the typed questions, never labels, targets or metadata
    (what leaves the machine when a remote predictor is scored)."""
    return {"state": record["state"], "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")} for qid, q in record["questions"].items()}}


def materialize(req):
    """A labelled request as the model's record, built by the serving path (d1a.api.to_record), with each question's label
    as an option index and its src, type, id and keys. A soft target ({key: weight}: option names, "false"/"true", or level
    indices as strings) becomes a distribution over the options, unnamed ones 0 (unknowable records: uniform)."""
    rec, meta = to_record(SystemOneRequest.model_validate(api_request(req)))
    for q, info, (qid, labelled) in zip(rec["questions"], meta, req["questions"].items()):
        label = labelled["label"]
        q["label"] = info["keys"].index(label) if info["type"] == "choice" else int(label)   # noul labels are bools, score labels indices
        q["src"], q["qtype"], q["qid"], q["keys"] = labelled["src"], info["type"], qid, info["keys"]
        if labelled.get("target") is not None:
            weights = [float(labelled["target"].get(k, 0.0)) for k in q["keys"]]
            if sum(weights) <= 0:
                raise ValueError(f"target for {qid} puts no mass on any option")
            q["target"] = [w / sum(weights) for w in weights]
    return rec
