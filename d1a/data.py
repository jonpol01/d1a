# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: the converters of the sources only Kev's suite freezer read (trec, dbpedia14, emotion, imdb, amazon, qnli, tweet_offensive, mmlu, paws, sciq, arc, openbookqa, csqa) removed; build() keeps the six default sources.
"""Convert public labelled datasets into TypeSafe-shaped requests, then through api.to_record() so the
training format is byte-identical to what /v1/systemone feeds the model.

A "labelled request" is {"state": JSONContent, "questions": {id: {type, instructions, criteria, "label": ..., "src": str}}}
  label: choice -> option key, noul -> bool, score -> level index.
materialize() -> internal record {"state": str, "questions": [{"instr", "options", "label": int, "src"}]}
"""
import hashlib
import json
from pathlib import Path
import random
from datasets import load_dataset
from .api import SystemOneRequest, to_record


REPOS = {"banking77": "legacy-datasets/banking77", "boolq": "google/boolq", "agnews": "fancyzhx/ag_news",
         "mnli": "nyu-mll/multi_nli", "sst5": "SetFit/sst5", "yelp": "Yelp/yelp_review_full"}


def source_seed(seed, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


class Source(random.Random):
    """One source's sampling context: a seeded RNG (the converters draw everything from it), the dataset revision to pin,
    and the provenance of every row drawn (`origins`, filled by _sample so build() can attach it to each record)."""

    def __init__(self, seed, revision=None):
        super().__init__(seed)
        self.revision = revision
        self.origins = []


def _dataset(repo, split, src):
    """repo may be 'owner/name' or 'owner/name:config'; the Source carries the revision to pin."""
    name, _, config = repo.partition(":")
    return load_dataset(name, config or None, split=split, revision=src.revision)

NONE = "None of the above"
# "None of the above" options must appear both as the correct answer and as a wrong alternative, with varied
# wording, or the model learns "this wording => pick it" (it did, in the first training run).
NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
                ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
                ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
                ("other", "An answer not covered by the other options"), ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}

AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance", "scitech": "Science and technology: research, gadgets, software, space"}
MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise", "contradiction": "The hypothesis contradicts the premise"}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP = ["1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
BANK_TEMPLATES = ["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"]


def _wrap_state(text, rng):
    r = rng.random()
    if r < 0.15: return {"document": text}
    if r < 0.25: return {"ticket": {"channel": rng.choice(["email", "chat", "web form"]), "body": text}}
    if r < 0.32: return [{"role": "customer", "content": text}]
    return text


def _instr(text, rng):
    return {"question": text, "focus": rng.choice(["Use only the information given.", "Pick the single best fit.", "Consider the whole message."])} if rng.random() < 0.15 else text


def _desc(desc, rng, p_null=0.3, p_struct=0.1):
    r = rng.random()
    if r < p_null: return None
    if r < p_null + p_struct: return {"what": desc}
    return desc


def _sample(ds, n, src):
    rows = []
    src.origins = []
    for i in src.sample(range(len(ds)), min(n, len(ds))):
        row = ds[i]
        if row.get("label", 0) == -1:
            continue
        text = next((row[k] for k in ("text", "premise", "passage", "content", "question", "sentence") if isinstance(row.get(k), str)), json.dumps(row, sort_keys=True))
        normalized = " ".join(text.casefold().split())
        src.origins.append({"row": i, "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
        rows.append(row)
    return rows


def _banking(split, n, rng):
    ds = _dataset("legacy-datasets/banking77", split=split, src=rng)
    names = ds.features["label"].names
    out = []
    for ex in _sample(ds, n, rng):
        t = rng.choice(BANK_TEMPLATES)
        crit = {k: _desc(t.format(k.replace("_", " ")), rng, p_null=0.5, p_struct=0.0) for k in names}
        out.append({"state": _wrap_state(ex["text"], rng), "questions": {"intent": {"type": "choice", "instructions": _instr("Which banking intent best describes this customer message?", rng), "criteria": crit, "label": names[ex["label"]], "src": "banking77"}}})
    return out


def _boolq(split, n, rng):
    ds = _dataset("google/boolq", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        q = {"type": "noul", "instructions": _instr(ex["question"].strip().rstrip("?") + "?", rng), "label": bool(ex["answer"]), "src": "boolq"}
        if rng.random() < 0.4: q["criteria"] = {"true": "The passage supports a yes answer", "false": "The passage supports a no answer or does not say"}
        out.append({"state": _wrap_state(ex["passage"], rng), "questions": {"answer": q}})
    return out


def _agnews(split, n, rng):
    ds = _dataset("fancyzhx/ag_news", split=split, src=rng)
    keys = list(AG)
    out = []
    for ex in _sample(ds, n, rng):
        y = keys[ex["label"]]
        qs = {"topic": {"type": "choice", "instructions": _instr("What is the topic of this article?", rng), "criteria": {k: _desc(v, rng) for k, v in AG.items()}, "label": y, "src": "agnews"}}
        for k in rng.sample(keys, 2):
            qs[f"is_{k}"] = {"type": "noul", "instructions": f"Is this article about {AG[k].split(':')[0].lower()}?", "label": k == y, "src": "agnews_yn"}
        out.append({"state": _wrap_state(ex["text"], rng), "questions": qs})
    return out


def _mnli(split, n, rng):
    ds = _dataset("nyu-mll/multi_nli", split=split, src=rng)
    keys = list(MNLI)
    return [{"state": _wrap_state(ex["premise"], rng), "questions": {"relation": {"type": "choice", "instructions": _instr(f'Hypothesis: "{ex["hypothesis"]}" How does it relate to the premise?', rng), "criteria": {k: _desc(v, rng) for k, v in MNLI.items()}, "label": keys[ex["label"]], "src": "mnli"}}}
            for ex in _sample(ds, n, rng) if ex["label"] >= 0]


def _sst5(split, n, rng):
    ds = _dataset("SetFit/sst5", split=split, src=rng)
    return [{"state": _wrap_state(ex["text"], rng), "questions": {"sentiment": {"type": "score", "instructions": _instr("What is the sentiment of this review sentence?", rng), "criteria": list(SST5), "label": ex["label"], "src": "sst5"}}} for ex in _sample(ds, n, rng)]


def _yelp(split, n, rng):
    ds = _dataset("Yelp/yelp_review_full", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        text = " ".join(ex["text"].split()[:220])
        qs = {"rating": {"type": "score", "instructions": _instr("How many stars did this reviewer give?", rng), "criteria": list(YELP), "label": ex["label"], "src": "yelp"},
              "recommend": {"type": "noul", "instructions": "Would this reviewer recommend the business?", "criteria": {"true": "Clearly positive overall", "false": "Negative or mixed"}, "label": ex["label"] >= 3, "src": "yelp_yn"}}
        out.append({"state": _wrap_state(text, rng), "questions": qs})
    return out


SOURCES = {"banking77": (_banking, "train", "test"), "boolq": (_boolq, "train", "validation"), "agnews": (_agnews, "train", "test"),
           "mnli": (_mnli, "train", "validation_matched"), "sst5": (_sst5, "train", "test"), "yelp": (_yelp, "train", "test")}


# Sources training refuses (d1a.train, d1a.suite.validate_training): the frozen suites hold records from these, kept for
# evaluation only. MMLU is a knowledge probe; Emotion/TweetEval are noisy-label honesty checks; QNLI/PAWS/SciQ measure
# reading transfer.
EVAL_ONLY = ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq")


def build(n_per_source, split="train", seed=0, exclude=(), only=(), revisions=None, sources=None, repos=None):
    sources = SOURCES if sources is None else sources
    repos = REPOS if repos is None else repos
    unknown = (set(exclude) | set(only)) - sources.keys()
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    reqs = []
    for name, (fn, tr, te) in sources.items():
        if name in exclude or (only and name not in only): continue
        src = Source(source_seed(seed, name), (revisions or {}).get(repos[name]))
        source_split = tr if split == "train" else te
        records = fn(source_split, n_per_source, src)
        if len(records) != len(src.origins):
            raise ValueError(f"provenance mismatch for {name}")
        for record, origin in zip(records, src.origins):
            record["_meta"] = {**origin, "source": name, "repo": repos[name], "revision": src.revision,
                               "split": source_split, "id": f"{name}/{source_split}/{origin['row']}"}
        reqs.extend(records)
    random.Random(seed).shuffle(reqs)
    return reqs


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order (always); sometimes add a 'none of the above' option, either as the correct
    answer (true option removed) or as a wrong alternative (true option kept); sometimes add an irrelevant distractor."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q; continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:                       # soft-target questions: permute only; inserting or swapping options would change the target's meaning
            keys = list(crit); rng.shuffle(keys); out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}; continue
        r = rng.random()
        none_options = [(k, v) for k, v in NONE_OPTIONS if k not in crit]
        distractors = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            nk, nd = rng.choice(none_options); crit.pop(y); crit[nk] = nd; y = nk
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            nk, nd = rng.choice(none_options); crit[nk] = nd
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            k = rng.choice(distractors); crit[k] = DISTRACTORS[k]
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}, "label": y}
    return out


def none_pair(req, rng):
    """Minimal pair for the none-of-the-above shortcut: the same state and question rendered twice, once with the true
    option present (a none option is wrong) and once with it removed (the same none option is right). Everything else,
    including option order, is identical, so the only difference the model can use is whether the evidence matches an
    option. Returns [] when the request has no eligible Choice (>=3 options, no none key already present, no soft target:
    removing the labelled option or adding a none option would change what the target means, as in augment)."""
    eligible = [(qid, q) for qid, q in req["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3 and q.get("target") is None]
    if not eligible: return []
    qid, q = rng.choice(eligible)
    nk, nd = rng.choice([o for o in NONE_OPTIONS if o[0] not in q["criteria"]] or [("none_of_these", None)])
    keys = list(q["criteria"]) + [nk]; rng.shuffle(keys)
    present = {**q, "criteria": {k: (nd if k == nk else q["criteria"][k]) for k in keys}}
    absent = {**present, "criteria": {k: v for k, v in present["criteria"].items() if k != q["label"]}, "label": nk}
    return [{"state": req["state"], "questions": {qid: present}}, {"state": req["state"], "questions": {qid: absent}}]


def load_records(path, source="custom"):
    """Labelled requests from a JSONL file, one per line, in the API's request shape plus a `label` on every question:

        {"state": "...", "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "other": null}, "label": "billing"},
                                       "urgent": {"type": "noul", "instructions": "...", "label": true},
                                       "priority": {"type": "score", "instructions": "...", "criteria": ["low", "medium", "high"], "label": 2}}}

    Labels: the option name for choice, true/false for noul, the level index (from 0) for score. `_meta` and per-question
    `src` are filled in so the records behave like a frozen suite's (source = `source`, id = line number)."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip(): continue
            r = json.loads(line)
            if "state" not in r or not isinstance(r.get("questions"), dict) or not r["questions"]:
                raise ValueError(f"{path}:{n + 1}: a record needs a state and a non-empty questions object")
            for qid, q in r["questions"].items():
                if "label" not in q: raise ValueError(f"{path}:{n + 1}: question {qid!r} has no label")
                q.setdefault("src", f"{source}_{q['type']}")
            text = json.dumps(r["state"], sort_keys=True, ensure_ascii=False) if not isinstance(r["state"], str) else r["state"]
            r["_meta"] = {**{"source": source, "variant": "clean", "id": f"{source}/{n}", "group_id": f"{source}/{n}", "row": n, "split": "custom",
                             "text_sha256": hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()}, **r.get("_meta", {})}
            records.append(r)
    if not records: raise ValueError(f"{path}: no records")
    return records


def api_request(record):
    """The /v1/systemone request body for a labelled record: state and typed questions only, never labels, targets or
    metadata (this is what leaves the machine when a remote predictor is scored)."""
    return {"state": record["state"], "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in record["questions"].items()}}


def materialize(req):
    """Labelled request -> internal record via the serving path (api.to_record), attaching int labels and src."""
    rec, meta = to_record(SystemOneRequest.model_validate(api_request(req)))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        y = src_q["label"]
        q["label"] = m["keys"].index(y) if m["type"] == "choice" else int(y)   # noul labels are bools, score labels level indices
        q["src"] = src_q["src"]; q["qtype"] = m["type"]; q["qid"] = qid; q["keys"] = m["keys"]
        if src_q.get("target") is not None:
            # soft target keyed by option name (choice), "false"/"true" (noul) or level index as a string (score); options the
            # target does not name get 0, then the vector is normalised. Used for unknowable records (uniform over the options).
            t = [float(src_q["target"].get(k, 0.0)) for k in q["keys"]]
            if sum(t) <= 0: raise ValueError(f"target for {qid} puts no mass on any option")
            q["target"] = [x / sum(t) for x in t]
    return rec
