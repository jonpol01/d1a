"""The plan-driven training-mix builder (#202; it built the D2 and D2-skills mixes): a JSON plan names, per training source,
how many records to draw and how (stratified, weighted, by label quotas, only records no earlier run trained on), from the
pinned suites only, after screening every eval partition. CPU only, no model.

    HF_HUB_OFFLINE=1 uv run python recipes/mix/build_mix.py recipes/mix/plans/d2-skills.json --out /data/train.jsonl \\
        --exposure <exposure_index.jsonl> --replay runs/labeler-replay --human <human-labeled.json>

The leak screens drop a training record whose normalised state is in any eval partition or eval-only kit, a near-duplicate
of one (the same 300-character prefix and word 5-shingle Jaccard >= 0.5; a PR with the same title and body Jaccard >= 0.9;
a word 5-shingle Jaccard >= 0.8 against its own family's eval partitions, exact over every eval state an inverted index
finds), a repeated state inside a source (conflicting labels drop every copy), a PR whose id is in a pr-labels eval
partition or the hand-labelled kit, and the Japanese twin of an English PR dropped as eval-like. The mix holds no exact
eval state and no eval PR id (checked again on the whole mix).

A plan covers every d1a.eval.suites.train_sources() entry and decision-v7 (in the mix, or "dv7_replay" > 0 for the trainer's
--replay), as recipes/skills/mix.py, recipes/pr-labeler/mix.py and d1a.training.train require (#167, #211), or names what it
leaves out and why: "allow_missing_sources": [...], "reason": "...".

Plan: {"seed": 0, "dv7_replay": 0, "sources": [{"source": "<suite>:<partition>", ...}]}, a source taking:
  "n"                 records to draw (without "quotas");
  "stratify"          a key ("q:<question>" its label, "_meta.<field>", "nq", "qset") whose values share n,
  "weights"           in proportion to these weights (default equal), or "natural" (the pool's own shares);
  "quotas"            [{"match": {key: value or "*"}, "n": k}, ...] filled greedily in order instead of n;
  "only_unseen"       only records no earlier run trained on (needs --exposure);
  "max_json_chars"    only records whose state and questions are at most this long as JSON;
  "order"             "shuffle" keeps the seeded shuffle; by default fresh records come first (needs --exposure to matter);
  "templated_cap"     at most this many records whose title occurs 50 or more times in the source.

--exposure is the per-record exposure index (one JSON line per record: source, row, state_sha256, "seen" {run: passes},
"phase_c" passes), private and outside git. Without it every record counts as never trained.

Writes <out> (each record is json.dumps(record, ensure_ascii=False) of its source row, byte for byte), <out>.index.jsonl
(source, row and state sha256 per line), <out>.json (the plan, records by source in d1a.training.train's sidecar shape
{"sources": {name: {"records": n}}}, label shares, reuse, the screens, the suites' manifest sha256 and the mix's sha256)
and <out>.smoke.jsonl (the 40 longest states). The same plan, seed, suites and exposure index give the same bytes.
recipes/mix/verify_mix.py checks a built mix.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from d1a.eval.suite import digest, load_split, normalise_text, read_jsonl  # noqa: E402
from d1a.eval.suites import manifest, resolve, train_sources  # noqa: E402

DECISION_V7 = "evals/v7/decision-v7"
# The eval partitions every mix is screened against, in this order (of two equally near eval states, the verifier names the
# first): every eval partition of the D1A suites (these first, then any other suite under evals/d1a), and these frozen
# suites' splits.
D1A_EVAL_SUITES = ("pr-labels", "routing", "ja-jglue", "night2", "external")
FROZEN_EVALS = (("evals/hard-v1", ("development", "test")), ("evals/devtools-v1", ("development", "test")),
                ("evals/documents-v1", ("development", "test")), ("evals/v7/decision-v7", ("development", "test", "calibration")),
                ("evals/v4/transfer-v4", ("development", "test")), ("evals/v4/decision-v4", ("development", "test", "calibration")),
                ("evals/v9/transfer-v9", ("development", "test")), ("evals/transfer-v2", ("development", "test")),
                ("evals/public-pool-v4", ("development", "test", "calibration")),
                ("evals/decision-v2", ("development", "test", "calibration")))
REPLAY_KIT, DEMO_KIT, HUMAN_KIT = ("EVAL-ONLY labeler-replay (distinct states)", "EVAL-ONLY playground demo requests",
                                   "EVAL-ONLY human-labeled owner PRs")
PR_ID_EVALS = ("pr-labels", HUMAN_KIT)   # the eval partitions and kits whose PR ids a training PR must not share
# word 5-shingle screen (Jaccard >= 0.8): a training source against the eval partitions of its own family
FAMILIES = {"pr-labels": ("pr-labels:", "EVAL-ONLY labeler", "EVAL-ONLY human"), "documents-v1": ("documents-v1:",),
            "devtools-v1": ("devtools-v1:",), "hard-v1": ("hard-v1:",), "routing": ("routing:", "EVAL-ONLY playground"),
            "ja-jglue": ("ja-jglue:",), "decision-v7": ("decision-v7:", "night2:", "decision-v4:", "public-pool-v4:", "transfer-v4:")}
NEAR_SHINGLE = 0.8
TEMPLATED = 50                         # a title this frequent in a source is a template
RECENT_STAGE = "v0.5 (skills stage)"   # the exposure index's latest finished run: its records come after fresher ones
TOK_PER_CHAR = 0.255                   # measured: Gemma tokens per JSON character of state and questions (v0.5 0.255, phase C 0.254)
MPS_TOK_S, A100_TOK_S, L4_TOK_S = 70.7, 1163, 542   # measured training tokens/s: M1 Max (phase C), HF A100 (v0.5), HF L4 (v0.4)
DV7_TOK = 230                          # decision-v7 train, tokens per record


# --- records ----------------------------------------------------------------------------------------------------------

def state_text(r):
    s = r["state"]
    return s if isinstance(s, str) else json.dumps(s, sort_keys=True, ensure_ascii=False)


def state_hash(r):
    """sha256 of the normalised state: the identity the exact screens compare."""
    return hashlib.sha256(normalise_text(state_text(r)).encode()).hexdigest()


def title(r):
    return " ".join(r["state"].split("\n", 1)[0].casefold().split()) if isinstance(r["state"], str) else ""


def line(r):
    return json.dumps(r, ensure_ascii=False) + "\n"


def word_shingles(t):
    """Hashed word 5-shingles of a text (the pool-level and the verifier's near-duplicate screens)."""
    w = re.findall(r"\w+", t.casefold())
    return {hashlib.blake2b(" ".join(w[i:i + 5]).encode(), digest_size=8).digest() for i in range(max(1, len(w) - 4))}


def nearest(sa, shared, E):
    """-> (Jaccard, j): the eval state E[j] (a shingle set) nearest to the shingle set `sa`, given `shared` {j: |sa & E[j]|}.
    Exact over every candidate, ties to the lowest j, so the answer does not depend on the order `shared` was filled in."""
    best, bj = 0, None
    for j in sorted(shared):
        n = shared[j]
        jac = n / max(1, len(sa) + len(E[j]) - n)   # |sa | E[j]| from the counts
        if jac > best: best, bj = jac, j
    return best, bj


def get(r, key):
    if key.startswith("q:"):
        q = r["questions"].get(key[2:])
        return q.get("label") if q else None
    if key.startswith("_meta."):
        return (r.get("_meta") or {}).get(key[6:])
    if key == "nq":
        return len(r["questions"])
    if key == "qset":
        return "+".join(sorted(r["questions"]))
    raise KeyError(key)


# --- sources and coverage ---------------------------------------------------------------------------------------------

def short(ref):
    """A source's name in a plan and a sidecar: evals/d1a/<suite>:<partition> -> <suite>:<partition>, evals/.../<suite> -> <suite>:train."""
    return ref.removeprefix("evals/d1a/") if ":" in ref else f"{ref.rpartition('/')[2]}:train"


def required(root=ROOT_DIR):
    """{plan name: source} for every source a fine-tune must train on or replay: train_sources() and decision-v7."""
    return {short(ref): ref for ref in [*train_sources(root), DECISION_V7]}


def uncovered(plan, counts=None, root=ROOT_DIR):
    """The required sources a plan (or, given its records by source, the mix it built) leaves out without naming them in
    "allow_missing_sources" with a "reason". decision-v7 is covered by the trainer when "dv7_replay" > 0."""
    need = required(root)
    names = {**{s: s for s in need}, **{ref: s for s, ref in need.items()}}
    allowed = plan.get("allow_missing_sources") or []
    if allowed and not str(plan.get("reason") or "").strip():
        raise ValueError("allow_missing_sources needs a reason: a fine-tune forgets what it neither trains on nor replays (#167)")
    if unknown := [s for s in allowed if s not in names]:
        raise ValueError(f"allow_missing_sources names no training source: {', '.join(unknown)} (they are: {', '.join(need)})")
    covered = {spec["source"] for spec in plan["sources"] if spec.get("quotas") or spec.get("n", 0) > 0}
    if counts is not None:
        covered = {s for s in covered if counts.get(s, 0) > 0}
    if plan.get("dv7_replay", 0) > 0:
        covered.add(short(DECISION_V7))
    return [s for s in need if s not in covered and s not in {names[a] for a in allowed}]


def check_plan(plan, root=ROOT_DIR):
    need = required(root)
    if unknown := [spec["source"] for spec in plan["sources"] if spec["source"] not in need]:
        raise ValueError(f"the plan names no training source: {', '.join(unknown)} (they are: {', '.join(need)})")
    if dup := [s for s, n in Counter(spec["source"] for spec in plan["sources"]).items() if n > 1]:
        raise ValueError(f"the plan names a source twice: {', '.join(dup)}")
    if missing := uncovered(plan, root=root):
        raise ValueError(f"the plan leaves out {', '.join(missing)}: a fine-tune forgets what it neither trains on nor replays "
                         "(#167, #211); add them, or name them in allow_missing_sources with a reason")


# --- loading ----------------------------------------------------------------------------------------------------------

def load_train(root=ROOT_DIR):
    """{plan name: records} of every required source, from its pinned partition (fetched and checked against the manifest)."""
    out = {}
    for name, ref in required(root).items():
        if ":" in ref:
            out[name] = read_jsonl(resolve(f"{Path(root) / ref.rpartition(':')[0]}:{ref.rpartition(':')[2]}", purpose="train"))
        else:
            out[name] = load_split(Path(root) / ref, "train")
    return out


def load_evals(root=ROOT_DIR, replay=None, human=None):
    """{name: records} of every eval partition (D1A_EVAL_SUITES, then any other D1A suite, then FROZEN_EVALS; test
    partitions only hashed, never scored) and the eval-only kits: the labeler replay's distinct states and the playground's
    demo requests (`replay`, a labeler replay directory), and the owner's hand-labelled PRs (`human`, a JSON list)."""
    root, evals = Path(root), {}
    others = sorted(m.parent.name for m in (root / "evals/d1a").glob("*/manifest.json") if m.parent.name not in D1A_EVAL_SUITES)
    for suite in (*D1A_EVAL_SUITES, *others):
        for name, part in manifest(root / "evals/d1a" / suite)["partitions"].items():
            if part["role"] == "eval":
                evals[f"{suite}:{name}"] = read_jsonl(resolve(f"{root / 'evals/d1a' / suite}:{name}"))
    for d, splits in FROZEN_EVALS:
        for s in splits:
            evals[f"{d.rpartition('/')[2]}:{s}"] = load_split(root / d, s, allow_test=True)
    if replay is not None:
        seen, states = set(), []
        for x in (Path(replay) / "labeler-calls.jsonl").read_text(encoding="utf-8").split("\n"):
            if x.strip() and "state" in (r := json.loads(x)) and r["state"] not in seen:
                seen.add(r["state"]); states.append({"state": r["state"], "questions": {}})
        evals[REPLAY_KIT] = states
    if human is not None:
        evals[HUMAN_KIT] = [{"state": r["state"], "questions": {}, **({"id": r["id"]} if r.get("id") else {})}
                            for r in json.loads(Path(human).read_text(encoding="utf-8"))]
    if replay is not None:
        evals[DEMO_KIT] = [{"state": r["body"]["state"], "questions": {}}
                           for r in json.loads((Path(replay) / "demo-requests.json").read_text(encoding="utf-8")) if "state" in r.get("body", {})]
    return evals


def load_exposure(path):
    expo = {}
    with open(path, encoding="utf-8") as f:
        for x in f:
            if x.strip():
                e = json.loads(x); expo[(e["source"], e["row"])] = e
    return expo


# --- the screens --------------------------------------------------------------------------------------------------------

def _ws_shingles(t, n=5):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def guards(train, evals):
    """-> {source: {row: reason}}: exact normalised state in any eval partition or kit ("exact-eval"), the same 300-character
    prefix and word 5-shingle Jaccard >= 0.5 ("near-eval"), a repeated state inside the source (conflicting labels: drop
    every copy, "dup-conflict"; agreeing: keep the first, "dup")."""
    exact, pref = set(), defaultdict(list)
    for rows in evals.values():
        for r in rows:
            t = normalise_text(state_text(r)).casefold()
            exact.add(state_hash(r))
            pref[" ".join(t.split())[:300]].append(t)
    drop = defaultdict(dict)
    for src, rows in train.items():
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[state_hash(r)].append(i)
            if state_hash(r) in exact:
                drop[src][i] = "exact-eval"; continue
            t = normalise_text(state_text(r)).casefold()
            if cands := pref.get(" ".join(t.split())[:300]):
                a = _ws_shingles(t)
                if any(len(a & (b := _ws_shingles(e))) / max(1, len(a | b)) >= 0.5 for e in cands):
                    drop[src][i] = "near-eval"
        for idx in groups.values():
            if len(idx) > 1:
                labs = {json.dumps({k: v.get("label") for k, v in rows[i]["questions"].items()}, sort_keys=True) for i in idx}
                for i in (idx if len(labs) > 1 else idx[1:]):
                    drop[src].setdefault(i, "dup-conflict" if len(labs) > 1 else "dup")
    return drop


def screen(train, evals):
    """guards(), then the PR title-body screen, the family shingle screen and the Japanese twins -> (drop, report)."""
    drop = guards(train, evals)
    pid = lambda r: str(r.get("id")).split(":")[0]  # noqa: E731   (a Japanese twin's id is its PR's, with a suffix)
    eval_prs = {pid(r) for k, rs in evals.items() if k.startswith(PR_ID_EVALS) for r in rs if r.get("id")}
    for src in [k for k in train if k.startswith("pr-labels")]:
        for i, r in enumerate(train[src]):
            if r.get("id") and pid(r) in eval_prs:
                drop[src].setdefault(i, "eval-pr-id")   # the same PR in train and in an eval partition, whatever its text
    norm = lambda t: " ".join(t.casefold().split())  # noqa: E731
    body = lambda t: (lambda w: set(zip(w, w[1:], w[2:], w[3:], w[4:])))(re.findall(r"\w+", t.split("body:", 1)[-1].casefold()))  # noqa: E731
    etitles = defaultdict(list)
    for k, rs in evals.items():
        if k.startswith(("pr-labels", "EVAL-ONLY labeler", "EVAL-ONLY human")):
            for r in rs:
                t = state_text(r); etitles[norm(t.split("\n", 1)[0])].append(t)
    for src in [k for k in train if k.startswith("pr-labels")]:
        for i, r in enumerate(train[src]):
            if i in drop[src]: continue
            t = state_text(r); bb = None
            for et in etitles.get(norm(t.split("\n", 1)[0]), ()):
                bb = bb or body(t); eb = body(et)
                if len(bb & eb) / max(1, len(bb | eb)) >= 0.9:
                    drop[src][i] = "near-eval-title-body"; break
    dropped, near_mid = Counter(), {}
    for fam, prefixes in FAMILIES.items():
        E = [word_shingles(state_text(r)) for k, rs in evals.items() if k.startswith(prefixes) for r in rs]
        ix = defaultdict(list)
        for j, es in enumerate(E):
            for x in es: ix[x].append(j)
        for src in [k for k in train if k.split(":")[0] == fam]:
            mid = 0
            for i, r in enumerate(train[src]):
                if i in drop[src]: continue
                sa = word_shingles(state_text(r)); c = Counter()
                for x in sa:
                    for j in ix.get(x, ()): c[j] += 1
                best, _ = nearest(sa, c, E)
                if best >= NEAR_SHINGLE:
                    drop[src][i] = "near-eval-shingle"; dropped[src] += 1
                elif best >= 0.5:
                    mid += 1
            near_mid[src] = mid
    # pr-labels:train-ja translates train PRs: a translation passes the within-language screens, so the twin of every English
    # PR dropped as eval-like goes too
    if "pr-labels:train" in train and "pr-labels:train-ja" in train:
        eng_bad = {pid(train["pr-labels:train"][i]) for i, why in drop["pr-labels:train"].items() if "eval" in why}
        for i, r in enumerate(train["pr-labels:train-ja"]):
            if pid(r) in eng_bad and i not in drop["pr-labels:train-ja"]:
                drop["pr-labels:train-ja"][i] = "ja-twin-of-eval-like"
    return drop, {"dropped_shingle_ge_0.8": dict(dropped), "pool_records_0.5_to_0.8_kept": near_mid}


# --- drawing ------------------------------------------------------------------------------------------------------------

def quota_pick(rows, order, quotas):
    """Greedy multi-question quota fill. quotas: [{"match": {key: value}, "n": k}] in priority order. A record counts toward
    every quota it matches; a pass only takes records that match it, preferring ones that overflow no single-key quota, then
    ones that help other unfilled quotas, then the preference order."""
    chosen, cnt = [], Counter()
    simple = [(i, q) for i, q in enumerate(quotas) if len(q["match"]) == 1]
    def matches(r, q): return all((get(r, k) is not None) if v == "*" else str(get(r, k)) == str(v) for k, v in q["match"].items())
    taken = set()
    rank = {i: k for k, i in enumerate(order)}
    for qi, q in enumerate(quotas):
        if cnt[qi] >= q["n"]: continue
        cands = [i for i in order if i not in taken and matches(rows[i], q)]
        def score(i):
            r = rows[i]
            over = sum(1 for j, s in simple if j != qi and matches(r, s) and cnt[j] >= s["n"])
            helps = sum(1 for j, s in enumerate(quotas) if j != qi and matches(r, s) and cnt[j] < s["n"])
            return (over, -helps)
        cands.sort(key=lambda i: (score(i), rank[i]))
        for i in cands:
            if cnt[qi] >= q["n"]: break
            taken.add(i); chosen.append(i)
            for j, s in enumerate(quotas):
                if matches(rows[i], s): cnt[j] += 1
    return chosen, {json.dumps(q["match"], sort_keys=True): (cnt[i], q["n"]) for i, q in enumerate(quotas)}


def strat_pick(rows, order, n, key, weights=None):
    """n records spread over the values of `key` in proportion to weights (default equal; "natural": the pool's shares),
    in preference order; a short group or rounding is topped up in preference order."""
    groups = defaultdict(list)
    for i in order: groups[str(get(rows[i], key))].append(i)
    w = {g: len(v) for g, v in groups.items()} if weights == "natural" else {g: (weights or {}).get(g, 1.0) for g in groups}
    tot = sum(w.values()); want = {g: int(round(n * w[g] / tot)) for g in groups}
    out = []
    for g in sorted(groups):
        out += groups[g][:want[g]]
    if len(out) < n:
        have = set(out); out += [i for i in order if i not in have][:n - len(out)]
    return out[:n], {g: sum(1 for i in out[:n] if str(get(rows[i], key)) == g) for g in groups}


def json_chars(r):
    return len(json.dumps({"s": r["state"], "q": r["questions"]}, ensure_ascii=False))


def build(plan, train, evals, exposure=None, root=ROOT_DIR):
    """-> (records, index, report): the mix a plan draws from `train` ({plan name: records}) after screening `evals`
    ({name: records}). `exposure` {(source, row): entry}: which earlier runs trained on each record (None: none did)."""
    check_plan(plan, root)
    if exposure is None and any(spec.get("only_unseen") for spec in plan["sources"]):
        raise ValueError("only_unseen needs the exposure index (--exposure): without it every record looks unseen")
    seed = plan.get("seed", 0)
    drop, pool_screen = screen(train, evals)
    out_rows, index, report = [], [], {"sources": {}}
    rng = random.Random(seed)
    for spec in plan["sources"]:
        src = spec["source"]; rows = train[src]

        def e(i, src=src, rows=rows):
            if exposure is None:
                return {"state_sha256": state_hash(rows[i]), "seen": {}, "phase_c": 0}
            x = exposure.get((src, i))
            if x is None or x["state_sha256"] != state_hash(rows[i]):
                raise ValueError(f"the exposure index does not match {src} row {i}: rebuild it for these suites")
            return x
        pool = [i for i in range(len(rows)) if i not in drop[src]]
        if spec.get("max_json_chars"):
            pool = [i for i in pool if json_chars(rows[i]) <= spec["max_json_chars"]]
        if spec.get("only_unseen"):
            pool = [i for i in pool if not e(i)["seen"] and not e(i)["phase_c"]]
        rng.shuffle(pool)
        # fresh first: not in the latest runs, fewest past passes; ties keep the seeded shuffle
        order = sorted(pool, key=lambda i: (e(i)["phase_c"] > 0, RECENT_STAGE in e(i)["seen"], sum(e(i)["seen"].values())))
        if spec.get("order") == "shuffle":   # keeps every unstratified label at its pool share (fresh-first skews them)
            order = list(pool)
        if "templated_cap" in spec:
            templated = {t for t, c in Counter(title(r) for r in rows).items() if c >= TEMPLATED}
            t = [i for i in order if title(rows[i]) in templated]
            cut = set(t[spec["templated_cap"]:]); order = [i for i in order if i not in cut]
        if "quotas" in spec:
            chosen, fill = quota_pick(rows, order, spec["quotas"])
        elif "stratify" in spec:
            chosen, fill = strat_pick(rows, order, spec["n"], spec["stratify"], spec.get("weights"))
        else:
            chosen, fill = order[:spec["n"]], None
        lab, reuse, chars = defaultdict(Counter), Counter(), []
        for i in chosen:
            r = rows[i]
            out_rows.append(r); index.append({"source": src, "row": i, "state_sha256": e(i)["state_sha256"]})
            for qid, q in r["questions"].items():
                if src.startswith(("pr-labels", "routing", "documents")): lab[qid][str(q.get("label"))] += 1
            for k in ("_meta.family", "_meta.source"):
                if get(r, k) and not src.startswith("pr-labels"): lab[k][str(get(r, k))] += 1
            seen = e(i)["seen"]
            for m in seen: reuse[m] += 1
            if e(i)["phase_c"]: reuse["phase C (running)"] += 1
            if not seen and not e(i)["phase_c"]: reuse["never trained"] += 1
            chars.append(json_chars(r))
        tok = TOK_PER_CHAR * sum(chars)
        report["sources"][src] = {"records": len(chosen), "pool_after_guards": len(pool), "fill": fill,
                                  "labels": {k: dict(v.most_common()) for k, v in lab.items()}, "reuse": dict(reuse.most_common()),
                                  "est_tokens": round(tok), "est_tok_per_rec": round(tok / max(1, len(chosen))),
                                  "est_mps_hours": round(tok / MPS_TOK_S / 3600, 2)}
    counts = {s: v["records"] for s, v in report["sources"].items()}
    if missing := uncovered(plan, counts, root):
        raise ValueError(f"the mix draws no records from {', '.join(missing)} (their pools are empty after the screens)")
    perm = list(range(len(out_rows))); random.Random(f"{seed}:mix").shuffle(perm)
    out_rows = [out_rows[k] for k in perm]; index = [index[k] for k in perm]
    text = "".join(line(r) for r in out_rows)
    # whole-mix checks: no eval state, no repeated state, no eval PR id
    eval_h = {state_hash(r) for rows in evals.values() for r in rows}
    mix_h = Counter(state_hash(r) for r in out_rows)
    eval_ids = {str(r.get("id")) for k, rows in evals.items() if k.startswith(PR_ID_EVALS) for r in rows}
    sev, blast, typ = Counter(), Counter(), Counter()
    for r in out_rows:
        q = r["questions"]
        if "sev" in q and str(q["sev"].get("label", "")).startswith("P"): sev[q["sev"]["label"]] += 1
        if "blast" in q: blast[q["blast"]["label"]] += 1
        if "type" in q and str(q["type"].get("label", "")).startswith("type/"): typ[q["type"]["label"]] += 1
    dv7 = plan.get("dv7_replay", 0)
    tok_all = sum(s["est_tokens"] for s in report["sources"].values()) + dv7 * DV7_TOK
    n_all = len(out_rows) + dv7
    share = lambda c: {k: f"{v} ({100 * v / sum(c.values()):.1f}%)" for k, v in sorted(c.items())}  # noqa: E731
    report["pool_screen"] = pool_screen
    report.update({"plan": plan, "mix_records": len(out_rows), "dv7_replay_by_trainer": dv7, "records_trained": n_all,
                   "steps_of_8": -(-n_all // 8),
                   "checks": {"eval_state_overlap": sum(1 for x in mix_h if x in eval_h), "repeated_states": sum(v - 1 for v in mix_h.values() if v > 1),
                              "pr_id_overlap_with_eval": sum(1 for r in out_rows if r.get("id") and str(r.get("id")) in eval_ids),
                              "eval_partitions_screened": len(evals)},
                   "eval_partitions": list(evals),
                   "guards_dropped": {k: dict(Counter(v.values())) for k, v in drop.items() if v},
                   "pr_sev": share(sev), "pr_blast": share(blast), "pr_type": share(typ),
                   "estimate": {"tokens": round(tok_all), "tok_per_rec": round(tok_all / max(1, n_all)),
                                "mps_hours": round(tok_all / MPS_TOK_S / 3600, 1), "a100_hours": round(tok_all / A100_TOK_S / 3600, 2),
                                "l4_hours": round(tok_all / L4_TOK_S / 3600, 2)},
                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    if report["checks"]["eval_state_overlap"] or report["checks"]["repeated_states"] or report["checks"]["pr_id_overlap_with_eval"]:
        raise ValueError(f"the mix failed its whole-mix checks: {report['checks']}")
    return out_rows, index, report


def write(out, rows, index, report, smoke_records=40):
    out = str(out)
    Path(out).write_text("".join(line(r) for r in rows), encoding="utf-8")
    Path(out + ".index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index), encoding="utf-8")
    smoke = sorted(rows, key=lambda r: len(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])))[-smoke_records:]
    Path(out + ".smoke.jsonl").write_text("".join(line(r) for r in smoke), encoding="utf-8")
    Path(out + ".json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", help="the plan (JSON), e.g. recipes/mix/plans/d2-skills.json")
    ap.add_argument("--out", required=True, help="the mix (JSONL); its sidecars go beside it")
    ap.add_argument("--root", default=str(ROOT_DIR), help="the checkout whose evals/ holds the pinned suites")
    ap.add_argument("--exposure", help="the per-record exposure index (JSONL); without it every record counts as never trained")
    ap.add_argument("--replay", help="the labeler replay kit (labeler-calls.jsonl, demo-requests.json): screened as eval-only")
    ap.add_argument("--human", help="the owner's hand-labelled PRs (JSON list of {state, id, ...}): screened as eval-only, by state and id")
    ap.add_argument("--no-kits", action="store_true", help="build without the eval-only kits (the sidecar lists what was screened)")
    a = ap.parse_args()
    if not a.no_kits and not (a.replay and a.human):
        ap.error("pass --replay and --human so the mix is screened against the eval-only kits too, or --no-kits")
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    try:
        check_plan(plan, a.root)
    except ValueError as err:
        ap.error(str(err))
    train = load_train(a.root)
    evals = load_evals(a.root, a.replay, a.human)
    try:
        rows, index, report = build(plan, train, evals, load_exposure(a.exposure) if a.exposure else None, a.root)
    except ValueError as err:
        raise SystemExit(f"build_mix: {err}")
    used = sorted({d for d, _ in FROZEN_EVALS} | {ref.rpartition(":")[0] if ":" in ref else ref for ref in required(a.root).values()}
                  | {f"evals/d1a/{m.parent.name}" for m in (Path(a.root) / "evals/d1a").glob("*/manifest.json")})
    report["manifests"] = {d: digest(Path(a.root) / d / "manifest.json") for d in used}
    report["exposure_sha256"] = digest(a.exposure) if a.exposure else None
    write(a.out, rows, index, report)
    print(json.dumps({k: report[k] for k in ("mix_records", "records_trained", "steps_of_8", "checks", "pr_sev", "pr_blast", "pr_type", "estimate", "sha256")}, indent=1))
    for s, v in report["sources"].items():
        print(f"{s:24s} {v['records']:5d} of {v['pool_after_guards']:5d}  ~{v['est_tok_per_rec']} tok/rec  {v['est_mps_hours']} h  reuse {v['reuse']}")


if __name__ == "__main__":
    main()
