"""One scorecard for a new checkpoint against the one it would replace (John, 2026-10-07, #202): every suite, the demos and
the labeler replay judged together. One use case's drop is a follow-up, not a veto. CPU only: reads the files
scripts/quality_gate.py writes (--out: suites/<suite>/{base,head}/rows.json, report.json), the live labeler check's dump
and the replay kit's labels; loads no model; prints aggregates only (no PR text).

    python scripts/decide.py --gate runs/quality-gate --labeler <labeler dump> --base-name v0.5 --head-name C \\
        --human <human-labeled.json> [--replay runs/labeler-replay] [--extra-gate <report-only gate>] [--out scorecard.json]

The gate is `quality_gate.py --run <served> --head-run <new> --all-suites --playground <d1a-playground>`. The labeler dump
holds, per side name, the answers to the replay kit's items: {"<name>": {"replay": [...], "human": [...]}}, item i the
kit's item i, each {question: [choice, label, p]}. --replay is the replay kit (questions.json, memory-items.jsonl: 98
decisions with review-bot labels) and --human the owner's labels (39 PRs; labels other than "unsure" count). Both are
private and never committed.

Lines: one per suite of the gate, and the live labeler's two parts (live:replay, live:human), each over every labelled
question. The pooled mean weighs each line equally; the night2 monitors and report-only suites are left out, and so is a
main-gate suite outside the --all-suites partitions (scored as "unexpected:<name>", report-only). A win or a loss is a
line whose 95% CI is above or below 0; the pr-labels_test:sev+offsets line (log-loss severity offsets fitted on
pr-labels_development, scored on pr-labels_test) counts too. The pooled CI resamples rows within each line; each suite
line's own CI resamples records (d1a.eval.metrics), so the pooled bound is slightly narrow when a record's questions move
together.

Verdict (the first that applies):
  INCOMPLETE  an input is missing or partial: any of the --all-suites partitions (EXPECTED) absent from the gate, without
              rows that pair on both sides, or that cannot be scored (rows.json corrupt, labels or option order differing
              between the sides, probabilities not finite); no report.json; a demo group (DEMOS) or the replay short of its
              requests or of its answered questions (a count that is not an int is short); any request a side did not answer
              in full (quality_gate's failures "answered by", "different questions answered", "different options"); no finite
              text latency ratio; no labeler dump or kit, or a dump whose base or head side lacks an item or a labelled
              question of the kit. Never a pass.
  VETO        V1: a card suite with delta < -2 or its CI lower bound < -4; V3: latency ratio above the floor's + 0.015;
              V4: a suite of n >= 150 worse by 5 points or more with its CI upper bound below 0. V2 (a person reads the
              flips of the safety demos, READ_FLIPS) is always listed and must be done before any deploy.
  BETTER      no veto, pooled delta >= 0, and either the pooled 95% CI lower bound > 0, or >= 3 significant wins on
              distinct sources (SOURCE: a -ja twin and sev+offsets count as their original's source) and no significant
              loss.
  NOT BETTER  otherwise; "near miss" when the pooled delta > 0 with >= 2 wins and no loss (not served or published: one
              same-recipe rerun with a new seed).
Every significant loss (a line, a night2 monitor, a report-only suite, or a per-question line by the sign test) and every
missed target becomes a follow-up issue; none of them is a veto by itself. The report-only gate never changes the verdict,
even when one of its suites is absent, corrupt or unpaired (listed under "report-only missing").
--expect-suites narrows EXPECTED for a dry run on a gate made without --all-suites; the verdict then says DRY RUN.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from d1a.eval.metrics import metrics, paired_bootstrap, scored_rows   # noqa: E402
from quality_gate import CARD_SUITES, suite_list                     # noqa: E402

# group: (requests, questions) the playground at 4baeee9 sends, as quality_gate.demo_requests lists them: 13 groups, 145
# requests, 201 questions. A playground change that adds or drops a demo example changes these.
DEMOS = {"demo:routing": (6, 6), "demo:guardrails": (8, 16), "demo:tool-gate": (8, 8), "demo:evals": (8, 16), "demo:gate": (4, 4),
         "demo:inbox": (16, 32), "demo:rerank": (12, 12), "demo:bulk": (60, 60), "demo:pr-labeler": (5, 15), "demo:control": (4, 4),
         "demo:photo": (6, 12), "demo:voice": (4, 8), "demo:video": (4, 8)}
REPLAY = (92, 276)                              # runs/labeler-replay/labeler-calls.jsonl: 92 distinct states, 3 questions each
# quality_gate.compare counts a request before it looks at the answers, so a request the head never answered (a crash, an
# OOM, HTTP 500, a timeout) still counts; these failures are the only trace of it
UNANSWERED = ("answered by", "different questions answered", "different options")
READ_FLIPS = ("demo:guardrails", "demo:tool-gate", "demo:gate", "demo:control", "demo:photo", "demo:voice", "demo:video", "demo:evals")
MONITOR = ("night2_dates", "night2_unknowable", "night2_assertion")   # decision-v7 train holds night2 states: forgetting monitors
FIT_POOL = {"pr-labels_development"}            # in the calibration fit and the severity-offsets fit: no ECE delta


def gate_name(suite):
    """A suite's directory under the gate's suites/ (as quality_gate.py names it)."""
    return suite.replace("/", "_").replace(":", "_")


CARD = tuple(gate_name(s) for s in CARD_SUITES)                       # V1 floor and the mean card-suite target (#58, #167)
# quality_gate --all-suites' partitions, as the gate's suite directories. Taken from the suite list, not from the gate, so a
# gate that stopped early or ran without --all-suites is INCOMPLETE.
EXPECTED = tuple(gate_name(s) for s in suite_list("", False, True))
SAMPLES = 2000
SIGNIFICANT = 0.05                              # per-question lines: two-sided sign test on fixes and breaks
# BETTER without a pooled lower bound > 0 needs >= 3 significant wins on distinct sources (the advisor's answer on #187,
# 2026-10-08): a -ja twin and the sev+offsets line count as their original's source, so one quirk cannot win twice. Null
# rate on the dry-run harness: 3/200 (1.5%), against 9/200 for ">= 2 wins".
SOURCE = {"pr-labels_development-ja": "pr-labels_development", "pr-labels_test-ja": "pr-labels_test",
          "pr-labels_test:sev+offsets": "pr-labels_test"}
BONUS = [round(0.1 * k, 1) for k in range(21)]  # the P1 bonuses the severity-offset rule sweeps


# --- rows ---------------------------------------------------------------------------------------------------------------

def load(d):
    """The scored rows, with a repeated (id, question) renamed by occurrence (devtools-v1 development holds one record id
    twice, which d1a.eval.metrics.paired_bootstrap refuses); both sides are scored in the same order, so the names pair."""
    rows, seen = scored_rows(json.loads((Path(d) / "rows.json").read_text(encoding="utf-8"))), Counter()
    for r in rows:
        k = (r["id"], r["question"]); seen[k] += 1
        if seen[k] > 1:
            r["id"] = f"{r['id']}#dup{seen[k]}"
    return rows


def paired(head, base, metric="acc"):
    r = paired_bootstrap(head, base, samples=SAMPLES, seed=0, metric=metric, aggregation="micro")
    k = f"micro_{metric}_delta"
    scale = 100 if metric in ("acc", "coverage_at_0_9") else 1
    return {"delta": r[k] * scale, "lo": r["ci95"][0] * scale, "hi": r["ci95"][1] * scale}


def top(r):
    return max(range(len(r["p"])), key=r["p"].__getitem__)


def flips(head, base):
    """(fixes, breaks) of head against base, by (id, question)."""
    b = {(r["id"], r["question"]): r for r in base}
    fx = br = 0
    for r in head:
        o = b[(r["id"], r["question"])]
        h_ok, b_ok = top(r) == r["label"], top(o) == o["label"]
        fx += h_ok and not b_ok; br += b_ok and not h_ok
    return fx, br


def per_class(rows):
    keys = rows[0]["keys"]
    lab, pred, hit = Counter(), Counter(), Counter()
    for r in rows:
        p = top(r)
        lab[r["label"]] += 1; pred[p] += 1; hit[r["label"]] += p == r["label"]
    return {keys[i].split("/")[-1].split(":")[-1]: {"recall": hit[i], "of": lab[i], "predicted": pred[i]} for i in range(len(keys))}


def sign_test(fixes, breaks):
    """Two-sided sign test p-value on fixes against breaks."""
    n, k = fixes + breaks, min(fixes, breaks)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def correct(rows):
    return {(r["id"], r["question"]): int(top(r) == r["label"]) for r in rows}


# --- severity offsets: 5 log-loss offsets on the raw logits, plus a P1 bonus chosen by a fixed rule ------------------------

def sev_rows(d):
    """The sev question's rows of <d>/rows.json: (ids, keys, raw logits, labels)."""
    rows = [r for r in json.loads((Path(d) / "rows.json").read_text(encoding="utf-8")) if r.get("question") == "sev"]
    keys = rows[0]["keys"]
    if not all(r["keys"] == keys for r in rows):
        raise ValueError("option order differs between rows")
    return [r["id"] for r in rows], keys, np.array([r["logits"] for r in rows], float), np.array([r["label"] for r in rows])


def fit_offsets(z, y, steps=3000, lr=0.5):
    """Per-option offsets added to the logits, fitted by log loss (gradient descent, centred)."""
    b = np.zeros(z.shape[1])
    onehot = np.eye(z.shape[1])[y]
    for _ in range(steps):
        p = np.exp(z + b - (z + b).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        b -= lr * (p - onehot).mean(0)
        b -= b.mean()
    return b


def offset_stats(pred, y, p1):
    tp = int(((pred == p1) & (y == p1)).sum()); pp = int((pred == p1).sum()); P = int((y == p1).sum())
    rec = tp / max(1, P); prec = tp / max(1, pp)
    return {"acc": float((pred == y).mean()), "tp": tp, "pp": pp, "P": P, "rec": rec, "prec": prec,
            "f1": 0 if tp == 0 else 2 * rec * prec / (rec + prec)}


def choose_offsets(z, y, p1):
    """The rule: fit the offsets; sweep a P1 bonus 0.0..2.0; keep the bonuses whose P1 recall >= 0.25 and precision >= 0.45 on
    these rows and take the most accurate (ties: the smaller bonus); if none qualifies, the bonus with the highest P1 F1.
    Returns (bonus, offsets, stats)."""
    b = fit_offsets(z, y); cands = []
    for bo in BONUS:
        bb = b.copy(); bb[p1] += bo
        cands.append((bo, bb, offset_stats((z + bb).argmax(1), y, p1)))
    ok = [c for c in cands if c[2]["rec"] >= 0.25 and c[2]["prec"] >= 0.45]
    if ok:
        return max(ok, key=lambda c: (c[2]["acc"], -c[0]))
    return max(cands, key=lambda c: (c[2]["f1"], -c[0]))


# --- lines --------------------------------------------------------------------------------------------------------------

def pooled(diffs, samples=SAMPLES, seed=0):
    """Equal weight per line: mean of the per-line accuracy deltas, 95% CI by resampling rows within each line."""
    rng = np.random.default_rng(seed)
    point = float(100 * np.mean([d.mean() for d in diffs]))
    boot = np.zeros(samples)
    for d in diffs:
        boot += d[rng.integers(0, len(d), size=(samples, len(d)))].mean(1)
    boot = 100 * boot / len(diffs)
    return {"delta": point, "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)), "lines": len(diffs)}


def item_bootstrap(d, item, samples=SAMPLES, seed=0):
    """Mean of d (x100) with a 95% CI from resampling items, so an item's questions stay together (as quality_gate's suites keep
    a record's sibling questions together)."""
    _, inv = np.unique(item, return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(s), size=(samples, len(s)))
    boot = 100 * s[pick].sum(1) / c[pick].sum(1)
    return {"delta": 100 * float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}


def suites_of(gate, card, diffs, tag="", expected=()):
    """Every suite of a gate. A problem is `missing` for the main gate and `report_only_missing` for the report-only one, so a
    broken report-only suite never changes the verdict. A suite that cannot be scored (rows.json unreadable or corrupt, labels
    or option order that differ between the sides, probabilities that are not finite) is a problem, not a crash. A main-gate
    suite outside `expected` is scored as report-only ("unexpected:"), so it is never a line."""
    present = sorted(d for d in (gate / "suites").iterdir() if d.is_dir() and not d.name.startswith(".")) if (gate / "suites").exists() else []
    card["report_only_missing" if tag else "missing"].extend(f"{tag}{n}: not in the gate" for n in expected if n not in {d.name for d in present})
    for d in present:
        ro = bool(tag) or d.name not in expected
        name = f"{tag or ('unexpected:' if ro else '')}{d.name}"
        missing = card["report_only_missing" if ro else "missing"]
        if not ((d / "base/rows.json").exists() and (d / "head/rows.json").exists()):
            missing.append(f"{name}: no rows on both sides"); continue
        try:
            base, head = load(d / "base"), load(d / "head")
            for side, got in (("base", base), ("head", head)):   # softmax output is finite and in [0, 1]; anything else is a broken read
                if bad := sum(1 for r in got if not all(math.isfinite(x) and 0 <= x <= 1 for x in r["p"])):
                    raise ValueError(f"{side}: {bad} rows with probabilities not finite or outside [0, 1]")
            cb, ch = correct(base), correct(head)
            if not cb or cb.keys() != ch.keys():
                missing.append(f"{name}: base {len(cb)} / head {len(ch)} questions do not pair"); continue
            keys = list(cb)
            acc = paired(head, base); mb, mh = metrics(base), metrics(head)
            s = {"n": len(keys), "base": 100 * mb["acc"], "head": 100 * mh["acc"], **acc, "fixes_breaks": flips(head, base),
                 "ece": [mb["ece"], mh["ece"]], "cov90": [mb["coverage_at_0_9"], mh["coverage_at_0_9"]], "report_only": ro}
            if d.name not in FIT_POOL:
                s["ece_delta"] = paired(head, base, "ece")
        except Exception as e:   # noqa: BLE001  (any failure to score a suite leaves it unjudged; the message says why)
            missing.append(f"{name}: cannot be scored ({type(e).__name__}: {str(e)[:120]})"); continue
        card["suites"][name] = s
        if d.name not in MONITOR and not ro:
            diffs[d.name] = np.array([ch[k] - cb[k] for k in keys], float)


def pr_lines(gate, card):
    """Per question on every PR partition; severity slices; severity offsets (fitted on development, scored on test)."""
    rows = {}
    for name in ("pr-labels_test", "pr-labels_development", "pr-labels_test-ja", "pr-labels_development-ja", "pr-labels_real17"):
        d = gate / "suites" / name
        if name not in card["suites"]:     # absent or unpaired: already listed as missing
            continue
        rows[name] = {s: load(d / s) for s in ("base", "head")}
        for q in ("type", "blast", "sev"):
            b = [r for r in rows[name]["base"] if r["question"] == q]; h = [r for r in rows[name]["head"] if r["question"] == q]
            if not b:
                continue
            card["pr"][f"{name}:{q}"] = {"n": len(b), "base": 100 * metrics(b)["acc"], "head": 100 * metrics(h)["acc"], **paired(h, b),
                                         "fixes_breaks": flips(h, b), "classes_base": per_class(b), "classes_head": per_class(h)}
    t = rows.get("pr-labels_test")
    if t:   # slice: severity on security PRs (labels joined by PR id)
        for s in ("base", "head"):
            typ = {r["id"]: r["keys"][r["label"]] for r in t[s] if r["question"] == "type"}
            sec = [r for r in t[s] if r["question"] == "sev" and typ.get(r["id"]) == "type/security"]
            if sec:
                card["slices"].setdefault("sev_on_security_prs", {})[s] = f"{100 * metrics(sec)['acc']:.1f} (n={len(sec)})"
    # offsets by the rule in choose_offsets, fitted per side on development rows of THIS gate, applied to test
    if "pr-labels_test" in card["suites"]:
        out = {}
        for s in ("base", "head"):
            ids, keys, z, y = sev_rows(gate / "suites/pr-labels_test" / s); p1 = keys.index("P1")
            dev = gate / "suites/pr-labels_development" / s
            if "pr-labels_development" in card["suites"]:
                _, kf, zf, yf = sev_rows(dev)
                if kf != keys:
                    raise ValueError("pr-labels_development and pr-labels_test order the sev options differently")
                bonus, b, _ = choose_offsets(zf, yf, p1); pred = (z + b).argmax(1); how = "fit on development"
            else:   # dry run without development rows: nested 5-fold CV by PR on test (an estimate, labelled as such)
                u = {q: k % 5 for k, q in enumerate(sorted({i.split(":")[0] for i in ids}))}; fold = np.array([u[i.split(":")[0]] for i in ids]); pred = np.zeros(len(y), int); bonus = []
                for f in range(5):
                    bo, b, _ = choose_offsets(z[fold != f], y[fold != f], p1); bonus.append(bo); pred[fold == f] = (z[fold == f] + b).argmax(1)
                how = "ESTIMATE: nested 5-fold CV on test (no development rows)"
            st = offset_stats(pred, y, p1); out[s] = {"ids": ids, "ok": (pred == y).astype(float), "stats": st, "bonus": bonus, "how": how}
        if out["base"]["ids"] != out["head"]["ids"]:
            raise ValueError("pr-labels_test's sev rows are not in the same order on both sides")
        d = out["head"]["ok"] - out["base"]["ok"]; rng = np.random.default_rng(0)
        bs = 100 * d[rng.integers(0, len(d), size=(SAMPLES, len(d)))].mean(1)
        card["pr"]["pr-labels_test:sev+offsets"] = {"n": len(d), "base": 100 * out["base"]["ok"].mean(), "head": 100 * out["head"]["ok"].mean(),
                                                    "delta": 100 * d.mean(), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                                    "P1": {s: f"{out[s]['stats']['tp']}/{out[s]['stats']['P']} prec {out[s]['stats']['prec']:.2f}" for s in out},
                                                    "bonus": {s: out[s]["bonus"] for s in out}, "how": out["head"]["how"]}


def slices_of(gate, card):
    """documents-v1 issue accuracy by product (joined by record id) and routing factory per question (has_target, decision)."""
    for s in ("base", "head"):   # only suites that scored (a corrupt one is already listed as missing)
        d = gate / "suites/documents-v1" / s
        if "documents-v1" in card["suites"]:
            rows = load(d); prod = {r["id"]: r["keys"][r["label"]] for r in rows if r["question"] == "product"}
            by = {}
            for r in rows:
                if r["question"] == "issue":
                    by.setdefault(prod.get(r["id"], "none"), []).append(r)
            card["slices"].setdefault("documents_issue_by_product", {})[s] = {k: f"{round(metrics(v)['acc'] * len(v))}/{len(v)}" for k, v in by.items()}
        f = gate / "suites/routing_factory-development" / s
        if "routing_factory-development" in card["suites"]:
            rows = load(f)
            for q in ("has_target", "decision"):
                rq = [r for r in rows if r["question"] == q]
                if rq:
                    card["slices"].setdefault(f"factory_{q}", {})[s] = {"acc": f"{round(metrics(rq)['acc'] * len(rq))}/{len(rq)}", "classes": per_class(rq)}


def kit_labels(replay, human):
    """The labelled questions of each item, as the labeler check asks and keeps them: replay = <replay>/memory-items.jsonl (its
    labels on questions.json's questions), human = the owner's labels file (its labels other than "unsure" on the questions it
    asks)."""
    replay = Path(replay).expanduser()
    asked = json.loads((replay / "questions.json").read_text(encoding="utf-8"))
    rep = [{q: lab for q, lab in json.loads(line)["labels"].items() if q in asked}
           for line in (replay / "memory-items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hum = []
    for item in json.loads(Path(human).expanduser().read_text(encoding="utf-8")):
        qs = {q for q in item["questions"] if q in item["labels"]} or set(item["questions"])
        hum.append({q: lab for q, lab in item["labels"].items() if q in qs and lab != "unsure"})
    return {"replay": rep, "human": hum}


def answered(item, want):
    """One dump item against the kit's: every labelled question answered ([choice, label, p] with a choice), the kit's label (so
    item i is the kit's item i), and nothing else."""
    return isinstance(item, dict) and item.keys() == want.keys() and all(
        isinstance(a, list) and len(a) == 3 and isinstance(a[0], str) and a[1] == want[q] for q, a in item.items())


def part_line(B, H, want):
    """A live part as one line over every labelled question (B, H: the two sides' items; want: the kit's labels)."""
    cells = [(i, q) for i, w in enumerate(want) for q in w]
    ok_b = np.array([B[i][q][0] == B[i][q][1] for i, q in cells], float); ok_h = np.array([H[i][q][0] == H[i][q][1] for i, q in cells], float)
    d = ok_h - ok_b
    return {"n": len(d), "base": 100 * ok_b.mean(), "head": 100 * ok_h.mean(), **item_bootstrap(d, np.array([i for i, _ in cells])),
            "fixes_breaks": (int((d > 0).sum()), int((d < 0).sum()))}, d


def labeler_lines(path, base, head, card, diffs, replay, human):
    try:
        dump = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        kit = kit_labels(replay, human)
    except (OSError, ValueError, KeyError) as e:
        card["missing"].append(f"labeler dump or kit unreadable ({type(e).__name__}: {e})"); return
    gaps = [f"labeler dump: no '{n}' side (has {sorted(dump)})" for n in (base, head) if n not in dump]
    for n in (base, head):
        if n not in dump:
            continue
        for part, want in kit.items():
            items = dump[n].get(part) if isinstance(dump[n], dict) else None
            if not isinstance(items, list) or len(items) != len(want):
                gaps.append(f"labeler dump: '{n}' {part} has {len(items) if isinstance(items, list) else 'no'} items, the kit {len(want)}"); continue
            bad = [i for i, (item, w) in enumerate(zip(items, want)) if not answered(item, w)]
            if bad:
                gaps.append(f"labeler dump: '{n}' {part}: {len(bad)} of {len(want)} items lack a labelled question's answer or the kit's label (first: item {bad[0]})")
    if gaps:
        card["missing"] += gaps; return
    for part, want in kit.items():
        B, H = dump[base][part], dump[head][part]
        if any(want):
            card["live"][f"live:{part}"], diffs[f"live:{part}"] = part_line(B, H, want)
        rare = {}
        for q in ("type", "blast", "sev"):
            pairs = [(b[q], h[q]) for b, h in zip(B, H) if q in b]
            if not pairs:
                continue
            fx = sum(b[0] != b[1] and h[0] == h[1] for b, h in pairs); br = sum(b[0] == b[1] and h[0] != h[1] for b, h in pairs)
            card["live"][f"{part}:{q}"] = {"n": len(pairs), "base": 100 * sum(b[0] == b[1] for b, _ in pairs) / len(pairs),
                                           "head": 100 * sum(h[0] == h[1] for _, h in pairs) / len(pairs), "fixes": fx, "breaks": br,
                                           "sign_p": sign_test(fx, br)}
            for tag in ("P0", "P1", "massive", "security"):
                nb = sum(str(b[0]).endswith(tag) for b, _ in pairs); nh = sum(str(h[0]).endswith(tag) for _, h in pairs)
                if nb or nh:
                    rare[tag] = [nb, nh]
        card["live"][f"{part}:rare_predictions_base_head"] = rare


def unanswered(failures):
    """quality_gate's failures for a request a side did not answer in full, counted by group (a demo, the replay, or
    'suite <name>' for a suite's predictions)."""
    out = Counter()
    for f in failures:
        if any(s in f for s in UNANSWERED):
            out[f.split(": ")[0] if f.startswith("suite ") else f.split(" | ")[0]] += 1
    return out


def report_lines(rep, card):
    """Completeness of the demos, the replay and the latency; V2 and V3."""
    g = rep.get("groups") or {}

    def raw(x, k):
        return g[x].get(k, 0) if isinstance(g.get(x), dict) else 0

    def full(x, k, want):   # quality_gate writes ints: NaN, a float, a string, null or a bool is not a count
        v = raw(x, k)
        return isinstance(v, int) and not isinstance(v, bool) and v >= want
    short = [f"{x} {raw(x, 'requests')!r}/{r} requests, {raw(x, 'questions')!r}/{q} questions"
             for x, (r, q) in {**DEMOS, "labeler-replay": REPLAY}.items()
             if not (full(x, "requests", r) and full(x, "questions", q))]
    if short:
        card["missing"].append("demos/replay short (answered by both sides): " + ", ".join(short))
    u = unanswered(rep.get("failures") or [])
    if u:
        card["missing"].append(f"requests not answered in full by both sides: {sum(u.values())} quality_gate failures ("
                               + ", ".join(f"{k} {v}" for k, v in u.items()) + ")")
    card["demo_flips"] = {x: g[x]["flips"] for x in g if g[x].get("flips")}
    lat, floor = (rep.get("latency") or {}).get("all text"), (rep.get("floor") or {}).get("all text")
    if not all(isinstance(x, dict) and isinstance(x.get("ratio"), (int, float)) and not isinstance(x.get("ratio"), bool)
               and math.isfinite(x["ratio"]) and x["ratio"] > 0 for x in (lat, floor)):
        card["missing"].append("latency: no finite positive 'all text' ratio for the head or the floor (no text request answered by both sides)")
    else:
        card["latency"] = {"ratio": lat["ratio"], "floor": floor["ratio"]}
        card["vetoes"]["V3 latency ratio <= floor + 0.015"] = lat["ratio"] <= floor["ratio"] + 0.015
    card["vetoes"]["V2 human read of flips (" + ", ".join(x for x in READ_FLIPS if card["demo_flips"].get(x)) + ")"] = "NEEDS HUMAN READ"


# --- the verdict --------------------------------------------------------------------------------------------------------

def decide(missing, vetoes, p, wins, losses):
    """The verdict from the parts of the scorecard: INCOMPLETE, VETO, BETTER or NOT BETTER (module docstring)."""
    hard = {k: v for k, v in vetoes.items() if isinstance(v, bool)}
    if p:
        net = (f"pooled {p['delta']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] over {p['lines']} lines ({p['lines'] - p['live']} suites, "
               f"{p['live']} live); {len(wins)} wins, {len(losses)} losses" + (f": {', '.join(losses)}" if losses else ""))
    if missing:
        return f"INCOMPLETE ({len(missing)} missing): " + "; ".join(missing[:6]) + (f"; +{len(missing) - 6} more" if len(missing) > 6 else "")
    if not all(hard.values()):
        return "VETO: " + ", ".join(dict.fromkeys(k.split(" ")[0] for k, v in hard.items() if not v))
    if p and p["delta"] >= 0 and (p["lo"] > 0 or (len({SOURCE.get(w, w) for w in wins}) >= 3 and not losses)):
        return f"BETTER ({net}); human read of flips pending"
    if p and p["delta"] > 0 and len(wins) >= 2 and not losses:
        return f"NOT BETTER (near miss: {net}); not served or published, one same-recipe rerun with a new seed"
    return "NOT BETTER" + (f" ({net})" if p else "")


def scorecard(gate, labeler=None, base_name="base", head_name="head", replay=ROOT / "runs/labeler-replay", human=None,
              extra_gate=None, expect_suites=None):
    """The scorecard as a dict; its "verdict" is the line the rule acts on."""
    gate = Path(gate).expanduser()
    card = {"suites": {}, "pr": {}, "slices": {}, "live": {}, "missing": [], "report_only_missing": [], "vetoes": {}, "targets": {},
            "follow_ups": []}
    expected = EXPECTED
    if expect_suites:
        expected = tuple(dict.fromkeys([*CARD, *(gate_name(s) for s in expect_suites.split(",") if s)]))
        card["expected_override"] = list(expected)
    diffs = {}   # line -> per-question correctness differences (head - base), for the pooled mean
    suites_of(gate, card, diffs, expected=expected)
    if extra_gate:
        if Path(extra_gate).expanduser().exists():
            suites_of(Path(extra_gate).expanduser(), card, {}, tag="report-only:")
        else:
            card["report_only_missing"].append(f"no report-only gate at {extra_gate}")
    pr_lines(gate, card)
    slices_of(gate, card)
    if not (labeler and Path(labeler).expanduser().exists()):
        card["missing"].append("labeler dump")
    elif not human:
        card["missing"].append("labeler kit: no --human (the owner's labels)")
    else:
        labeler_lines(labeler, base_name, head_name, card, diffs, replay, human)
    if (gate / "report.json").exists():
        report_lines(json.loads((gate / "report.json").read_text(encoding="utf-8")), card)
    else:
        card["missing"].append("report.json")
    S, L = card["suites"], card["live"]
    for k in CARD:
        if S.get(k):
            card["vetoes"][f"V1 {k} delta >= -2.0 and CI lower >= -4.0"] = S[k]["delta"] >= -2.0 and S[k]["lo"] >= -4.0
    cat = [k for k, v in S.items() if not v["report_only"] and v["n"] >= 150 and v["delta"] <= -5.0 and v["hi"] < 0]
    card["vetoes"]["V4 no suite (n >= 150) worse by 5 points or more with its CI below 0" + (f" ({', '.join(cat)})" if cat else "")] = not cat
    # the net picture: one line per suite and per live part, plus the severity a served offset gives
    lines = {**{k: v for k, v in S.items() if not v["report_only"] and k not in MONITOR},
             **{k: v for k, v in L.items() if k.startswith("live:")},
             **{k: v for k, v in card["pr"].items() if k == "pr-labels_test:sev+offsets"}}
    wins = sorted(k for k, v in lines.items() if v["lo"] > 0); losses = sorted(k for k, v in lines.items() if v["hi"] < 0)
    card["pooled"] = {**pooled(list(diffs.values())), "live": sum(k.startswith("live:") for k in diffs)} if diffs else None
    card["wins"], card["losses"] = wins, losses
    # targets: a miss opens an issue, never a veto
    P = card["pr"]; T = card["targets"]
    if P.get("pr-labels_test:sev+offsets"):
        T["sev+offsets head >= base+offsets"] = bool(P["pr-labels_test:sev+offsets"]["delta"] >= 0)
    if L.get("replay:blast"):
        r = L["replay:blast"]; T["replay blast head >= base"] = r["head"] >= r["base"]
    for part in ("replay", "human"):
        if f"{part}:rare_predictions_base_head" not in L:
            continue
        rare = L[f"{part}:rare_predictions_base_head"]
        T[f"{part}: P0/P1/massive/security predictions <= base + 3"] = all(h <= b + 3 for b, h in rare.values())
    f = S.get("routing_factory-development")
    if f:
        T["factory coverage@0.9 head >= base"] = f["cov90"][1] >= f["cov90"][0]
    T["no non-fit suite ECE worse by > 0.02 (CI lower > 0)"] = not [k for k, v in S.items() if "ece_delta" in v and v["ece"][1] - v["ece"][0] > 0.02 and v["ece_delta"]["lo"] > 0]
    sk = [S[k]["delta"] for k in CARD if S.get(k)]
    if sk:
        T["mean card-suite delta >= -0.5"] = sum(sk) / len(sk) >= -0.5
    # follow-ups: every significant loss (lines; monitors and report-only suites, which are not lines; per-question lines by
    # the sign test, since a bootstrap CI over a handful of items is degenerate) and every missed target
    per_q = {**{k: tuple(v["fixes_breaks"]) for k, v in P.items() if "fixes_breaks" in v},
             **{k: (v["fixes"], v["breaks"]) for k, v in L.items() if "sign_p" in v}}
    card["follow_ups"] = ([f"significant loss: {k}" for k in losses]
                          + [f"significant loss (monitor): {k}" for k in MONITOR if S.get(k) and S[k]["hi"] < 0]
                          + [f"significant loss (report-only): {k}" for k, v in S.items() if v["report_only"] and v["hi"] < 0]
                          + [f"significant loss (per question, {br} breaks / {fx} fixes, sign test p {sign_test(fx, br):.1g}): {k}"
                             for k, (fx, br) in per_q.items() if br > fx and sign_test(fx, br) < SIGNIFICANT]
                          + [f"target missed: {k}" for k, v in T.items() if v is False])
    verdict = decide(card["missing"], card["vetoes"], card["pooled"], wins, losses)
    if expect_suites:
        verdict += f" [DRY RUN: --expect-suites, {len(expected)} of the {len(EXPECTED)} --all-suites partitions expected; not a deploy verdict]"
    card["verdict"] = verdict
    return card


def show(card):
    S, P, L, T = card["suites"], card["pr"], card["live"], card["targets"]
    for k, v in S.items():
        print(f"{k:42s} n={v['n']:5d} {v['base']:.1f}->{v['head']:.1f} d={v['delta']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}] fix/brk {v['fixes_breaks']} "
              f"ECE {v['ece'][0]:.3f}->{v['ece'][1]:.3f} cov@.9 {v['cov90'][0]:.2f}->{v['cov90'][1]:.2f}" + ("  [monitor]" if k in MONITOR else ""))
    for k, v in P.items():
        print(f"{k:42s} n={v['n']:5d} {v['base']:.1f}->{v['head']:.1f} d={v['delta']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}]"
              + (f" P1 {v['P1']} bonus {v['bonus']} ({v['how']})" if "P1" in v else ""))
    for k, v in L.items():
        if k.startswith("live:"):
            print(f"{k:42s} n={v['n']:5d} {v['base']:.1f}->{v['head']:.1f} d={v['delta']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}] fix/brk {v['fixes_breaks']}")
        else:
            print(f"{k:42s} {v}")
    print("slices", card["slices"]); print("demo flips", card.get("demo_flips")); print("latency", card.get("latency")); print("pooled", card["pooled"])
    for k, v in {**card["vetoes"], **T}.items():
        print(("VETO " if k in card["vetoes"] else "TARGET ") + ("PASS " if v is True else "FAIL " if v is False else f"{v} ") + k)
    print("report-only missing (not in the verdict):", card["report_only_missing"])
    print("missing:", card["missing"]); print("follow-ups:", card["follow_ups"]); print("VERDICT:", card["verdict"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", required=True, help="quality_gate.py's --out (suites/<suite>/{base,head}/rows.json, report.json)")
    ap.add_argument("--extra-gate", help="a report-only gate (its own --out): scored and listed, never in the verdict")
    ap.add_argument("--labeler", help="the live labeler check's dump: per side name, its answers to the replay kit's items")
    ap.add_argument("--base-name", required=True, help="the base side's name in the labeler dump (e.g. v0.5)")
    ap.add_argument("--head-name", required=True, help="the head side's name in the labeler dump")
    ap.add_argument("--replay", default=str(ROOT / "runs/labeler-replay"), help="the replay kit (questions.json, memory-items.jsonl); private")
    ap.add_argument("--human", help="the owner's labels for the live:human line (human-labeled.json); private")
    ap.add_argument("--out", help="also write the scorecard as JSON here")
    ap.add_argument("--expect-suites", help="dry runs only: the suites (quality_gate --suites names) a gate made without "
                    "--all-suites was asked for; the five card suites stay expected, and the verdict is marked DRY RUN")
    a = ap.parse_args(argv)
    card = scorecard(a.gate, a.labeler, a.base_name, a.head_name, a.replay, a.human, a.extra_gate, a.expect_suites)
    show(card)
    if a.out:
        out = Path(a.out).expanduser()
        out.write_text(json.dumps(card, indent=1, default=str) + "\n", encoding="utf-8"); print("wrote", out)
    return card


if __name__ == "__main__":
    main()
