"""Drop broken machine translations: a description cut to under a quarter of the original's first 1,200 characters, a
title and description mixed up, or (for languages with their own script) a description not in that script. A title
left in English over a translated description is kept: developers often write titles like `fix(api): ...` in English.
Usage: clean.py LANG ORIGINAL.jsonl TRANSLATED.jsonl OUT.jsonl"""
import json, re, sys
SCRIPT = {"ja": r"[぀-ヿ一-鿿]", "zh": r"[一-鿿]", "ko": r"[가-힯]", "th": r"[฀-๿]", "ru": r"[Ѐ-ӿ]"}


def parts(state):
    return state.split("\n", 1)[0].removeprefix("title: "), state.split("\nbody:\n", 1)[1].split("\nfiles:", 1)[0]


def broken(lang, r, original):
    t, b = parts(r["state"]); ot, ob = original
    if "BODY" in t or "TITLE" in b[:20]: return "mixed up"
    if ob == "(empty)": return None
    if len(b) < 0.25 * min(len(ob), 1200): return "cut short"
    if lang in SCRIPT and len(re.findall(SCRIPT[lang], b)) < 0.15 * len(b): return "not translated"
    if lang not in SCRIPT and b == ob[:len(b)]: return "not translated"
    return None


lang, src, trans, out = sys.argv[1:5]
orig = {json.loads(l)["id"]: parts(json.loads(l)["state"]) for l in open(src)}
kept, dropped = [], {}
for l in open(trans):
    r = json.loads(l); why = broken(lang, r, orig[r["id"].rsplit(":", 1)[0]])
    if why: dropped[why] = dropped.get(why, 0) + 1
    else: kept.append(l)
open(out, "w", encoding="utf-8").write("".join(kept))
print(trans.rsplit("/", 1)[-1], "kept", len(kept), "dropped", dropped)
