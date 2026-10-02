"""Translated copies of PR-labeler records, for any language: the title and description translated by an instruction
model, everything else (author, stats, files, questions, labels) unchanged. The questions stay in English, as the
labeling job asks them. Descriptions are cut to 1,200 characters before translating. clean.py drops broken outputs.

    # on a Mac: Gemma 4 E4B on MLX, batched
    python translate.py ja train.jsonl train_ja.jsonl --n 960 --seed 3
    # anywhere else: any OpenAI-compatible chat endpoint (vLLM, LM Studio, Ollama, llama.cpp server)
    python translate.py ko train.jsonl train_ko.jsonl --n 960 --endpoint http://localhost:8000/v1 --model google/gemma-4-E4B-it

LANG is a code such as ja, ko, zh, es, fr, de, pt, vi, id, th, ru (other codes are passed to the model as written).
"""
import argparse, json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor

MLX_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Simplified Chinese", "es": "Spanish", "fr": "French", "de": "German",
         "pt": "Brazilian Portuguese", "vi": "Vietnamese", "id": "Indonesian", "th": "Thai", "ru": "Russian"}
PROMPT = ("Translate this GitHub pull request into natural {lang}, as a {lang}-speaking developer would write it. Keep code, "
          "file paths, identifiers, commands, URLs and Markdown symbols exactly as they are. Reply in exactly this form and "
          "nothing else:\nTITLE: <{lang} title>\nBODY: <{lang} description>\n\nTITLE: {title}\nBODY: {body}")
FIELD = re.compile(r"(?:^|\n)[\s*_#>-]*TITLE:\s*(.+?)\s*\n[\s*_#>-]*BODY:\s*(.*)", re.S)   # tolerate Markdown around the labels


def split_state(state):
    return state.split("\n", 1)[0].removeprefix("title: "), state.split("\nbody:\n", 1)[1].split("\nfiles:", 1)[0]


def mlx_generate(prompts):
    """Batched generation on Apple Silicon; yields one text per prompt, in order."""
    from mlx_lm import batch_generate, load
    model, tok = load(MLX_MODEL)
    for i in range(0, len(prompts), 32):
        chunk = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True) for p in prompts[i:i + 32]]
        yield from batch_generate(model, tok, chunk, max_tokens=700).texts


def endpoint_generate(prompts, endpoint, model, workers=8):
    """Any OpenAI-compatible /chat/completions endpoint, a few requests in flight; yields one text per prompt, in order."""
    import urllib.request
    def one(p):
        body = {"model": model, "temperature": 0, "max_tokens": 700, "messages": [{"role": "user", "content": p}]}
        req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            print("request failed:", str(e)[:120], file=sys.stderr); return ""
    with ThreadPoolExecutor(workers) as ex:
        yield from ex.map(one, prompts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lang"); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=0, help="translate this many records, sampled with --seed (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL (…/v1); without it, Gemma 4 E4B on MLX (Apple Silicon)")
    ap.add_argument("--model", default="gemma-4-e4b-it", help="model id for --endpoint")
    a = ap.parse_args()
    name = NAMES.get(a.lang, a.lang)
    recs = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    random.Random(a.seed).shuffle(recs)
    if a.n: recs = recs[:a.n]
    jobs = []
    for r in recs:
        title, body = split_state(r["state"])
        jobs.append((r, title, body, PROMPT.format(lang=name, title=title, body=body[:1200])))
    jobs.sort(key=lambda j: len(j[3]))
    texts = endpoint_generate([j[3] for j in jobs], a.endpoint, a.model) if a.endpoint else mlx_generate([j[3] for j in jobs])
    done, t0 = 0, time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        for i, ((r, title, body, _), text) in enumerate(zip(jobs, texts), 1):
            m = FIELD.search(text or "")
            if m and "BODY" not in m[1]:
                new_title, new_body = m[1].strip(), re.sub(r"\s+", " ", m[2]).strip()
                if (new_title, new_body) != (title, body):
                    state = r["state"].replace(f"title: {title}", f"title: {new_title[:240]}", 1)
                    state = state.replace(f"body:\n{body}\nfiles:", f"body:\n{new_body[:3500] or '(empty)'}\nfiles:", 1)
                    f.write(json.dumps({**r, "state": state, "id": f"{r['id']}:{a.lang}"}, ensure_ascii=False) + "\n"); done += 1
            if i % 32 == 0 or i == len(jobs): print(f"{i}/{len(jobs)} done, {done} kept, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
