"""Labeled pull requests from a GitHub repository (default NousResearch/hermes-agent): closed PRs that carry the labeling
job's taxonomy (type/*, P0-P4, sweeper:blast-*), with their title, body, author, stats and up to 40 files. One JSON line
per PR; appends to OUT and skips PRs already in it. Balanced over the given labels (default: the seven type/* labels);
GitHub's search returns at most 1,000 results per query, so each label is fetched month by month. Needs the GitHub CLI
(`gh auth login`).

    python fetch_prs.py prs.jsonl 1200                                   # up to 1,200 PRs per type label
    python fetch_prs.py prs.jsonl 400 P0,P4,sweeper:blast-massive         # then top up rare labels
    python fetch_prs.py prs.jsonl 500 --repo owner/name                   # another project with the same labels"""
import calendar, json, subprocess, sys, time
from pathlib import Path

REPO = "NousResearch/hermes-agent"   # --repo overrides
TYPES = ["type/bug", "type/docs", "type/feature", "type/perf", "type/refactor", "type/security", "type/test"]
MONTHS = [(y, m) for y in (2025, 2026) for m in range(1, 13)][::-1]   # newest first
Q = """query($q: String!, $after: String) { search(query: $q, type: ISSUE, first: 50, after: $after) {
  pageInfo { hasNextPage endCursor }
  nodes { ... on PullRequest { number title body createdAt state merged additions deletions changedFiles author { login }
    labels(first: 30) { nodes { name } }
    files(first: 40) { nodes { path additions deletions changeType } } } } } }"""
STATUS = {"ADDED": "added", "DELETED": "removed", "MODIFIED": "modified", "RENAMED": "renamed", "COPIED": "copied", "CHANGED": "changed"}


def gql(q, after=None):
    args = ["gh", "api", "graphql", "-f", f"query={Q}", "-f", f"q={q}"] + (["-f", f"after={after}"] if after else [])
    for attempt in range(5):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0: return json.loads(r.stdout)["data"]["search"]
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(r.stderr[:300])


def to_row(n):
    pr = {"title": n["title"], "user": {"login": (n.get("author") or {}).get("login", "")}, "body": n.get("body") or "",
          "additions": n["additions"], "deletions": n["deletions"], "changed_files": n["changedFiles"]}
    files = [{"status": STATUS.get(f["changeType"], f["changeType"].lower()), "filename": f["path"], "additions": f["additions"], "deletions": f["deletions"]}
             for f in (n.get("files") or {}).get("nodes") or []]
    labels = [l["name"] for l in n["labels"]["nodes"]]
    return {"number": n["number"], "created": n["createdAt"], "merged": n["merged"], "labels": labels, "pr": pr, "files": files}


def main(out, per_type, labels=None):
    seen = set()
    if Path(out).exists():
        seen = {json.loads(l)["number"] for l in open(out)}
    with open(out, "a") as f:
        for t in labels or TYPES:
            got = sum(1 for l in open(out) if t in json.loads(l)["labels"]) if Path(out).exists() else 0
            for y, m in MONTHS:
                if got >= per_type: break
                q = f'repo:{REPO} is:pr is:closed label:"{t}" created:{y}-{m:02d}-01..{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}'
                after = None
                while got < per_type:
                    s = gql(q, after)
                    for n in s["nodes"]:
                        if not n or n["number"] in seen: continue
                        labels = [l["name"] for l in n["labels"]["nodes"]]
                        if sum(l in TYPES for l in labels) != 1: continue   # exactly one type: unambiguous target
                        seen.add(n["number"]); f.write(json.dumps(to_row(n)) + "\n"); got += 1
                    f.flush()
                    if not s["pageInfo"]["hasNextPage"]: break
                    after = s["pageInfo"]["endCursor"]
            print(t, got, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--repo" in args:
        i = args.index("--repo"); REPO = args[i + 1]; del args[i:i + 2]
    main(args[0], int(args[1]), args[2].split(",") if len(args) > 2 else None)
