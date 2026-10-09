#!/usr/bin/env bash
# Regenerate the SWE verifier study's outputs from the released files, without re-scoring with the model: every output
# comes from the scored rows.
#
#     recipes/swe-verifier/reproduce.sh                          # the repos' main
#     REVISION=<tag or commit> recipes/swe-verifier/reproduce.sh  # a fixed release (MODEL_REVISION / DATA_REVISION override it per repo)
#
# The two repos are private until the release; REVISION pins both to one tag or commit, so a run is reproducible.
# Downloads JohnP1/d1a-swe-verifier-r1 (the r1 adapter) and JohnP1/d1a-swe-verifier-eval (the scored rows) into a fresh
# temporary directory (WORK to choose it), checks the dataset against its MANIFEST.json, reruns the scripts that wrote
# each output, and compares every regenerated output with the released one. Exit status 1 if any differs.
#   held-out evaluation (AUROC, calibration, risk-coverage, latency)   evaluate.py     -> evaluate.txt, test-merged.jsonl
#   best-of-k                                                          best_of_k.py    -> best_of_k.txt
#   within-issue AUROC on mixed-outcome issues                         within_issue.py -> within-issue.txt
#   anytime scoring and the restart-budget replay                      analyze.py, budget_sim.py -> analyze-cut10-*.txt, budget-*.txt
#   split leakage                                                      leakage.py's metrics on its released out-of-fold predictions
#   feedback rounds                                                    self-improve/curve.jsonl, as self_improve.py wrote it
# Split leakage is recomputed from leakage/predictions.npz; refitting its folds needs the full source dataset (leakage.py).
# Needs the `hf` CLI (logged in while the repos are private) and a Python with numpy, scipy, scikit-learn, pyarrow and
# huggingface_hub (PYTHON, default python3; `uv sync` provides them).
set -euo pipefail
REVISION=${REVISION:-main}
MODEL_REPO=${MODEL_REPO:-JohnP1/d1a-swe-verifier-r1}; DATA_REPO=${DATA_REPO:-JohnP1/d1a-swe-verifier-eval}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/swe-verifier-reproduce.XXXXXX")}
M=$WORK/model; D=$WORK/data; O=$WORK/out; E=$D/eval
mkdir -p "$O"
echo "work directory: $WORK"

hf download "$MODEL_REPO" --revision "${MODEL_REVISION:-$REVISION}" --local-dir "$M" > /dev/null
hf download "$DATA_REPO" --repo-type dataset --revision "${DATA_REVISION:-$REVISION}" --local-dir "$D" > /dev/null
for f in adapter_config.json adapter_model.safetensors head.pt; do [ -s "$M/$f" ] || { echo "model: $f missing" >&2; exit 1; }; done

# The download matches the manifest: every listed file with its sha256 and size, and no file left out of it.
"$PYTHON" -I - "$D" <<'EOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); man = json.loads((root / "MANIFEST.json").read_text())
listed = {f["path"]: f for f in man["files"]}
present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts
           and p.name not in ("MANIFEST.json", ".gitattributes")}   # the Hub adds .gitattributes
bad = [p for p, f in listed.items() if not (root / p).is_file() or (root / p).stat().st_size != f["size"]
       or hashlib.sha256((root / p).read_bytes()).hexdigest() != f["sha256"]]
if bad or present != set(listed):
    sys.exit(f"MANIFEST.json mismatch: changed {bad}, unlisted {sorted(present - set(listed))}, missing {sorted(set(listed) - present)}")
print(f"dataset: {len(listed)} files match MANIFEST.json")
EOF

# The same commands and flags the study ran, on the released rows.
"$PYTHON" "$HERE/evaluate.py" --train-heuristics "$E/heuristics-train.jsonl" \
  --dev zero-shot="$E/dev-zero-shot.jsonl" --dev r1="$E/dev-r1.jsonl" --test zero-shot="$E/test-zero-shot.jsonl" --test r1="$E/test-r1.jsonl" \
  --merged "$O/test-merged.jsonl" > "$O/evaluate.txt" 2>&1
"$PYTHON" "$HERE/best_of_k.py" "$O/test-merged.jsonl" --fit "$E/heuristics-train.jsonl" --k 2,4,8 > "$O/best_of_k.txt" 2>&1
"$PYTHON" "$HERE/within_issue.py" --fit "$E/heuristics-train.jsonl" zero-shot="$E/mixed-zero-shot.jsonl" r1="$E/mixed-r1.jsonl" > "$O/within-issue.txt" 2>&1
for model in zero-shot r1; do
  "$PYTHON" "$HERE/analyze.py" "$E/test-cut10-$model.jsonl" > "$O/analyze-cut10-$model.txt" 2>&1
  "$PYTHON" "$HERE/budget_sim.py" --full "$E/test-$model.jsonl" --cut "$E/test-cut10-$model.jsonl" --dev-full "$E/dev-$model.jsonl" \
    --dev-cut "$E/dev-cut10-$model.jsonl" --step 10 --budgets 30,45,60,90 > "$O/budget-$model.txt" 2>&1
done

# Split leakage: leakage.py's report and result.json, from the out-of-fold predictions it saved.
"$PYTHON" - "$HERE" "$D/leakage/predictions.npz" "$O" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from leakage import repo_bootstrap_diff, within_issue_auroc
from zero_shot import auroc, calibration
z = np.load(sys.argv[2], allow_pickle=False); out = Path(sys.argv[3])
y, groups, issues = z["y"], z["repo"], z["issue"]; rows = [{"issue": str(i)} for i in issues]
protos, names = ("random runs", "by issue", "by repo"), ("issue prior", "heuristics", "text", "heuristics + text")
preds = {p: {n: z[f"{p}|{n}"] for n in names} for p in protos}
lines = [f"{len(rows)} runs, {len(set(map(str, issues)))} issues, {len(set(groups))} repositories, {y.mean():.1%} resolved"]
result = {"runs": len(rows), "issues": len(set(map(str, issues))), "repos": len(set(groups)), "resolved": float(y.mean()), "protocols": {}}
for proto, p in preds.items():
    lines.append(f"\n{proto}:"); result["protocols"][proto] = {}
    for name, q in p.items():
        e, b, _ = calibration(y, q); w, nw = within_issue_auroc(rows, y, q)
        result["protocols"][proto][name] = {"auroc": round(auroc(y, q), 4), "ece": round(e, 4), "brier": round(b, 4), "within_issue_auroc": round(w, 4)}
        lines.append(f"  {name:18s} AUROC {auroc(y, q):.3f}  ECE {e:.3f}  Brier {b:.3f}  within-issue AUROC {w:.3f} ({nw} issues)")
lines.append("\nAUROC inflation against the by-repository split (95% CI, bootstrap over repositories):"); result["inflation"] = {}
for name in names:
    for proto in ("random runs", "by issue"):
        d, lo, hi = repo_bootstrap_diff(y, preds[proto][name], preds["by repo"][name], groups)
        result["inflation"][f"{name} | {proto}"] = [round(float(d), 4), round(float(lo), 4), round(float(hi), 4)]
        lines.append(f"  {name:18s} {proto:12s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
(out / "leakage.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out / "leakage-result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
EOF

# Feedback rounds: the served verifier after each round.
"$PYTHON" -I - "$D/self-improve/curve.jsonl" > "$O/rounds.txt" <<'EOF'
import json, sys
print("round  outcomes  test AUROC  test ECE  gate (delta log loss, 95% CI)")
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line); g = r.get("gate")
    gate = "-" if g is None else f"{'promoted' if r['promoted'] else 'rejected'}: {g['delta_log_loss']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]"
    print(f"{r['round']:>5}  {r['outcomes']:>8}  {r['test']['auroc']:>10.3f}  {r['test']['ece']:>8.4f}  {gate}")
EOF

status=0
check() {   # check NAME REGENERATED RELEASED
  if cmp -s "$2" "$3"; then echo "  match    $1"; else echo "  DIFFERS  $1"; diff "$3" "$2" | head -20 | sed 's/^/           /'; status=1; fi
}
echo "regenerated outputs against the released ones:"
for f in evaluate.txt test-merged.jsonl best_of_k.txt within-issue.txt analyze-cut10-zero-shot.txt analyze-cut10-r1.txt budget-zero-shot.txt budget-r1.txt; do
  check "eval/$f" "$O/$f" "$E/$f"
done
grep -v '^shard ' "$D/leakage/run.log" > "$O/leakage-released.txt"
check "leakage/run.log" "$O/leakage.txt" "$O/leakage-released.txt"
check "leakage/result.json" "$O/leakage-result.json" "$D/leakage/result.json"
echo; cat "$O/rounds.txt"
echo; echo "outputs: $O"
exit $status
