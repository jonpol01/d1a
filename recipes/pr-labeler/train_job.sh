#!/bin/bash
# PR-labeler fine-tune on one CUDA GPU (an L4 is enough): built for Hugging Face Jobs (launch_hf_job.sh), and runs the
# same on your own NVIDIA machine in that image (it uses absolute paths: /d1a, /data, /init, /runs). Trains from INIT on the PR-labeler records of
# the pinned D1A suites (evals/d1a: pr-labels, ja-jglue, routing; mixed by mix.py) plus replay of D1A's earlier data,
# calibrates on pooled decision-v7 + PR development rows, uploads the checkpoint, then scores INIT ("before") and the new
# checkpoint ("after") at full length on the English and Japanese test sets and the real PRs, plus forgetting checks.
#
# Environment (defaults in brackets):
#   D1A_SHA            d1a commit to train with [required]
#   REPO, PREFIX       Hub repo and folder for the checkpoint and scores [required], e.g. JohnP1/d1a-e4b-runs pr-labeler-v2
#   INIT               starting checkpoint, repo[@revision][:subfolder] [JohnP1/d1a-e4b@v0.3]
#   EN                 English PRs from train.jsonl: every rare-label PR (blast, P0, P1, P4) plus a sample of the rest; 0 = all [0]
#   EN_SKIP            1 = continue a previous round (same EN, same seed): its unsampled PRs plus its rare-label PRs again [0]
#   EXTRA              more training partitions of evals/d1a/pr-labels, space-separated, [train-ja train-blast]; mix.py refuses a mix that leaves one out
#   REPLAY_DV7, REPLAY_JA, REPLAY_ROUTING   replay of decision-v7, Japanese JGLUE, routing [1500, 500, 300]
#   LR, MAX_STATE, CKPT  learning rate, longest PR document in tokens, resume-point directory [5e-5, 1536, /ckpt]
# Measured on an L4: ~1.26 s per PR record (PR documents average ~680 tokens); budget the timeout from that.
set -uo pipefail
: "${D1A_SHA:?}" "${REPO:?}" "${PREFIX:?}"
INIT=${INIT:-JohnP1/d1a-e4b@v0.3}; EN=${EN:-0}; EN_SKIP=${EN_SKIP:-0}
EXTRA=${EXTRA:-"train-ja train-blast"}; REPLAY_DV7=${REPLAY_DV7:-1500}; REPLAY_JA=${REPLAY_JA:-500}; REPLAY_ROUTING=${REPLAY_ROUTING:-300}
LR=${LR:-0.00005}; MAX_STATE=${MAX_STATE:-1536}; CKPT=${CKPT:-/ckpt}
export INIT EN EN_SKIP EXTRA REPLAY_JA REPLAY_ROUTING
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -d /d1a ] || git clone -q https://github.com/jonpol01/d1a /d1a
cd /d1a && git checkout -q "$D1A_SHA"
export UV_LINK_MODE=copy PYTHONUNBUFFERED=1
uv sync -q --frozen --python 3.13 --extra serve
uv run --frozen python - <<'PY'
import os
from huggingface_hub import snapshot_download
spec, _, sub = os.environ["INIT"].partition(":"); repo, _, rev = spec.partition("@")
snapshot_download(repo, revision=rev or None, allow_patterns=[f"{sub}/*"] if sub else None, local_dir="/init/raw")
os.symlink(f"/init/raw/{sub}" if sub else "/init/raw", "/init/start")
PY
mkdir -p /data
# shellcheck disable=SC2086  # EXTRA is a list of partition names
uv run --frozen python recipes/pr-labeler/mix.py --out /data/train.jsonl --en "$EN" --en-skip "$EN_SKIP" --extra $EXTRA \
  --replay-ja "$REPLAY_JA" --replay-routing "$REPLAY_ROUTING" || { echo "FAILED training mix"; exit 1; }
upload() {
  uv run --frozen python - "$1" "$2" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(); api.create_repo(os.environ["REPO"], private=True, exist_ok=True)
api.upload_folder(folder_path=sys.argv[1], path_in_repo=f"{os.environ['PREFIX']}/{sys.argv[2]}", repo_id=os.environ["REPO"],
                  commit_message=f"{os.environ['PREFIX']}: {sys.argv[2]}", ignore_patterns=["README.md", "**/README.md"])
print("uploaded", sys.argv[2], flush=True)
PY
}
COMMON="--base google/gemma-4-E4B --base_revision 411aa17b749aa952df1359d2dcea73917a544d9a --init_from /init/start \
 --data /data/train.jsonl --suite evals/v7/decision-v7 --replay $REPLAY_DV7 --max_state $MAX_STATE --device cuda --lora 16 \
 --batch 2 --accum 4 --lr $LR --dtype bf16 --weights_dtype bf16 --checkpointing 1 --seed 0"
echo "== smoke: 5 steps + checkpoint"
uv run --frozen python -m d1a.training.train $COMMON --epochs 1 --max_steps 5 --out /runs/smoke 2>&1 | grep -E "replay:|training requests|dropped|saved|Error" | tail -6
[ -f /runs/smoke/head.pt ] || { echo "FAILED smoke wrote no checkpoint"; exit 1; }
echo "== train"
status=0
uv run --frozen python -m d1a.training.train $COMMON --epochs 1 --save_every_minutes 15 --resume 1 --out "$CKPT/$PREFIX" 2>&1 | tee /runs/train.log || status=$?
rm -rf /runs/new && cp -r "$CKPT/$PREFIX" /runs/new
[ -f /runs/new/head.pt ] || { echo "FAILED train exit $status"; mkdir -p /runs/log && cp /runs/train.log /runs/log/; upload /runs/log log; exit 1; }
cp /runs/train.log /runs/new/; upload /runs/new checkpoint || { echo "FAILED checkpoint upload"; exit 1; }
echo "== calibrate (pooled: decision-v7 calibration + PR development)"
uv run --frozen python -m d1a.eval.benchmark --run /runs/new --suite evals/v7/decision-v7 --split calibration --device cuda --out /runs/cal 2>&1 | tail -1
uv run --frozen python -m d1a.eval.benchmark --run /runs/new --data evals/d1a/pr-labels:development --context serving --device cuda --out /runs/calpr 2>&1 | tail -1
uv run --frozen python -m d1a.training.calibrate --run /runs/new --rows /runs/cal/rows.json --rows /runs/calpr/rows.json --allow-in-distribution 2>&1 | tail -2
upload /runs/new checkpoint || { echo "FAILED calibrated checkpoint upload"; exit 1; }
echo "== scoring at full length (each result uploaded as it lands)"
mkdir -p /runs/eval
score() { uv run --frozen python -m d1a.eval.benchmark --run "$1" "${@:3}" --device cuda --out /runs/eval/$2 2>&1 | tail -1; upload /runs/eval eval >/dev/null; echo "scored $2"; }
for d in real17 test-ja test; do
  score /runs/new after-pr-$d --data evals/d1a/pr-labels:$d --context serving
  score /init/start before-pr-$d --data evals/d1a/pr-labels:$d --context serving
done
score /runs/new after-decision-v7 --suite evals/v7/decision-v7
score /runs/new after-ja --data evals/d1a/ja-jglue:development
score /runs/new after-routing-factory --data evals/d1a/routing:factory-development
echo DONE
