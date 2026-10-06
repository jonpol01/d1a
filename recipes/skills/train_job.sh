#!/bin/bash
# The skills stage (#175) on one CUDA GPU (an L4 is enough), built for Hugging Face Jobs (launch_hf_job.sh): trains from
# INIT on recipes/skills/mix.py's mix plus decision-v7 replay, calibrates the way D1A-E4B v0.4 was (pooled decision-v7
# calibration + PR development), and uploads the checkpoint. Scoring is not here: it runs where the earlier checkpoints
# were scored, so every number in the comparison comes from the same path (recipes/README.md).
#
# Environment (defaults in brackets):
#   D1A_SHA            d1a commit to train with [required]
#   REPO, PREFIX       Hub repo and folder for the checkpoint [required], e.g. JohnP1/d1a-e4b-runs skills-v1
#   INIT               starting checkpoint, repo[@revision][:subfolder] [JohnP1/d1a-e4b-runs:skills-b0-300/init]
#   REPLAY_DV7, LR, MAX_STATE, CKPT   decision-v7 replay, learning rate, state tokens, resume-point dir [1000, 2e-5, 6400, /ckpt]
#   (the mix's longest state is 6,209 tokens with the Gemma 4 tokenizer: 6,400 keeps every record)
# The first thing it runs is 5 steps on the 40 longest states (the memory worst case): a card that cannot hold them stops
# the job within minutes.
set -uo pipefail
: "${D1A_SHA:?}" "${REPO:?}" "${PREFIX:?}"
INIT=${INIT:-JohnP1/d1a-e4b-runs:skills-b0-300/init}; REPLAY_DV7=${REPLAY_DV7:-1000}
LR=${LR:-0.00002}; MAX_STATE=${MAX_STATE:-6400}; CKPT=${CKPT:-/ckpt}
export INIT
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
mkdir -p /data /runs
uv run --frozen python recipes/skills/mix.py --out /data/train.jsonl --smoke /data/smoke.jsonl || { echo "FAILED training mix"; exit 1; }
upload() {
  uv run --frozen python - "$1" "$2" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
assert api.repo_info(os.environ["REPO"]).private, "the run repo must be private"
api.upload_folder(folder_path=sys.argv[1], path_in_repo=f"{os.environ['PREFIX']}/{sys.argv[2]}", repo_id=os.environ["REPO"],
                  commit_message=f"{os.environ['PREFIX']}: {sys.argv[2]}", ignore_patterns=["README.md", "**/README.md"])
print("uploaded", sys.argv[2], flush=True)
PY
}
COMMON="--base google/gemma-4-E4B --base_revision 411aa17b749aa952df1359d2dcea73917a544d9a --init_from /init/start \
 --device cuda --lora 16 --lora_targets all --batch 2 --accum 4 --lr $LR --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
 --max_state $MAX_STATE --epochs 1 --seed 0"
echo "== smoke: 5 steps + checkpoint on the 40 longest states"
uv run --frozen python -m d1a.training.train $COMMON --data /data/smoke.jsonl --max_steps 5 --out /runs/smoke 2>&1 | grep -E "training requests|dropped|saved|Error|out of memory" | tail -6
[ -f /runs/smoke/head.safetensors ] || { echo "FAILED smoke wrote no checkpoint"; exit 1; }
uv run --frozen python -c "import json; m = json.load(open('/runs/smoke/training_metrics.json')); print('smoke peak GB', round(m['peak_device_bytes'] / 1e9, 1), 's/step', m['step_seconds'])"
echo "== train"
status=0
uv run --frozen python -m d1a.training.train $COMMON --data /data/train.jsonl --suite evals/v7/decision-v7 --replay "$REPLAY_DV7" \
  --save_every_minutes 15 --resume 1 --out "$CKPT/$PREFIX" 2>&1 | tee /runs/train.log || status=$?
rm -rf /runs/new && cp -r "$CKPT/$PREFIX" /runs/new
[ -f /runs/new/head.safetensors ] || { echo "FAILED train exit $status"; mkdir -p /runs/log && cp /runs/train.log /runs/log/; upload /runs/log log; exit 1; }
cp /runs/train.log /data/train.jsonl.json /runs/new/; upload /runs/new checkpoint || { echo "FAILED checkpoint upload"; exit 1; }
echo "== calibrate (as v0.4: pooled decision-v7 calibration + PR development)"
uv run --frozen python -m d1a.eval.benchmark --run /runs/new --suite evals/v7/decision-v7 --split calibration --identical-options 0 --device cuda --out /runs/cal 2>&1 | tail -1
uv run --frozen python -m d1a.eval.benchmark --run /runs/new --data evals/d1a/pr-labels:development --context serving --identical-options 0 --device cuda --out /runs/calpr 2>&1 | tail -1
uv run --frozen python -m d1a.training.calibrate --run /runs/new --rows /runs/cal/rows.json --rows /runs/calpr/rows.json --allow-in-distribution 2>&1 | tail -2
upload /runs/new checkpoint || { echo "FAILED calibrated checkpoint upload"; exit 1; }
echo DONE
