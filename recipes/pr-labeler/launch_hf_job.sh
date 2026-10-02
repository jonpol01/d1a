#!/bin/bash
# Launch train_job.sh on Hugging Face Jobs (one NVIDIA L4, $0.80/hour at the time of writing). Resume points go to an HF
# bucket, so a job that hits its timeout can be relaunched with the same PREFIX and continues where it stopped.
#
#   ./launch_hf_job.sh pr-labeler-v2 5h EN=3500 EN_SKIP=1 INIT=JohnP1/d1a-e4b@v0.3
#
# Arguments: PREFIX (the folder for this run's checkpoint and scores), TIMEOUT (a hard cap; budget ~1.26 s per PR record
# plus ~1 h of scoring), then any train_job.sh variables as NAME=value. Needs `hf auth login` with a token that can write
# to REPO and BUCKET. Set REPO, BUCKET, FLAVOR in the environment to override the defaults below.
set -euo pipefail
PREFIX=${1:?PREFIX}; TIMEOUT=${2:?TIMEOUT}; shift 2
REPO=${REPO:-JohnP1/d1a-e4b-runs}; BUCKET=${BUCKET:-hf://buckets/JohnP1/d1a-train}; FLAVOR=${FLAVOR:-l4x1}
SHA=$(git rev-parse origin/main)   # the job clones this commit, so it must be pushed
ENV=(-e D1A_SHA="$SHA" -e REPO="$REPO" -e PREFIX="$PREFIX")
for kv in "$@"; do ENV+=(-e "$kv"); done
hf jobs run -d --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN "${ENV[@]}" -v "$BUCKET:/ckpt" \
  ghcr.io/astral-sh/uv:python3.13-bookworm bash -c "$(cat "$(dirname "$0")/train_job.sh")"
