#!/bin/bash
# Launch train_job.sh on Hugging Face Jobs (one NVIDIA L4, $0.80/hour at the time of writing). Resume points go to an HF
# bucket, so a job that hits its timeout can be relaunched with the same PREFIX and continues where it stopped.
#
#   ./launch_hf_job.sh skills-v1 8h                         # from the B0 skills checkpoint (#175)
#   ./launch_hf_job.sh skills-v1 8h INIT=JohnP1/d1a-e4b@v0.4
#
# Arguments: PREFIX (the folder for this run's checkpoint), TIMEOUT (a hard cap on the bill; budget ~1.3 s per record,
# 15,320 records, plus ~30 min of setup, smoke and calibration), then train_job.sh variables as NAME=value. Needs
# `hf auth login` with a token that can write to REPO and BUCKET. REPO, BUCKET, FLAVOR in the environment override the defaults.
set -euo pipefail
PREFIX=${1:?PREFIX}; TIMEOUT=${2:?TIMEOUT}; shift 2
REPO=${REPO:-JohnP1/d1a-e4b-runs}; BUCKET=${BUCKET:-hf://buckets/JohnP1/d1a-train}; FLAVOR=${FLAVOR:-l4x1}
SHA=$(git rev-parse origin/main)   # the job clones this commit, so it must be pushed
ENV=(-e D1A_SHA="$SHA" -e REPO="$REPO" -e PREFIX="$PREFIX")
for kv in "$@"; do ENV+=(-e "$kv"); done
hf jobs run -d --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN "${ENV[@]}" -v "$BUCKET:/ckpt" \
  ghcr.io/astral-sh/uv:python3.13-bookworm bash -c "apt-get -qq update >/dev/null && apt-get -qq install -y git >/dev/null; $(cat "$(dirname "$0")/train_job.sh")"   # the image has no git
