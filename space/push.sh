#!/bin/bash
# Publish space/ to the Hugging Face Space, with d1a pinned to a commit that is on origin/main:
#   space/push.sh [<commit>]        (default: origin/main)
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch -q origin
SHA=$(git rev-parse "${1:-origin/main}")
git merge-base --is-ancestor "$SHA" origin/main || { echo "$SHA is not on origin/main"; exit 1; }
OUT=$(mktemp -d)
cp space/app.py space/README.md "$OUT/"
sed "s/D1A_SHA/$SHA/" space/requirements.txt > "$OUT/requirements.txt"
hf upload JohnP1/d1a "$OUT" . --repo-type space --commit-message "d1a $SHA"
rm -rf "$OUT"
