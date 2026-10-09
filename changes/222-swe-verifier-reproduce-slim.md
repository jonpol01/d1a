### Changed

- `recipes/swe-verifier/reproduce.sh` (#222) rebuilds the self-improvement texts (`self-improve/feedback.jsonl`,
  `train-r{1,2,3}.jsonl`) from `nebius/SWE-agent-trajectories` at the pinned revision with the new `rebuild_texts.py`,
  and checks each against the manifest's sha256. `JohnP1/d1a-swe-verifier-eval` now ships them as ids, labels and scores
  (`*.ids.jsonl`, each decision naming its source shard, row and run_key) instead of re-hosting the source's text.
