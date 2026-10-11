### Changed

- `recipes/swe-verifier/reproduce.sh` says how to get access when it cannot download the adapter or the scored rows
  (both private at the time of writing), instead of stopping on the bare Hugging Face error; exit 2. The recipe's
  README describes reproducing the paper's tables.
