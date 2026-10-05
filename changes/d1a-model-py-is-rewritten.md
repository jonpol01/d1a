### Changed

- `d1a/model.py` is rewritten in D1A's own code, no longer derived from Kev. Every public name stays, and the encoding,
  the masks, every scoring path (batched, row form, prefix cache, shared prefix) and the training gradients are identical
  to the previous version on tiny Gemma 4 and Qwen3.5 models (`scripts/equivalence/model.py`, new), and on real weights
  (`scripts/equivalence/real_weights.py --module d1a/model.py`).
