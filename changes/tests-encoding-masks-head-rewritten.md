### Changed

- The tests of the packed encoding, the block-causal masks and the pointer head are rewritten as `tests/test_encoding.py`,
  no longer derived from Kev. The new tests check:
  - every mask against the attention rule, on every query and key pair: random layouts, a padded batch, and Gemma 4's
    sliding window;
  - delimiter forgery under both the Qwen and the Gemma 4 tokenizer, each given the other's control tokens too;
  - the whole layout, through `rows_of`;
  - that `PointerHead.many` scores each question as `forward` does.
