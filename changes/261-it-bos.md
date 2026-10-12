### Fixed

- Gemma 4 instruction-tuned trunks (`google/gemma-4-E2B-it`) are encoded with the `<bos>` their model was trained with. The
  -it tokenizer adds no special tokens (its base twin adds `<bos>`), so `d1a.core.encoding.layout()` derived no leading ids
  and every -it record would have started without `<bos>`. The layout now falls back to the tokenizer's `bos_token_id`
  when it adds nothing up front; Qwen (no bos token) and Gemma base are unchanged (#227).
