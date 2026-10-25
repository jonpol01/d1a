### Added

- `d1a.training.train --embedding_dtype bf16`, with `--weights_dtype fp32`: the base loads in bf16 and everything but the
  embedding tables goes to fp32, so the forward is the fp32 model's for a base stored in bf16 (Gemma 4) and a QAT trunk
  (`--base_ct`) stays exact, while the tables keep half the memory. Gemma 4 E4B's fp32 trunk (16 GB) then trains beside its
  3.5B-parameter tables in 7 GB instead of 14 (#355).
