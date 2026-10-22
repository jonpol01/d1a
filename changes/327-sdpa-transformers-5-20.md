### Fixed

- Gemma 4 loads under transformers 5.20 when SDPA is asked for (#324): the CUDA default, `D1A_ATTN=sdpa`, the torch
  serving default on Apple GPUs and the torch-only media server. 5.20 refuses `attn_implementation="sdpa"` by name for
  Gemma 4 (its audio tower has none); D1A now leaves the choice to transformers, which gives the text model SDPA on
  5.17 and 5.20 alike. Answers are bit-identical to before on 5.17.
