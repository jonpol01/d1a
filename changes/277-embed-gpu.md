### Fixed

- `/v1/embeddings` runs its GGUF embedder on the GPU (#277). On the CPU, llama-cpp-python's default, its vectors were only
  ~0.99 cosine to the ones LM Studio stored from the same file; now they are the same (1.00000 on the Mac mini).
