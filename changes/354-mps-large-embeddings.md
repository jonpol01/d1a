### Fixed

- PyTorch on MPS: Gemma 4's per-layer embeddings came back as zeros for every token past row 2^31 / the table's width
  (torch before 2.14, pytorch/pytorch#188756): from id 239,675 on E2B (tabs, some CJK, emoji, the image tokens) and from
  199,729 on E4B (every punctuation and digit token). D1A now reads tables that big by indexing on MPS (#354, #353), in
  training, scoring, PyTorch chat and the PyTorch media server. MLX, CUDA and the CPU were never affected.
