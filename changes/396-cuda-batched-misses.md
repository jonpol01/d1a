### Changed

- On NVIDIA GPUs, a PyTorch checkpoint serves requests that arrive together and whose state is not cached in packed passes
  of several, instead of one at a time (#396). Each pass holds requests of similar length, up to 16,384 tokens, and a
  request's answers do not depend on its batch. Cached states, kept prefixes, media, MLX builds and other devices are
  served as before.
