### Changed

- APPLE.md (en, ja, ko): the photo, voice and video gap both Mac builds share is a split between CUDA and the CPU,
  not a fault of the builds: PyTorch fp32 on the CPU agrees with MLX within 0.0014, and Mac media answers are now
  measured against it (#405, #386; which side is right is #404).
