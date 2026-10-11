### Fixed

- The model server could abort on Apple Silicon ("commit an already committed command buffer") when photo, voice or video
  requests overlapped other work (#244): torch's MPS stream is not thread-safe, yet an MLX model's thread synced it on
  every batch while the media encoder used it from a request thread. With an MLX model the model thread no longer touches
  torch's MPS stream (MLX waits for its own work), and the encoder copies its result to the CPU under its lock. Text and
  media still run side by side.
