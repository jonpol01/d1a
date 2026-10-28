### Changed

- The Hugging Face Space runs D1A-E4B-it QAT (#363), pinned by commit, in place of D1A-E4B v0.6.1, whose answers there were
  wrong; under ZeroGPU it now loads the adapter on the CPU before moving the model to the GPU.
