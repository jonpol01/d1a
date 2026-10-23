### Fixed

- The torch-only media server (`python -m d1a.serving.media`) answers videos with transformers 5.20, whose
  `Gemma4Model.forward` drops the frame count `get_video_features` needs (#332).
