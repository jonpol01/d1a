### Fixed

- Video decisions work with transformers 5.20 (#322). Its video processor sends a video's frames in one flat batch with
  `num_frames_per_video`, which its `get_video_features` requires; the media encoder now passes it when the processor
  gives it. Fresh installs, which resolve to 5.20, answered every video request with a 422. The soft tokens are
  bit-identical to transformers 5.17's.
