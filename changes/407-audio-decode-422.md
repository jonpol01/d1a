### Fixed

- `POST /v1/systemone/media` answers an audio clip that does not decode, or holds no samples, with a 422 and libsndfile's
  reason instead of a 500 (#406, #407). Photos (#398) and video already did.
