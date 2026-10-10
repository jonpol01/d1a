### Changed

- `GET /v1/models` lists only `d1a-latest`; `kev-latest` and `jev-latest` are no longer listed. The server still answers
  any model name a request sends, so existing clients and an unconfigured TypeSafe SDK client keep working.
