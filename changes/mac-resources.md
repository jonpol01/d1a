### Added

- AGENTS.md: a "Mac resources" section (mandatory). Long jobs on the M1 Max start behind a disk and RAM preflight, fp32 models of 4B or more run only overnight with 25 GB free, a guard pauses D1A jobs on low disk or fast swap growth, and Hugging Face cache entries are deleted only through `scan_cache_dir` (#227).
