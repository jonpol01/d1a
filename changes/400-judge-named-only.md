### Changed

- Outcomes from an independent blind judge (`src: "judge"`) count only for a reader that names that source (#400): `--src`,
  a source list, or a per-question `outcomes.sources` that lists it under the question or `"*"`. A reader of every source,
  the default `outcomes.sources`, and a question a per-question filter does not cover keep every source but judge, so a
  later judge label never replaces the reviewer's there.
