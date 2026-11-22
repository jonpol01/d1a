### Fixed

- Date facts (`D1A_DATE_FACTS`, `--date_facts`) find a date written right against Japanese text, such as
  `August 8, 2026までに`, the form the Japanese date rewrite writes (#414, #411). English states get byte-identical facts.
