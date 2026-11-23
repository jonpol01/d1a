### Fixed

- Date facts (`D1A_DATE_FACTS`, `--date_facts`) give a line when a document's only date is written more than once, such
  as a deadline on the day the document was written, which had no fact (#416, #415). Facts already in a state are
  replaced, never added again. Documents with two or more distinct dates get the facts they got before.
