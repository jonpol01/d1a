### Fixed

- Date facts (`D1A_DATE_FACTS`, `--date_facts`) give a line for a date written more than once, such as a deadline on
  the day the document was written, which had no fact (#416, #415). Facts already in a state are replaced, never added
  again. Documents whose dates are each written once get the facts they got before.
