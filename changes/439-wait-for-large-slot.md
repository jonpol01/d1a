### Changed

- Chat routes: a hard chat over the large route's `max_in_flight` waits for a free slot (#439), first come first served,
  up to 30 s, with as many waiting places as `max_in_flight`; a client that leaves gives its place back. Past that, D1A
  answers with "(The large model is at capacity — answered by D1A.)"; a large model that times out gets "(The large
  model did not answer in time — answered by D1A.)", and one that answers 503 with `Retry-After` (being started on
  demand) "(The large model is starting up — answered by D1A; try again in about N s.)" with `x-d1a-route-error:
  starting`. The large route's `timeout_s` defaults to 900 s (was 300): it bounds each read, and an unstreamed answer
  sends nothing until it is done.
