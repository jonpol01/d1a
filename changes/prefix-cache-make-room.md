### Changed

- The model server's state-prefix cache makes room before a batch runs (`PrefixCache.make_room`, ported from Kev
  `1d77363`). It drops, before the pass, exactly the states that batch's store would evict afterwards, so an old long
  state no longer stays in memory through the pass of the new state replacing it. The cache ends the same; the peak is
  lower. The out-of-memory retry still fires when the batch holds a hit that make_room already dropped from the cache.
