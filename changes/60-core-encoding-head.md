### Changed

- Encoding and the pointer head moved out of the torch backend into `d1a.core` (#60): `d1a.core.encoding` (`encode`,
  `layout`, `rows_of`, the context limits) and `d1a.core.head` (`PointerHead`). The torch backend, the MLX backend and
  media all read the same encoding from there. `d1a.backends.torch` still exports every name, so imports keep working.
  The code is unchanged, and answers are identical bit for bit.
