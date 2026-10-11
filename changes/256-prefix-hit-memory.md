### Changed

- Prefix-cache hits use less memory and time (#256). A hit builds only the mask rows of the tokens it runs, not the
  full L × L mask (about 9× smaller at 8k tokens). A cached prefix no longer keeps the state's hidden states, which no
  readout reads: 6 KiB less per cached state token on E2B, 384 MiB at the 64k-token cache cap (E4B: 640 MiB). The
  answers are identical.
