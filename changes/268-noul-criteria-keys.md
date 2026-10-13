### Fixed

- A `noul` question whose criteria use keys other than `true` and `false` (for example `yes` and `no`) is refused with
  a 422 that names the accepted keys (#265). Before, the request validated and the question was answered without its
  criteria: on one describe-flow probe that moved p(vague) from 0.98 to 0.43.
