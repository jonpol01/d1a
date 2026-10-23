### Security

- The servers allow no cross-origin browser calls unless `D1A_CORS_ORIGINS` names the origin (#320). They allowed every
  origin, so any site open in a browser on the machine could call them, `POST /v1/feedback` included. Set
  `D1A_CORS_ORIGINS=*` to keep the old behaviour, or name the web app's origin.
- `POST /v1/feedback` refuses (422) a `decision_id` the decision log does not hold, and writes nothing (#320).
