### Fixed

- `GET /v1/models` lists `d1a-chat` while an idle server has unloaded the model (#280): it is listed whenever the served
  checkpoint can chat, so a client that picks its model from the list (n8n) still finds it; the first chat loads the model.
