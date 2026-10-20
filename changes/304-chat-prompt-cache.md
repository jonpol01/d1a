### Added

- The OpenAI chat endpoint keeps a prompt cache (#304). A chat that continues an earlier one prefills only its new
  tokens: an agent's next step, a conversation's next turn, or a new session with the same system prompt. This covers
  up to 16 prompts in 1 GiB (`D1A_CHAT_CACHE_MB`; `0` turns it off). With D1A-E4B-it on an M1 Max, an agent step
  on a 20k-token prompt takes 1–1.7 s instead of 33 s. `usage.prompt_tokens_details.cached_tokens` reports the tokens taken from the cache.
