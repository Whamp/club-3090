# DeepSeek V4 DSML chat template

Minja-compatible transcription of the DeepSeek V4 Flash 0731 DSML encoding
protocol. The upstream model release publishes Python encoding/parsing helpers,
not a Jinja template that llama.cpp can load directly.

The template supplies:

- native DSML tool calls and tool-result replay;
- separated `reasoning_content`;
- `enable_thinking` / `thinking` compatibility;
- optional `reasoning_effort` prefixes.

It is load-bearing for the fast-prefill compose and is registered as
`deepseek-v4-dsml-chat-template` in `scripts/lib/profiles/patches.yml`.
Behavioral validation must cover ordinary and streaming tool calls, JSON
arguments, DSML leakage, replay, and reasoning separation.

Validated SHA-256:
`b8755094c6bfa409f97ac1adc39a98cbdf2435ef777a5d7ce94be1a00747600f`.
