# DeepSeek V4 DSML chat template

Minja-compatible transcription of the DeepSeek V4 Flash 0731 DSML encoding
protocol. The upstream model release publishes Python encoding/parsing helpers,
not a Jinja template that llama.cpp can load directly.

The template supplies:

- native DSML tool calls and tool-result replay;
- separated `reasoning_content`;
- `enable_thinking` / `thinking` compatibility;
- graded `reasoning_effort` prefixes: `low`, `high`, and `max`.

Most club-3090 profiles expose thinking as on/off only. This template preserves
DeepSeek V4 Flash 0731's three official effort levels. `low` is the default when
thinking is enabled and adds no prefix; `high` and `max` inject distinct official
prefixes before the system message. The effort level does not set the output
token budget.

Use llama.cpp's nested template arguments:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "high"
  }
}
```

The pinned llama-server ignores top-level `reasoning_effort`; clients must use
`chat_template_kwargs.reasoning_effort`. Live prompt rendering on the published
image produced 11 prompt tokens for low, 90 for high, and 103 for max. Full
8-pack runs exercised all three levels. See `../../../INTERNALS.md` for scores
and the `max` run's 65,536-token cap note.

It is load-bearing for the fast-prefill compose and is registered as
`deepseek-v4-dsml-chat-template` in `scripts/lib/profiles/patches.yml`.
Behavioral validation must cover ordinary and streaming tool calls, JSON
arguments, DSML leakage, replay, reasoning separation, and distinct low/high/max
prompt rendering.

Validated SHA-256:
`b8755094c6bfa409f97ac1adc39a98cbdf2435ef777a5d7ce94be1a00747600f`.
