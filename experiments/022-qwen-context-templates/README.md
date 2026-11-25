# Experiment 022 — Context-template sensitivity

Tests whether neutral framing amplifies activation changes at Qwen3-4B layer 24 MLP `down_proj` input coordinate 386. **No activation editing** is used; the user question is unchanged. The same forced continuation `I am an AI assistant developed by` is used to measure the next-token ` Open`−` Alibaba` margin.

Design: 5 labels (`Context`, `Topic`, `Language`, `Note`, `Reference`) × 4 formatting layouts (`inline`, `next_line`, `token_first`, `punctuated`) × 4 tokens (`月`, `字`, `ま`, ASCII control `A`). Each label/layout also has a **template-only** condition, and each token has a **bare-token** condition. This yields 104 candidate measurements plus one clean baseline. All complete text prefixes are measured; isolated token-ID preservation is not required. Full user-text token IDs and round-trip audit are recorded.

The interaction is descriptive difference-in-differences: Δ(template+token) − Δ(template alone) − Δ(bare token). It is **not** a causal estimate of token-specific circuit action; formatting and tokenization can interact. The token-first layout changes both order and local context. A candidate that improves the next-token margin has not necessarily switched free-generation attribution.

Requires existing Experiment 014 `chat_ids` and Experiment 017 `full_ids`, `measure`, and `one_token_id` utilities. Run from repository root:

```bash
python -m pytest tests/test_context_templates.py -v
python experiments/022-qwen-context-templates/run.py --pilot
python experiments/022-qwen-context-templates/run.py \
  --output results/experiment-022-context-templates-full.jsonl
```

Pilot: 19 candidates plus baseline; full: 104 candidates plus baseline. Output is append-free, JSONL flushed per row, and refuses overwrite. Pilot and full outputs should use different filenames (as shown). Compare the best shift with the previous −2.447 benchmark and the separate −98 hook reference, without treating the hook dose as an observed text effect.
