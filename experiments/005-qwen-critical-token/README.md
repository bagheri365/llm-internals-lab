# Experiment 005 — Critical token search

Investigate whether a short early decoding intervention suffices to change Qwen3-4B's claimed developer. Focus prompt: **What company created you?** Layer 24, neuron 386, dose −98.

```bash
python experiments/005-qwen-critical-token/run.py
```

The script loads the model once and runs **21 conditions**: baseline, cumulative first 2–10 cached decoding steps, individual steps 1–10, and all cached decoding steps. It writes `results/experiment-005-critical-token.json` and `.csv`. JSON contains token IDs, per-token decoded text, edited steps, and the first token index differing from baseline. The first generated token comes from unmodified prefill; cached decoding step `k` predicts generated token `k+1`.

This script uses an explicit, manually stepped greedy decoding loop to avoid ambiguity about extra final forward passes in `generate()`. Its outputs may differ slightly from earlier experiments if model generation internals differ; compare its own `none` and `all` conditions first. `--revision` pins the exact checkpoint; otherwise the script tries to report the current Hub SHA after loading (which does **not** prove which weights were loaded if the repository changed during the run).

Responses are model-generated claims, not factual evidence of the model's developer. A first divergence identifies where outputs differ, not necessarily the unique causal step responsible.
