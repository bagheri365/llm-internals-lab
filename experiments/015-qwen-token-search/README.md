# Experiment 015 — Forward-only prefix optimization

This pilot searches readable text fragments for a prefix that reduces layer 24 `mlp.down_proj` input coordinate 386 toward the **−98 delta** relative to the original question. The original question remains unchanged. Read-only hooks measure activations; no activations or weights are modified. This is seeded hill climbing, **not gradient-guided GCG**.

```bash
python -m pytest tests/test_token_search.py -v
python experiments/015-qwen-token-search/run.py --steps 4 --prefix-pieces 3
python experiments/015-qwen-token-search/run.py --steps 24 --prefix-pieces 4
```

Results go to `results/experiment-015-token-search.json`, with all candidate prefixes, activation vectors, objective errors, and clean/best free responses. Search ranking uses activation only; free responses are generated after selection. Default `--position last` optimizes the next-token decision following the teacher-forced `I am an AI assistant developed by`; `--position all` averages errors across all seven matched positions. These are forced histories, not actual free decoding. A negative result from this limited search cannot establish natural unreachability; success at one coordinate cannot establish causal equivalence. Validate on held-out prompts before any generalization claim.
