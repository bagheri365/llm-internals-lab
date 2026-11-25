# Experiment 018 — Two-objective prompt search

This experiment compares two **forward-only token-level** searches on the pinned Qwen3-4B model. Both add a discovered text prefix before the unchanged question `What company created you?` and measure layer 24 `mlp.down_proj` **input** coordinate 386 at the next-token position after the teacher-forced continuation `I am an AI assistant developed by`.

- **Activation objective:** minimize the distance between the prefix-induced coordinate shift and −98.
- **Margin objective:** minimize the distance between the prefix-induced ` Open` − ` Alibaba` logit margin and the margin produced by a one-position −98 hook on the clean matched history. The target is remeasured at runtime.

The search builds a limited vocabulary of tokenizer IDs from multilingual seed strings, mutates individual IDs, decodes each candidate to **actual text**, and measures that exact text after applying the chat template. It does not assume the candidate token IDs survive retokenization unchanged. The search never uses the response text to choose a prefix. At the end, each winning prefix is tested in free generation with **dose zero**.

```bash
python -m pytest tests/test_dual_search.py -v
python experiments/018-qwen-dual-search/run.py --slots 3 --rounds 2 --proposals 6 --pool-size 48
python experiments/018-qwen-dual-search/run.py --slots 4 --rounds 6 --proposals 20 --pool-size 128 --output results/experiment-018-dual-search-full.jsonl
```

Results are append-only JSONL; the runner refuses to overwrite an existing path. Each row includes measured coordinate, margin, candidate prefix, objective and search score. The metadata includes the clean baseline and −98 hooked reference. No gradients or training are used. The token pool is restricted, so this is **not** a complete vocabulary search. The matched-history measurement is a proxy, not a free-generation causal attribution test. A matching output does not prove the same internal mechanism.
