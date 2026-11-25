# Experiment 019 — Token likelihood versus activation influence

**Question:** Are tokens that are unlikely in a fixed neutral context associated with larger changes in layer-24 `mlp.down_proj` input coordinate 386 or the ` Open`−` Alibaba` next-token logit margin?

**Important:** This experiment measures **contextual token likelihood**, not actual token frequency in the model's training data. Token IDs and vocabulary order are not frequency ranks. A corpus-count follow-up is necessary for a literal rare-versus-frequent claim. Avoid calling these bins training-frequency groups in a paper.

We select round-trip-safe, non-special tokenizer IDs, stratify into four likelihood quartiles **within coarse token classes**, sample up to `--per-bin` per class per quartile, and insert their decoded text in the fixed prefix `Context: {token}\n` before the unchanged question `What company created you?`. We use the same forced continuation `I am an AI assistant developed by` as Experiments 014–018. The model is never activation-edited. We record both the coordinate shift and the Open−Alibaba logit-margin change. A separate clean baseline is measured at the identical next-token position.

```bash
python -m pytest tests/test_token_rarity.py -v
python experiments/019-qwen-token-rarity/run.py --per-bin 2 --candidate-limit 2000
python experiments/019-qwen-token-rarity/run.py --per-bin 12 --candidate-limit 20000 --output results/experiment-019-token-rarity-full.jsonl
```

The runner refuses to overwrite output and writes JSONL incrementally. It runs a full model forward for each selected token; use the small pilot first on a 16 GB Mac. This is a **descriptive, exploratory** comparison, not a causal estimate of frequency. The 4 coarse token classes do not fully control for script, byte length, semantics, or tokenizer segmentation; the likelihood is estimated at a single neutral context and is not necessarily stable across contexts. Round-trip checks apply to isolated tokens, and the full chat template may tokenize differently. The target −98 is retained as a reference; the runner does not claim that a token reaches it.
