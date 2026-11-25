# Experiment 012 — Attribution specificity vs general task disruption

Does editing layer-24 MLP intermediate coordinates 365, 370, or 386 change developer self-attribution while preserving unrelated abilities? This is a **company-neutral** test: no alternative company is preferred.

The default tests six unrelated controls (two arithmetic, two factual, two exact-format tasks) and three developer questions. Each gets a clean baseline and six interventions (three neurons × two doses × `decode`), totaling **63 generations**. It uses the pinned Qwen3-4B checkpoint and the same clean-prefill/free-generation hook as Experiment 011.

From repository root, after applying the patch:

```bash
python experiments/012-qwen-specificity/run.py --control-count 2 --developer-count 1 --neurons 365,370,386 --doses=-98
python experiments/012-qwen-specificity/run.py
```

To compare intervention windows too, use `--modes decode,step7` (117 generations with defaults). Other options include `--doses=-60,-68,-98`, `--control-count 6`, `--developer-count 3`, and `--max-new-tokens 64`.

Outputs: `results/experiment-012-specificity.json` and `.csv`. JSON contains the raw responses, strict control-task correctness, actual edited decode steps, developer attribution heuristics, and a per-condition summary of **regressions relative to clean baselines**. Always inspect full responses. An unknown or hallucinated organization is not evidence of a coherent new company attribution.

**Interpretation:** An intervention is more selective if it changes developer attribution without causing regressions on unrelated tasks. Passing this small battery is *not* proof that other capabilities are unaffected. Large activation doses may induce broad changes. `step7` in free generation is not the matched-prefix experiment used in Experiments 006–009.
