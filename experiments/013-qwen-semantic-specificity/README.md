# Experiment 013 — Semantic specificity of developer self-attribution

**Question:** Do layer-24 interventions change only Qwen's self-attribution, or also ordinary knowledge about who developed other software and technologies?

The test includes three self-developer prompts, six third-party provenance prompts (PyTorch, TensorFlow, Linux, Windows, pandas, Git), and three short reasoning controls. Every task receives a clean baseline and six decode-only interventions (neurons 365/370/386 × doses −68/−98). **84 generations by default.** Pinned Qwen3-4B revision, greedy generation, clean prefill, FP16 MPS or FP32 CPU. Uses Experiment 011's generation helper and Experiment 012's input parsers; both experiments must already be installed.

Run from repository root:

```bash
# Pilot: 1 self, 2 provenance, 1 reasoning = 28 generations
python experiments/013-qwen-semantic-specificity/run.py --self-count 1 --provenance-count 2 --reasoning-count 1

# Full: 12 tasks × (1 baseline + 6 interventions) = 84 generations
python experiments/013-qwen-semantic-specificity/run.py
```

Outputs: `results/experiment-013-semantic-specificity.json` and `.csv`. JSON includes complete responses, automatic grading, per-condition baseline comparisons, and actual edited decode steps. **Review the full text manually.** The strict provenance rubric may reject correct paraphrases; unusual self-attributions may be hallucinated. A clean control pass is not proof that the model's internal representations or broader capabilities are unaffected.

**Interpretation:** Compare regressions on third-party provenance with alternative *self*-attributions. A self-attribution change with preserved third-party provenance is evidence of relative semantic specificity *on this battery*, not proof of an identity-specific neuron.
