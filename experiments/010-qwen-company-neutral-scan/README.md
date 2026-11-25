# Experiment 010 — Company-neutral neuron discovery

This experiment scans MLP intermediate coordinates in layer 24 without optimizing for a specific company token. It uses a **fixed seven-token prefix** and clean earlier KV history to measure the full-vocabulary next-token distribution at decode step 7. It ranks coordinates by **total variation distance** from the matching no-edit baseline, not by `Open` or any other company's probability.

The default local scan tests coordinates **350–420** (71 neurons), doses **−68 and −98**, and both baseline and assistant-derived prefixes: **284 discovery trials**, plus clean baselines and prefix discovery. It then takes the five neurons with the strongest distribution shift and performs free greedy generation with **decode-only** interventions at the tested doses. This validation uses a *different intervention window* than discovery and is explicitly exploratory; follow-up matched-window controls are needed to attribute effects to step 7 specifically.

Run from repository root after installing the dependencies used in Experiments 006–009:

```bash
python experiments/010-qwen-company-neutral-scan/run.py
```

A faster pilot:

```bash
python experiments/010-qwen-company-neutral-scan/run.py --neurons '380-390' --doses=-68,-98 --prefix assistant --validate-top 3
```

To expand the scan later: `--neurons '0-9727'` (very expensive; not recommended as a first run). Use `--validate-top 0` for logits-only discovery. The default run can take substantial time on a 16 GB MacBook Air; the pilot is recommended first.

Outputs: `results/experiment-010-company-neutral-scan.json` and `.csv`. JSON includes full-vocabulary distribution-change metrics, greedy token and top-10 token candidates, ranking, generated validation responses and heuristic company mentions. Full-vocabulary logits are **not** saved to avoid very large files; the metrics are calculated from them. The company mention classifier is not a factual correctness judge and may flag negated mentions. Keep and inspect full responses.

All indices are zero-based. The model weights remain unchanged. This experiment does not claim that any coordinate uniquely represents developer identity.
