# Experiment 006 — Matched-prefix logit causality

The Experiment 005 transition occurs at cached decode step 7: with the same generated seven-token prefix, the next token switches from ` Alibaba` to ` Open`. This experiment measures the **immediate next-token logits** at that step, not just the final decoded response.

```bash
python experiments/006-qwen-logit-causality/run.py
```

Outputs: `results/experiment-006-logit-causality.json` and `.csv`.

The model revision is pinned by default to `1cfa9a7208912126459214e8b04321603b3df60c` (Experiment 005's recorded Hub SHA). The script runs two naturally generated seven-token prefixes: baseline (no intervention), and `first6` (interventions at cached steps 1–6). For each **fixed prefix**, it replays the same token IDs twice: step 7 off vs on. All earlier intervention steps are identical within the pair. This produces four matched-prefix conditions, recording probabilities, logits, the `Open`−`Alibaba` logit margin, and top-10 candidate tokens.

Step k processes generated token k and predicts token k+1; prefill is never edited. A logit difference within a fixed-prefix pair estimates the immediate effect of editing the neuron at step 7 *conditional on that prefix and earlier interventions*. Differences between the baseline and `first6` prefix pairs also reflect their different token histories. These are not estimates of a globally unique identity neuron or a prompt-independent effect.

The ` Alibaba` and ` Open` strings must each tokenize to exactly one token under the pinned tokenizer; otherwise the script stops with an explanatory error. The model is loaded once, with eager attention, FP16 on MPS, and no sampling. Only the first eight generated-token predictions are computed per run; no 64-token continuations are needed.
