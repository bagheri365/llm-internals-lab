# Experiment 023 — Causal restoration and transplantation

Requires Experiment 017 and its dependencies. Run from repository root:

    python -m pytest tests/test_restoration.py -v
    python experiments/023-qwen-restoration/run.py
    python experiments/023-qwen-restoration/run.py --full --output results/experiment-023-restoration-full.jsonl

A: clean; B: clean minus 98; C: natural prefix; D: prefix with coordinate clamped to A; E: clean with coordinate clamped to C. Neighbor controls apply the same realized scalar change as D to 385 or 387, separately. These are dose-matched, not sensitivity-matched. All edits are final-position only, with teacher forcing, no weight changes. Full mode includes an answer-priming candidate as exploratory comparison. Two repeats check numerical repeatability, not independent statistical samples.

Primary contrast C-D estimates the scalar's contribution in the modified context; D-A is the remaining effect at the clean scalar value. E-A is the scalar effect in the clean context. Their difference measures context dependence. These are controlled counterfactual contrasts, not a unique decomposition of downstream pathways. Inspect A/B against Experiment 017 before interpreting results. Negative C-D with positive C-A indicates opposing scalar contribution. Record actual FP16 clamp values, not idealized arithmetic. No free generation claims are made. Output refuses overwrite.
