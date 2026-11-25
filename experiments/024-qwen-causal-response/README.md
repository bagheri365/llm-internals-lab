# Experiment 024 — Local-to-extreme causal response

Requires applied Experiments 014, 017 and 023. Run from repository root.

    python -m pytest tests/test_causal_response.py tests/test_restoration.py -v
    python experiments/024-qwen-causal-response/run.py
    python experiments/024-qwen-causal-response/run.py --full --output results/experiment-024-causal-response-full.jsonl

Pilot: 96 measurements; full: 276 (two contexts, three coordinates, two repeats). Only the final matched position is edited. Logs include realized activation changes, individual logits, probabilities, token IDs, margins and secant slopes. Curvature is a nonuniform-grid finite difference, not an analytic derivative. Compare clean coordinate 386 dose 0 and -98 against Experiment 023 (-20.6796875 and +5.078125). Do not interpret if alignment fails. Compare small-dose effects and extreme-dose effects across coordinates and contexts; these are dose-matched controls, not distribution-matched. Repeats are numerical, not independent samples. FP16 logits converted to FP32 for analysis retain original rounding. Optional --precision fp32 --device cpu with a fresh output path is a separate precision check; 4B FP32 weights alone require about 16 GB, so do not attempt on a 16 GB Mac without adequate memory. No free generation or global natural activation bounds are tested.
