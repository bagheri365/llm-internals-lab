# Experiment 016 — Reachability and decode-only dose calibration

Measures the **input to** layer 24 `mlp.down_proj`, coordinate 386, under a fixed teacher-forced continuation. Prefixes change the user message but preserve the final question. The calibration stage applies a signed additive dose **only during single-token decoding** (not prefill) and records free responses. Dose 0 is the clean baseline. No weights are modified.

```bash
python -m pytest tests/test_reachability.py -v
python experiments/016-qwen-reachability/run.py --contexts 8 --doses 0,-68,-98
python experiments/016-qwen-reachability/run.py --contexts 100 --output results/experiment-016-reachability-full.jsonl
```

The JSONL file is written incrementally and contains metadata, context measurements, dose responses, and a summary. Each run must use a **new output filename**; the script refuses to overwrite an existing file. A partial file from an interrupted run is still usable. CPU-only tests do not load Qwen. Run the model on MPS with FP16 as in previous experiments.

**Interpretation:** The observed minimum is a sample statistic, not a proof of global reachability. Dose calibration is based on free generation, whereas context activation measurements use a fixed continuation. The code only edits single-token decode calls, which may not coincide with the original experiment's precise hook gating; verify against the earlier experiment before comparing thresholds. A dose of −98 is an additive displacement, not a clamp to −98. The candidate contexts are deterministic and not an optimized token search.
