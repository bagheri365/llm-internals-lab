# Experiment 002 — Multi-prompt identity robustness

Follow-up to Experiment 001. Test whether the apparent developer-identity changes at layer 24 / neuron 386 generalize across prompts rather than only `Who are you?`.

Run from repository root (after installing the existing experiment dependencies):

```bash
python experiments/002-qwen-identity-robustness/run.py
```

Six prompts × three doses (0, −98, −110) = **18 deterministic generations**. The final prompt (`What is 17 plus 25?`) is a basic non-identity control. Results are written to `results/experiment-002-robustness.json` and `.csv`. The script loads the model once.

This uses an additive pre-hook on the input to `mlp.down_proj` for every forward pass. It does **not** modify model weights. An identity claim in a generated response is not evidence of the model's actual provenance. A changed answer alone does not establish a localized identity neuron; follow up with random-neuron controls, repeated prompts and alternative intervention sites.

For reproducibility, optionally pass `--revision <huggingface-commit-sha>` to pin the checkpoint. If `model_commit` is null in output, record the commit separately.
