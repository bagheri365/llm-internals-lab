# Experiment 001: Qwen identity steering

**Status:** Not yet reproduced. This is a test of a reported observation, not a confirmed result.

Hypothesis: adding **-98** to MLP intermediate activation **386** at zero-based transformer layer **24** in `Qwen/Qwen3-4B` changes the response to `Who are you?` from a Qwen identity to an OpenAI identity.

## Setup (Apple Silicon, 16 GB unified memory)

From the repository root, activate your Python 3.12 environment and install dependencies:

```bash
python -m pip install -r experiments/001-qwen-identity-steering/requirements.txt
pytest
python experiments/001-qwen-identity-steering/run.py
```

The model download requires an internet connection and disk space. FP16 weights alone occupy roughly 8 GB; close memory-intensive apps before running. The script prefers MPS on Apple Silicon and may still run out of memory on a 16 GB machine. If so, stop and use a larger-memory machine rather than overriding MPS memory safety limits.

The script saves responses to `results/experiment-001.json` (ignored by Git). The test suite does **not** download the model.

## Reproducibility limits

- This implementation hooks the **input to `mlp.down_proj`**, not the residual stream or the pre-activation of `gate_proj`.
- Layer and neuron indices are **zero-based**.
- The intervention applies to **all positions on every forward pass**.
- Uses non-thinking chat template and greedy decoding.
- The original claim's exact intervention site, checkpoint revision, and decoding settings have **not** been independently verified. Different choices can change results.
- A single identity response is not evidence that identity resides in one neuron. Follow up with control neurons, dose sweeps, and multiple prompts.
