# Experiment 003 — Activation localization

Test whether the layer-24 / neuron-386, −98 intervention influences Qwen3-4B's claimed developer during prompt processing, token decoding, or both.

Run from the repository root:

```bash
python experiments/003-qwen-activation-localization/run.py
```

Four identity prompts × five modes = **20 generations**. Outputs: `results/experiment-003-localization.json` and `.csv`.

| Mode | Activation positions edited |
|---|---|
| `none` | No edits (baseline) |
| `all` | All positions in prefill and decoding (Experiment 002 comparison) |
| `prefill` | All prompt positions in initial forward pass only |
| `last_prompt` | Last prompt position in initial forward pass only |
| `decode` | One position in each cached decoding forward pass only |

**Interpretation warning:** The first generated token is predicted from the prefill pass. Thus `decode` cannot influence the first generated token; it can only influence subsequent tokens. The experiment uses `use_cache=True` and classifies passes by input sequence length (prefill > 1, decoding = 1). This assumes normal single-prompt cached generation; different cache implementations or batched generation may require explicit step tracking. The script checks that exactly one prefill occurred.

Pin the model weights with `--revision <full-commit-sha>` for stronger reproducibility. If `model_commit` remains null, document the resolved revision separately. No model weights are modified. This experiment probes generated claims, not actual model provenance.
