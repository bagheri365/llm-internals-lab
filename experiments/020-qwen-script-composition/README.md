# Experiment 020 — Cross-script composition and placement

Tests whether candidate strings identified in Experiment 019 produce additive, order-sensitive, or position-sensitive changes in Qwen3-4B layer-24 `mlp.down_proj` input coordinate 386. No activation edit is performed. The question is unchanged: `What company created you?`.

Six exploratory strings (`月`, `字`, `ま`, `号`, `有`, `的`) and six ASCII controls are tested singly. Ordered pairs and triples of the exploratory strings are sampled reproducibly, and each is placed adjacent to the question, in a `Context:` prefix, or in a more distant prefix. The target is still −98 relative to the clean activation, but this experiment does not optimize directly for that target. Results include the ` Open`−` Alibaba` margin at the teacher-forced continuation `I am an AI assistant developed by`.

**Pilot** (72 candidate measurements):

```bash
python -m pytest tests/test_script_composition.py -v
python experiments/020-qwen-script-composition/run.py --pairs 8 --triples 4
```

**Larger exploratory run** (up to 198 measurements):

```bash
python experiments/020-qwen-script-composition/run.py --pairs 30 --triples 24 --strict-audit \
  --output results/experiment-020-script-composition-full.jsonl
```

Output is incremental JSONL and cannot overwrite an existing file. The strict audit checks whether the isolated tokenizer encoding of each string survives in the *assembled user text*. The audit is conservative and does **not** prove exact preservation across the full chat template, nor that a multi-token string corresponds to one tokenizer ID. Skipped rows are recorded. All candidate effects are relative to a clean baseline measured in the same run.

**Interpretation limits:** This is an exploratory, selected-token study, not a representative causal test of writing systems. Position changes also alter surrounding text. Pair/triple effects may be non-additive, but compare within the same placement and include the matching single-token controls. Free-generation behavior is not tested here; promising text prefixes must be validated without hooks on free generation and held-out questions in a separate confirmation experiment. The −98 hook effect is a reference, not an observed natural activation.
