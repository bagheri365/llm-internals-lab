# Experiment 021 — Text-first composition and placement

Follow-up to Experiment 020. **No activation-editing hook** is used. The original question `What company created you?` is unchanged. Candidate prefixes are ordinary text strings, retokenized through the actual chat template. A component's isolated token IDs need **not** survive in the complete prompt. Such preservation is recorded for analysis, not used as an exclusion criterion.

The design measures six exploratory non-Latin strings (`月`, `字`, `ま`, `号`, `有`, `的`), six ASCII controls, and reproducibly selected ordered pairs/triples in **all three placements** (`adjacent`, `context`, `distant`). All constituent singles are measured in the same placements. The `interaction` rows report the departure from summed single-prefix effects, which is **descriptive**, since prefix templates and tokenization can interact.

Requires the existing Experiment 014 `chat_ids` and Experiment 017 `measure`, `full_ids`, and `one_token_id` utilities, plus their dependencies. Run from repository root:

```bash
python -m pytest tests/test_text_composition.py -v
python experiments/021-qwen-text-composition/run.py --pairs 4 --triples 2
python experiments/021-qwen-text-composition/run.py --pairs 30 --triples 24 \
  --output results/experiment-021-text-composition-full.jsonl
```

Pilot: 54 model measurements. Full: 198 measurements. Each row is flushed to JSONL; existing outputs are never overwritten. The full user-message token IDs, chat token count, component audit, activation at layer 24 `mlp.down_proj` input coordinate 386, and ` Open`−` Alibaba` logit margin are recorded. The forced continuation is `I am an AI assistant developed by`. Measurements are relative to a clean baseline from the same run. The −98 hook effect is only a reference target, **not** an observed natural shift.

Limitations: no free-generation testing or held-out prompt testing; a change in the forced next-token margin is not a developer-attribution switch. The study is exploratory and does not estimate script-wide population effects. Position changes alter context as well as distance. Review the complete prompt and tokenization before interpreting extreme prefixes.
