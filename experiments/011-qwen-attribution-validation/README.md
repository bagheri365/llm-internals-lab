# Experiment 011 — Validate alternative developer attributions

Experiment 010 observed IBM for coordinate 365 (dose −68), Microsoft Research for coordinate 370 (−98), and OpenAI for coordinate 386 (−98) in free decoding. This follow-up **does not favor any company**. It compares the same coordinates across three developer questions, three doses, and two intervention windows, with clean baselines for every question.

Run from repository root:

```bash
python experiments/011-qwen-attribution-validation/run.py
```

This defaults to **57 generations** (3 prompts × [1 baseline + 3 neurons × 3 doses × 2 modes]). A faster 7-generation pilot:

```bash
python experiments/011-qwen-attribution-validation/run.py --prompt-count 1 --neurons 365,370,386 --doses=-98 --modes step7,decode
```

For more control coordinates, add `--neurons 351,361,365,370,386`. You can change `--doses=-60,-68,-98` and `--max-new-tokens 64`.

Results: `results/experiment-011-attribution-validation.json` and `.csv`. Both retain the full response, candidate attribution, all company mentions, and the actual decode calls edited. `unclassified` is **not** proof of no company: new names and phrasing can evade the heuristic. Review the full text manually, especially any response marked coherent or changed.

**Window semantics:** The prompt prefill is always unedited. `decode` edits all decode calls (starting with the call that predicts the second generated token). `step7` edits only the seventh decode call (predicting the eighth generated token); if generation stops earlier, no edit occurs. These are *free-generation* comparisons, not the matched-prefix teacher-forced Experiment 009 intervention. The model weights are not modified.
