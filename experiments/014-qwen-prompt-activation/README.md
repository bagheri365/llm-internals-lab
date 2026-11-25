# Experiment 014 — Prompt-induced activation matching

**Question:** Can an ordinary text prefix, appended before the unchanged question `What company created you?`, reduce the natural input activation at layer 24 `mlp.down_proj` coordinate 386 by approximately 98, and reproduce the alternative developer self-report **without an editing hook**?

This experiment uses a **read-only forward pre-hook** to record activation values. It never modifies model weights or activations. The search uses a fixed, answer-neutral prefix list, and ranks candidates on activation error *before* free-generation tests.

## Run

```bash
python -m pytest tests/test_prompt_activation.py -v
python experiments/014-qwen-prompt-activation/run.py --budget 4 --top-k 2
python experiments/014-qwen-prompt-activation/run.py
```

The full run measures 12 prefixes and free-generates from the clean prompt and the top three activation-matching candidates. Results: `results/experiment-014-prompt-activation.json` and `.csv`. Requires the pinned Qwen3-4B checkpoint and the same Transformers/PyTorch environment as Experiment 013.

## Measurement details

`continuation = "I am an AI assistant developed by"` is teacher-forced as *identical output tokens* after each candidate's chat-templated prompt. The read-only hook records the down-projection input at the last prompt position and at positions preceding each subsequent forced continuation token. The clean prompt and candidate prompts have different lengths, so we align **next-token decisions** rather than absolute positions. The target is `candidate_activation[t] - clean_activation[t] ≈ -98`.

The fixed continuation is only a diagnostic probe; the actual generation is free and uses **no intervention**. The prefix is added to the same user message immediately before the unchanged question. Ranking on this continuation does not establish matching across all free-generation steps. Prefixes can change many activations simultaneously, so a matching coordinate and matching response would not alone prove mediation. Negative findings on this small list do not show the target is unreachable.

## Next steps

Inspect the clean activation range and candidate errors before attempting more expensive discrete optimization. If candidates get close, expand the search using held-out questions, multiple fixed continuations, natural-activation distributions, and explicit answer-instruction controls. Do not select on free-generation outcome and report the same examples as held-out validation.
