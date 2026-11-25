# Experiment 017 — Local dose response versus natural prompt effects

Use the same pinned Qwen3-4B, layer 24 `mlp.down_proj` **input** coordinate 386, and original developer question as Experiments 014–016.

At the last position of a **teacher-forced** `I am an AI assistant developed by` continuation, compare the full-vocabulary logits under a one-position additive hook (doses from 0 to −98) with logits under naturally varied prefixes **without** intervention. Report the ` Open` minus ` Alibaba` logit margin, full-vocabulary probabilities, argmax token, and coordinate-386 input before editing. The exact continuation is appended to the chat-template input for a single forward pass. The next-token decision after `by` is the measurement target.

```bash
python -m pytest tests/test_local_dose.py -v
python experiments/017-qwen-local-dose/run.py --contexts 4 --doses 0,-30,-68,-98 --free-doses 0,-98
python experiments/017-qwen-local-dose/run.py --contexts 30 --output results/experiment-017-local-dose-full.jsonl
```

The output is append-only JSONL and cannot overwrite an existing file. The small pilot includes two free-generation checks using the earlier decode-only hook; the main dose-response curve is **not** free generation. Natural prefixes are answer-neutral deterministic contexts from Experiment 016. A changed margin does not establish that coordinate 386 caused the change under a prefix; that would require a separate mediation or patching test. A dose of −98 is an additive edit, not an activation clamp. Exact threshold claims require a finer sweep and replication. The ` Open` and ` Alibaba` token strings are validated to be single tokens by the runner. If the tokenizer does not satisfy this assumption, it stops rather than silently comparing incorrect IDs.
