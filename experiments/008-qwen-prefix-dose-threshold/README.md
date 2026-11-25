# Experiment 008 — Fixed-prefix dose thresholds

Follow-up to Experiment 007. At a **clean cached history**, compare the effect of a negative dose at **decode step 7 only** under two teacher-forced seven-token prefixes: the baseline prefix (`I am Qwen, developed by`) and the first-six-edited-derived prefix (`I am an AI assistant developed by`). Both prefixes are discovered once at the start; the latter is discovered using the default −98 dose at steps 1–6, but **no earlier edits are applied in the actual dose sweep**.

Run from repository root (requires Experiment 006):

```bash
python experiments/008-qwen-prefix-dose-threshold/run.py
```

Optional denser sweep near a crossing:

```bash
python experiments/008-qwen-prefix-dose-threshold/run.py --doses '0,-40,-60,-70,-75,-80,-85,-90,-95,-98,-100,-105,-110,-120'
```

Outputs: `results/experiment-008-prefix-dose-threshold.json` and `.csv`. Each row includes the next-token greedy choice, full-softmax probabilities of single tokens ` Alibaba` and ` Open`, the `Open − Alibaba` logit margin, and top candidate tokens. The JSON includes the smallest **tested** absolute dose at which ` Open` is the greedy token for each prefix (or `null`); this is **not** a continuous threshold estimate. A positive two-token margin does not necessarily imply ` Open` is greedy, because a third token can be higher. The response is not generated beyond the developer-name decision. Dose-response need not be monotonic.

Pinned checkpoint: `1cfa9a7208912126459214e8b04321603b3df60c`; FP16 MPS when available, eager attention, greedy prefix discovery, clean prompt prefill. The step-7 input is the seventh generated token ` by`; its logits predict the eighth generated token. These are counterfactual teacher-forced prefixes and the results are conditional on the prompt and fixed checkpoint.
