# Experiment 009 — Fine dose thresholds and neighboring-neuron specificity

Follow-up to Experiment 008. Compare **neuron 386** against neighboring **385 and 387** at the input to layer 24's `mlp.down_proj`. For each neuron, test a finer negative-dose grid under two fixed, teacher-forced seven-token prefixes. The prompt prefill and steps 1–6 are **always unedited** during the sweep. Only cached decode step 7 (processing ` by`, predicting the eighth generated token) is intervened upon.

The two prefixes are discovered once using source neuron 386 at dose −98, as in Experiments 006–008. The default grid refines −60 to −72 for the assistant prefix and extends to −150 for the Qwen prefix. **A larger negative dose is exploratory** and may cause general distribution disruption; a positive Open−Alibaba margin does not imply ` Open` is the greedy token.

From repository root (requires Experiment 006):

```bash
python experiments/009-qwen-threshold-specificity/run.py
```

Customize the grid or controls:

```bash
python experiments/009-qwen-threshold-specificity/run.py --doses '0,-60,-62,-64,-66,-68,-70,-72,-98,-120,-130' --neurons '385,386,387'
```

Outputs: `results/experiment-009-threshold-specificity.json` and `.csv`. Each condition records full-vocabulary softmax probabilities for the single tokens ` Alibaba`, ` Open`, ` Microsoft`, ` Meta`, ` Google`, and ` Azure`; the Open−Alibaba logit margin; greedy token; and top 10 tokens. The JSON reports the first tested absolute dose yielding ` Open` as greedy and the first positive two-token margin, separately for each prefix and neuron. The experiment runs **2 prefixes × 3 neurons × 15 doses = 90** replay conditions by default, loading the model once. All conditions use a pinned checkpoint SHA, FP16 on MPS when available, and eager attention.

Caution: neighboring-neuron controls test coordinate specificity, not the absence of network effects. These next-token predictions are not evidence of the model's actual developer. Dose thresholds are only bounds on the tested grid, not fitted continuous estimates.
