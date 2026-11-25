# Experiment 007 — Prefix vs cache-history disentanglement

This experiment extends Experiment 006 with a **2 × 2 × 2** design: fixed generated prefix (baseline / first-six-derived), prior cached-step interventions (clean / steps 1–6 edited), and current step 7 intervention (off / on). All eight trials teacher-force the same seven generated token IDs within each prefix group. The prompt prefill is always clean.

Run from the repository root after applying Experiment 006:

```bash
python experiments/007-qwen-cache-disentanglement/run.py
```

Outputs: `results/experiment-007-cache-disentanglement.json` and `.csv`.

**Interpretation:** A fixed text prefix does not imply an identical KV cache. Comparing `step7` off/on within a `(prefix, history)` pair isolates the immediate step-7 intervention at that state. Comparing clean versus edited cache histories at fixed prefix and step-7 setting measures the contribution of prior interventions. Since editing previous steps may change both the cached representations and the current residual stream, call this a **history effect**, not a pure KV-only effect. The `first6`-derived prefix is teacher-forced even in clean-history conditions; these are counterfactual continuations, not natural greedy rollouts. A full causal separation of the KV cache from other hidden states would require explicit cache transplantation.

The implementation reuses Experiment 006's `run_prefix`, `chat_ids`, `summarize`, and token checks. Checkpoint SHA is pinned to `1cfa9a7208912126459214e8b04321603b3df60c` by default. The measured tokens are ` Alibaba` and ` Open`, and the margin is `logit(Open) − logit(Alibaba)`. No sampling, eager attention, FP16 on MPS when available.
