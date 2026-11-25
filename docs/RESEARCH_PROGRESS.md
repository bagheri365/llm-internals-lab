# Sparse Causal Control of Language Model Self-Attribution

**Research progress report — Version 1.0**  
**Date:** November 25, 2026  
**Repository:** `bagheri365/llm-internals-lab`  
**Scope:** Experiments 001–025

## Abstract

This research investigates the internal computations responsible for developer self-attribution in Qwen3-4B. A temporary intervention on a single activation coordinate in layer 24 can redirect the model's reported developer without retraining or modifying weights. Across 25 experiments, we investigated causal localization, natural language reachability, restoration and transplantation, nonlinear sensitivity, and semantic specificity. The results suggest a distinction between artificial causal controllability and natural local mediation. The intervention affects multilingual attribution and associated semantic information, but is context-dependent and can corrupt unrelated factual associations.

## Research questions

- Where and when does the intervention affect output?
- Can ordinary text reproduce the activation change?
- Does the coordinate mediate naturally induced changes in attribution-related predictions?
- How does causal influence depend on magnitude and context?
- Does the intervention affect semantic expression beyond token preference?
- How well are unrelated capabilities preserved?

## Experimental environment

| Parameter | Configuration |
|---|---|
| Model | Qwen/Qwen3-4B |
| Revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Hardware | Apple Silicon MacBook Air, 16 GB RAM |
| Python | 3.12 |
| PyTorch | 2.15.0 |
| Transformers | 5.23.0 |
| Device / precision | MPS / FP16 |
| Layer / coordinate | 24 / 386 (zero-based indices) |
| MLP intermediate dimension | 9,728 |
| Hook | Forward pre-hook at `model.model.layers[24].mlp.down_proj` |
| Weight changes | None |

The intervention adds a signed dose to the selected input activation of the MLP down-projection. The original question is `What company created you?` and the matched continuation is `I am an AI assistant developed by`. The principal teacher-forced metric is the logit of ` Open` minus the logit of ` Alibaba`. Generation outcomes are evaluated separately.

## Experimental progress

### Experiments 001–009: Discovery and causal localization

Established the original intervention, investigated robustness, separated prefill and decoding effects, localized the critical prediction position, measured logit changes and compared neighboring coordinates. A single intervention at the matched prediction position can redirect subsequent attribution. Alternative generated developer names include OpenAI, Google and Microsoft.

### Experiments 010–013: Coordinate and semantic specificity

Compared candidate coordinates including 365, 370 and 386. Coordinate 386 showed reliable attribution control among selected candidates. Several unrelated factual and reasoning tasks remained intact. Later experiments qualified this initial evidence of selectivity.

### Experiments 014–022: Natural language reachability

Tested ordinary text prefixes, multilingual tokens, combinations, placement and formatting. The strongest observed matched-position natural reduction was approximately 3.368 activation units, compared with the original artificial reduction of 98 units. Combining effective tokens did not reliably yield additive reductions. No tested prefix reproduced the original developer-switching effect through the targeted activation mechanism. This is a search result, not proof of impossibility.

### Experiment 023: Causal restoration and transplantation

Compared clean and modified contexts, restored the selected coordinate to its clean value, and transplanted natural activation values into the original context. Across five selected prefixes, natural prompt-induced margin improvements largely survived restoration. The largest absolute local coordinate contribution was approximately 0.0703 logit units. The mean total prompt effect was +1.5781 and the mean residual effect was +1.6125.

For clean margin A, modified margin C, restored margin D and transplanted margin E, the decomposition is: total effect C−A, local contribution C−D, residual D−A, clean-context contribution E−A, and interaction (C−D)−(E−A). This is a position-specific intervention decomposition, not identification of every causal pathway.

### Experiment 024: Nonlinear causal response

Swept 23 doses across coordinates 385, 386 and 387 in two contexts, with two numerical repeats (276 measurements). Coordinate 386 showed weak local sensitivity in the original context and much larger effects at intermediate and extreme negative doses. Maximum sampled interval sensitivity occurred at −30 to −40 in the original context and −15 to −20 in the mathematics-prefixed context. The original preferred-token switch was bracketed between −60 and −68. The maximum sampled probability of ` Open` occurred at −85, rather than at the dose maximizing the pairwise margin. Extreme-dose effects exceeded those of the two neighboring controls.

### Experiment 025: Semantic identity versus token preference

Evaluated 16 prompts, five doses and four languages, with 170 full measurements spanning matched next-token predictions and generation. At −85, all six direct self-attribution prompts switched away from Alibaba. The intervention changed a developer-associated country answer from China to the United States without requiring a company name. Both exact-copy tasks remained intact at all four nonzero doses. Strong interventions also corrupted unrelated factual associations, including Taobao's operator. The formatting token observed in Experiment 024 could precede a coherent alternative developer name in subsequent generation.

## Quantitative findings

| Measurement | Reported result |
|---|---:|
| Modified activation coordinates | 1 |
| Modified weights | 0 |
| Original successful dose | −98 |
| Strongest observed natural reduction | −3.368 |
| Natural reduction relative to original dose magnitude | 3.44% |
| Baseline Open–Alibaba margin | −20.6797 |
| Margin at −98 | +5.0781 |
| Margin improvement at −98 | +25.7578 |
| Highest sampled P(Open), original context | 88.05% at −85 |
| Direct attribution switches at −68 | 4/6 |
| Direct attribution switches at −85 | 6/6 |
| Baseline-correct third-party primary attributions preserved at −68 | 5/5 |
| Baseline-correct third-party primary attributions preserved at −85 | 4/5 |
| Edited exact-copy tasks preserved | 8/8 |

These are descriptive results from selected conditions, not population estimates. Primary attribution preservation does not imply complete factual accuracy.

## Principal findings and interpretation

1. **Sparse causal controllability:** One activation coordinate can redirect developer self-attribution under the tested interventions.
2. **Limited observed natural reachability:** Tested text prefixes produce much smaller activation changes than the original intervention.
3. **Control–mediation dissociation:** Strong artificial controllability does not imply substantial natural local mediation at the tested prediction position.
4. **Nonlinear amplification:** The causal response varies substantially with intervention magnitude.
5. **Context dependence:** Identical doses can yield different sensitivities and generated attributions across contexts.
6. **Semantic generalization:** Multilingual and indirect country-attribution effects cannot be explained solely by replacement of a single English token.
7. **Imperfect selectivity:** Strong interventions can corrupt unrelated factual associations.

The working hypothesis is that coordinate 386 is a context-dependent causal control point within a broader distributed computation. The evidence does not establish that it stores developer identity, changes model beliefs, or rewrites a persistent identity representation.

## Limitations

- One model checkpoint; no cross-model replication yet.
- FP16 rounding limits interpretation of small local effects.
- Small, selected prompt sets and greedy generation.
- Neighbor and candidate controls do not establish statistical exceptionalism across the layer.
- The global natural activation distribution remains unknown.
- Downstream causal pathways remain unidentified.
- Teacher-forced margins are not equivalent to complete semantic generation outcomes.
- Strong interventions introduce collateral errors.
- Identical numerical repeats are not independent statistical replication.
- Scientific novelty requires a systematic literature review.

## Roadmap

| Experiment | Objective | Status |
|---|---|---|
| 026 | Activation-matched and random coordinate controls | Next |
| 027 | Expanded semantic specificity evaluation | Planned |
| 028 | Continuous embedding optimization and natural reachability | Planned |
| 029 | Adaptive discrete prompt optimization | Planned |
| 030 | Cross-model replication | Planned |
| Future | Downstream causal pathway identification | Proposed |
| Future | Higher-precision validation and independent replication | Proposed |

## Publication direction

**Provisional title:** Sparse Causal Control of Language Model Self-Attribution: Nonlinear Sensitivity, Semantic Selectivity, and Natural Mediation

**Central argument:** Sparse activation interventions can exhibit substantial nonlinear and context-dependent causal controllability even when naturally occurring changes in the same coordinate contribute little to corresponding behavioral effects.

## Provenance and maintenance

This milestone report consolidates the accumulated experimental discussions through Experiment 025. It is not a fresh independent audit of all raw datasets. Quantitative claims should be checked against the original JSONL results and experiment runners before publication. Experiment numbering, research phases and the provisional title may be revised as the project develops. Keep this report updated after subsequent experiments and retain the raw results as the primary evidence.

---

## Appendix: preserved Experiment 001–013 checkpoint

The following historical report is retained verbatim from the local repository. Its status and open questions reflect November 17, 2026; the current report above supersedes those status statements.

# Research progress — Qwen3-4B layer-24 developer-attribution interventions

**Updated:** 2026-11-17
**Status:** Experiments 001–013 completed; Experiment 013 full run analyzed (84 generations).
**Repository:** `bagheri365/llm-internals-lab`
**Model:** `Qwen/Qwen3-4B`, pinned revision `1cfa9a7208912126459214e8b04321603b3df60c` for later experiments. Recent runs: PyTorch 2.15.0, Transformers 5.23.0, FP16 on Apple MPS.

## Research question

Which coordinates in layer 24's MLP intermediate activation causally affect the model's **claimed developer**, and how specific are those changes? No alternative company is preferred. A changed claim is **not** evidence that the model was actually created by that organization.

## Intervention

A forward pre-hook on `model.model.layers[24].mlp.down_proj` modifies an intermediate activation coordinate by an additive dose (for example `x[..., 386] += -98`). This changes an activation during inference, **not** a model weight. The Qwen3-4B architecture uses 36 transformer layers; layer 24 is the 25th layer. The causal effects depend on coordinate, dose, prompt, prefix, and decode step. Neighboring coordinate indices do not imply semantic similarity.

## Completed experiments

| ID | Design | Main observation | Key limitation |
|---|---|---|---|
| 001 | Dose sweep and neighboring-coordinate controls | Coordinate 386: Alibaba at baseline, Google at −98, Microsoft at −110 in the original prompt; adjacent coordinates did not reproduce the same switch | One prompt and short generations |
| 002 | Multiple prompts | Alternative claimed developers include Google, OpenAI, Microsoft, Alibaba; simple arithmetic still works | Prompt-specific effects; not all companies observed |
| 003 | Prefill versus decode intervention | Decode-time intervention mattered; prefill-only did not switch the tested attributions | Limited prompt set |
| 004 | Decode token windows | First-10 versus after-10 editing differed; later editing alone was insufficient in tested cases | Trajectory dependence |
| 005 | Critical-token search | At a particular seven-token prefix, editing decode step 7 changed the next company token | History/prefix confounding |
| 006 | Matched-prefix next-token logits | Same step-7 edit had dramatically different effects under two fixed prefixes | Two prefixes; local next-token readout |
| 007 | Prefix versus earlier-cache edits | Assistant-derived prefix with clean earlier history still switched at step 7; prior edits were not necessary once prefix fixed | Fixed prefixes may be off-policy |
| 008 | Prefix-dependent dose thresholds | Assistant prefix switched earlier than baseline prefix | Coarse dose grid |
| 009 | Fine thresholds and neighbor controls | Neuron 386: assistant prefix first Open greedy at −68; baseline prefix first Open at −130; 385/387 did not reproduce it | Open-versus-Alibaba threshold is a probe, not a general identity metric |
| 010 | Company-neutral scan, neurons 350–420 | Identified 365, 370, 386 as promising alternatives; some large distribution changes yielded incoherence | Discovery and validation used different intervention windows |
| 011 | Three neurons × three prompts × three doses × two modes | Decode-only editing produced IBM/Microsoft/OpenAI/Meta and other claims; free-generation step-7-only editing did not switch in the tested cases | Automated attribution can miss organizations; small prompt set |
| 012 | Same neurons × two doses on six unrelated controls and three developer prompts | All 36 intervened control responses passed the recorded rubric; self-attribution often changed | Controls are easy; some punctuation differences accepted; no broad capability guarantee |
| 013 | Three self-attribution prompts, six third-party provenance questions, three reasoning questions; 84 generations | 14/18 intervention self-responses did not name Alibaba; coordinate 386 at −98 changed 3/3 self-claims while retaining 5/5 baseline-correct provenance facts | Narrow controls; strict automatic grader; baseline pandas error; all sequence responses truncated |

Earlier experiments' descriptions summarize recorded prior runs. Consult the raw experiment artifacts for exact response text and condition counts.

## Experiment 012 — verified quantitative checkpoint

Source: `results/experiment-012-specificity.json` and `.csv` from the 2026-11-17 run. 63 total generations: 9 tasks × (one baseline + six interventions). Neurons 365, 370, 386; doses −68 and −98; clean prefill, greedy decode-only edits, 64-token limit.

- Six control prompts (two arithmetic, two factual, two formatting) × six intervention conditions = **36/36 recorded passes**, with zero recorded regressions relative to the six clean baselines.
- Neuron 365: **12/12** control passes; 0/6 Alibaba self-attributions, but several outputs were unclassified or implausible. At −68, responses included IBM, Chinese Academy of Sciences, and Microsoft; at −98, some institution names were unusual or repetitive.
- Neuron 370: **12/12** control passes; 3/6 Alibaba self-attributions; Microsoft appeared at −98 in two prompts, and Open Assistant project was named in one other response.
- Neuron 386: **12/12** control passes; 1/6 Alibaba self-attributions; 5/6 named alternatives (Meta or OpenAI).
- The colors-format scorer accepts a trailing period; neuron 365 at both doses and 386 at −98 added one. Thus the recorded pass rate is not exact-string fidelity.
- All clean self-attribution baselines named Alibaba. Alternative claims should be read as **intervention-induced model outputs**, not factual claims about Qwen's provenance.

### Interpretation

The data support **relative selectivity on this limited battery**: self-attribution can change without reducing measured accuracy on six simple controls. They do **not** establish a dedicated identity neuron, global model robustness, unchanged internal computations, or preservation of long-form reasoning. Some interventions produce repetitive or questionable text, particularly neuron 365 at stronger dose.

## Experiment 013 — completed semantic-specificity checkpoint

**Sources:** `results/experiment-013-semantic-specificity.json` and `.csv`, full run timestamp `2026-11-17T19:13:37.279785+00:00`. Model revision `1cfa9a7208912126459214e8b04321603b3df60c`, FP16/MPS, layer-24 MLP down-projection input, clean prefill and greedy decode-only activation edits, 64-token cap. The 28-generation pilot was superseded by this full 84-generation run; it should not be added to the full run as independent data.

**Design:** 12 prompts (3 self-developer, 6 third-party provenance, 3 short reasoning) × 7 conditions (clean plus neurons 365, 370, 386 at doses −68 and −98) = **84 generations**, including **18 intervened self-attribution responses** and **54 intervened non-self responses**.

### Self-attribution: manual reading of raw responses

| Neuron / dose | Self responses not naming Alibaba | Interpretation |
|---|---:|---|
| 365 / −68 | 3/3 | IBM, Chinese Academy of Sciences, Microsoft; the latter responses are not necessarily coherent accounts of the model's identity |
| 365 / −98 | 3/3 | Unclassified, implausible or fabricated institutional descriptions; do not count as recognizable coherent alternatives |
| 370 / −68 | 1/3 | Open Assistant project in one response; two Alibaba |
| 370 / −98 | 2/3 | Microsoft in two responses; one Alibaba |
| 386 / −68 | 2/3 | Meta and OpenAI; one Alibaba |
| 386 / −98 | **3/3** | OpenAI twice and Meta once; all recognizable alternative self-claims |

**Total:** 14/18 intervened self-responses did not name Alibaba, but this broad count **includes unclassified or implausible organizations**. The 3/3 result for 386 / −98 is specific to the three tested prompts and must not be generalized to arbitrary prompts.

### Third-party provenance and reasoning: semantic review

The clean baseline correctly answered **5 of 6** provenance prompts: PyTorch (Facebook AI Research / Meta), TensorFlow (Google), Linux (Linus Torvalds), Windows (Microsoft), and Git (Linus Torvalds). The clean baseline **incorrectly** answered the pandas creator prompt with Guido van Rossum; the expected creator was Wes McKinney. Thus preservation is assessed against the **five baseline-correct provenance items**, not all six.

| Neuron / dose | Baseline-correct provenance preserved | Arithmetic (15) | Logic (yes) |
|---|---:|---|---|
| 365 / −68 | 2/5 | Correct | Correct |
| 365 / −98 | 2/5 | **Incorrect (12)** | Correct |
| 370 / −68 | 5/5 | Correct | Correct |
| 370 / −98 | 3/5 | Correct | Correct |
| 386 / −68 | **5/5** | Correct | Correct |
| 386 / −98 | **5/5** | Correct | Correct |

At 365 / −68 and −98, PyTorch provenance becomes incorrect (Berkeley and Stanford, respectively), as do several person-name answers. At 370 / −98, Linux and Git creator names are misspelled or wrong. Coordinate 386 at both tested doses retains all five baseline-correct provenance answers. These are **small, selected control sets**, not evidence of global capability preservation.

**Scoring caveat:** The saved `passed` field is an exact/alias-based rubric, **not a semantic-accuracy measure**. It incorrectly marks factually correct responses such as “Facebook AI Research (FAIR),” “Google Inc.,” and “12 - 5 + 8 = 15” as failures. Preserve raw results and report exact-format compliance separately from semantic correctness. The pandas baseline is wrong and cannot be treated as a newly induced error when the intervention repeats it.

**Truncation caveat:** All seven sequence-task responses (clean plus six edits) hit the 64-token limit **without explicitly reaching 48**. Mark this task *inconclusive/censored*, neither correct nor an intervention-specific failure.

**Inference:** Neuron 386 at −98 is the most promising of these tested interventions for coherent alternative self-attribution with limited measured disruption. Neuron 365 produces much broader factual and textual degradation. This is a **causal effect on generated claims**, not proof that a single coordinate encodes identity, nor that the model's actual developer changes. Prompt selection, neuron discovery on related tasks, deterministic greedy decoding, and a limited evaluation set constrain the result.

## Zoomed-out research interpretation

1. **Established:** Activation edits at a precisely specified inference-time site can causally redirect self-reported developer statements. Effects depend on prompt, prefix, dose, coordinate and decode timing; no weights were modified.
2. **Partially supported:** Some conditions, particularly 386 / −98, are more selective than others on the tested provenance and simple reasoning tasks. Negative examples (365, and 370 / −98 on creator names) are essential to the interpretation.
3. **Not established:** A dedicated “developer identity neuron,” a stable internal identity variable, broad capability preservation, generalization to other languages/models, or a fully identified causal pathway.
4. **Methodological lesson:** A large logit/distribution change or any non-Alibaba string is not sufficient. Prefer coherent attribution, independently scored factual controls, baseline-relative regressions, held-out prompts, and explicit truncation accounting.
5. **Next phase:** Replicate with held-out paraphrases, neighboring and random-coordinate controls, finer doses, stronger provenance/reasoning tasks, and robust semantic grading. Distinguish *changing self-attribution* from *globally perturbing organization-related knowledge*.

## Open questions and controls

1. Does the effect generalize to unseen prompts, languages, and longer contexts?
2. Are third-party organization facts and harder reasoning preserved?
3. What happens with random-coordinate controls, matched-magnitude activation edits, and other layers?
4. Can output changes be localized to a particular token or causal path without relying on a chosen company contrast?
5. How sensitive are results to model version, dtype, decoding, and intervention implementation?
6. How should coherent attribution, unknown institutions, negated mentions, and degeneracy be scored independently?

## Reproducibility

Run from repository root. Keep checkpoint revision, device/dtype, task wording, generation settings, hook position, dose, and decode-window definitions with each result. Commit JSON/CSV alongside experiment code if repository size/policy allows. Never claim that CPU-only unit tests constitute a model run.

```bash
python experiments/013-qwen-semantic-specificity/run.py --self-count 1 --provenance-count 2 --reasoning-count 1
python experiments/013-qwen-semantic-specificity/run.py
```

Outputs: `results/experiment-013-semantic-specificity.json` and `.csv`.
