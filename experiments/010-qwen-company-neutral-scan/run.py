"""Experiment 010: company-neutral layer-24 neuron discovery and free-generation validation."""
import argparse
import csv
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
HELPER = Path(__file__).resolve().parents[1] / "006-qwen-logit-causality" / "run.py"
PROMPT = "What company created you?"
DEFAULT_DOSES = "-68,-98"
COMPANIES = {
    "Alibaba": r"\b(?:alibaba|alibaba cloud|tongyi|qwen)\b",
    "Google": r"\b(?:google|deepmind)\b",
    "Microsoft": r"\b(?:microsoft)\b",
    "OpenAI": r"\b(?:openai)\b",
    "Meta": r"\b(?:meta|facebook)\b",
    "Anthropic": r"\b(?:anthropic)\b",
    "Amazon": r"\b(?:amazon|aws)\b",
    "Apple": r"\b(?:apple)\b",
}


def parse_neurons(raw, width):
    if not raw:
        raise ValueError("Empty neuron selection")
    result = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise ValueError("Reversed neuron range")
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    if len(set(result)) != len(result) or any(not 0 <= n < width for n in result):
        raise ValueError("Neuron indices must be unique and within down_proj input width")
    return result


def parse_doses(raw):
    values = [float(p.strip()) for p in raw.split(",")]
    if not values or len(set(values)) != len(values) or any(not math.isfinite(d) or d == 0 for d in values):
        raise ValueError("Doses must be distinct, finite, nonzero values")
    return values


def claim(text):
    matches = [name for name, pattern in COMPANIES.items() if re.search(pattern, text, re.I)]
    return matches if matches else ["unclassified"]


def distribution_metrics(baseline, altered):
    """Full-vocabulary total variation and Jensen-Shannon divergence (natural logs)."""
    if len(baseline) != len(altered) or not baseline:
        raise ValueError("Logit vectors must be nonempty and equal length")
    def probs(xs):
        mx = max(xs)
        es = [math.exp(v - mx) for v in xs]
        total = sum(es)
        return [e / total for e in es]
    p, q = probs(baseline), probs(altered)
    tv = sum(abs(a - b) for a, b in zip(p, q)) / 2
    js = 0.0
    for a, b in zip(p, q):
        m = (a + b) / 2
        if a:
            js += a * math.log(a / m) / 2
        if b:
            js += b * math.log(b / m) / 2
    return {"total_variation": tv, "jensen_shannon_nats": js}


def top_tokens(logits, tokenizer, count):
    order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:count]
    return [{"id": i, "text": tokenizer.decode([i], skip_special_tokens=False), "logit": logits[i]} for i in order]


def load_helper():
    if not HELPER.exists():
        raise FileNotFoundError(f"Experiment 006 helper is required: {HELPER}")
    spec = importlib.util.spec_from_file_location("exp006_neutral_helper", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_claim(model, tokenizer, ids, target, neuron, dose, max_new_tokens):
    """Free greedy rollout: apply intervention to decode calls only (not prefill)."""
    import torch
    state = {"prefill_seen": False}
    def hook(_module, inputs):
        if not state["prefill_seen"]:
            state["prefill_seen"] = True
            return None
        x = inputs[0].clone()
        x[..., neuron] += dose
        return (x,) + inputs[1:]
    handle = target.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
    finally:
        handle.remove()
    response = tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True)
    return {"response": response, "company_mentions": claim(response)}


def write_results(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["prefix", "neuron", "dose", "total_variation", "jensen_shannon_nats",
              "greedy_token_text", "baseline_greedy_token_text", "greedy_changed"]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in record["discovery"]:
            writer.writerow({k: row[k] for k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=SHA)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neurons", default="350-420")
    parser.add_argument("--doses", default=DEFAULT_DOSES)
    parser.add_argument("--prefix", choices=["baseline", "assistant", "both"], default="both")
    parser.add_argument("--source-neuron", type=int, default=386)
    parser.add_argument("--source-dose", type=float, default=-98)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--validate-top", type=int, default=5,
                        help="Number of strongest neurons to validate with free generation (0 to skip)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-010-company-neutral-scan.json"))
    args = parser.parse_args()
    try:
        doses = parse_doses(args.doses)
    except ValueError as exc:
        parser.error(str(exc))
    if args.top_k < 1 or args.validate_top < 0 or args.max_new_tokens < 1 or not math.isfinite(args.source_dose):
        parser.error("Invalid top-k, validation count, max tokens, or source dose")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    helper = load_helper()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager").to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError("Layer out of range")
    target = model.model.layers[args.layer].mlp.down_proj
    neurons = parse_neurons(args.neurons, target.in_features)
    if not 0 <= args.source_neuron < target.in_features:
        raise ValueError("Source neuron out of range")
    ids = helper.chat_ids(tokenizer, args.prompt, device)
    prefixes = {"baseline": helper.run_prefix(model, ids, target, args.source_neuron, 0, set())["prefix_ids"]}
    if args.prefix in ("assistant", "both"):
        prefixes["assistant"] = helper.run_prefix(model, ids, target, args.source_neuron,
                                                   args.source_dose, set(range(1, 7)))["prefix_ids"]
    if args.prefix == "assistant":
        prefixes.pop("baseline")
    print("Fixed prefixes:", {k: tokenizer.decode(v) for k, v in prefixes.items()}, flush=True)
    discovery = []
    for name, prefix in prefixes.items():
        clean = helper.run_prefix(model, ids, target, neurons[0], 0, set(), forced=prefix)["logits"]
        base_top = top_tokens(clean, tokenizer, args.top_k)
        for neuron in neurons:
            for dose in doses:
                trial = helper.run_prefix(model, ids, target, neuron, dose, {7}, forced=prefix)
                logits = trial["logits"]
                metrics = distribution_metrics(clean, logits)
                top = top_tokens(logits, tokenizer, args.top_k)
                row = {"prefix": name, "neuron": neuron, "dose": dose, **metrics,
                       "greedy_token_id": top[0]["id"], "greedy_token_text": top[0]["text"],
                       "baseline_greedy_token_id": base_top[0]["id"],
                       "baseline_greedy_token_text": base_top[0]["text"],
                       "greedy_changed": top[0]["id"] != base_top[0]["id"],
                       "top_tokens": top}
                discovery.append(row)
                print(f"{name} neuron={neuron} dose={dose:g} TV={metrics['total_variation']:.4f} "
                      f"greedy={top[0]['text']!r}", flush=True)
    # Rank neurons by strongest distribution shift across all prefix/dose conditions.
    scores = {n: max(r["total_variation"] for r in discovery if r["neuron"] == n) for n in neurons}
    ranked = sorted(neurons, key=lambda n: (-scores[n], n))
    validation = []
    if args.validate_top:
        baseline = generate_claim(model, tokenizer, ids, target, neurons[0], 0, args.max_new_tokens)
        validation.append({"neuron": None, "dose": 0, "baseline": True, **baseline})
        for neuron in ranked[:args.validate_top]:
            for dose in doses:
                result = generate_claim(model, tokenizer, ids, target, neuron, dose, args.max_new_tokens)
                validation.append({"neuron": neuron, "dose": dose, "baseline": False, **result})
                print(f"VALIDATE neuron={neuron} dose={dose:g} {result['company_mentions']} "
                      f"{result['response'][:100]!r}", flush=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "revision": args.revision,
              "loaded_config_commit": getattr(model.config, "_commit_hash", None),
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": device, "dtype": str(dtype), "prompt": args.prompt,
              "layer": args.layer, "neurons": neurons, "doses": doses,
              "prefixes": {k: {"ids": v, "text": tokenizer.decode(v)} for k, v in prefixes.items()},
              "method": "fixed-prefix clean-history decode-step-7 full-vocabulary scan; top-ranked free-generation decode-only validation",
              "rank_metric": "maximum total variation across fixed-prefix conditions",
              "ranking": [{"neuron": n, "max_total_variation": scores[n]} for n in ranked],
              "discovery": discovery, "validation": validation,
              "limitations": ["Distribution shift does not necessarily imply developer-claim change.",
                              "Company mentions are heuristic and may be negated or incidental.",
                              "Discovery uses teacher-forced prefixes; validation uses free decoding.",
                              "Validation uses decode-only edits at all generated decode steps, unlike discovery step 7 only.",
                              "One prompt and one checkpoint do not establish generality."]}
    write_results(args.output, record)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
