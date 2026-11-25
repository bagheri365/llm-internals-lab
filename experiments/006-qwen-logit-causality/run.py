"""Matched-prefix causal comparison of the next developer-name token."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPT = "What company created you?"


def intervention_steps(prefix_kind, edit_seventh):
    if prefix_kind not in ("baseline", "first6"):
        raise ValueError(prefix_kind)
    return set(range(1, 7)) if prefix_kind == "first6" else set()


def replay_steps(prefix_kind, edit_seventh):
    return intervention_steps(prefix_kind, edit_seventh) | ({7} if edit_seventh else set())


def target_metrics(logits, target_ids):
    """Pure-Python metrics from a 1D sequence of float logits."""
    import math
    maximum = max(logits)
    denom = sum(math.exp(v - maximum) for v in logits)
    return {name: {"token_id": idx, "logit": float(logits[idx]),
                   "probability": math.exp(logits[idx] - maximum) / denom}
            for name, idx in target_ids.items()}


def choose_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected a single token for {text!r}, got {ids}")
    return ids[0]


def chat_ids(tokenizer, prompt, device):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return ids.to(device)


def run_prefix(model, ids, target, neuron, dose, steps, *, forced=None, n_decode=7):
    """Generate or replay exactly seven cached steps. Return next-token logits.

    When forced is provided, its seven tokens are fed as inputs to the seven
    cached passes. The prompt prefill is never modified. Pass k predicts token
    k+1. For natural rollout, each pass uses the prior greedy token.
    """
    import torch
    phase = {"step": 0}
    edited = []

    def hook(_module, inputs):
        k = phase["step"]
        if k == 0 or k not in steps:
            return None
        edited.append(k)
        x = inputs[0].clone()
        x[..., neuron] += dose
        return (x,) + inputs[1:]

    handle = target.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            first = int(out.logits[0, -1].argmax().item())
            tokens = [first]
            if forced is not None and (len(forced) != n_decode or forced[0] != first):
                raise ValueError("Forced prefix must have n_decode tokens and match unmodified prefill")
            for k in range(1, n_decode + 1):
                phase["step"] = k
                token = forced[k - 1] if forced is not None else tokens[k - 1]
                out = model(
                    input_ids=torch.tensor([[token]], device=ids.device),
                    past_key_values=cache,
                    cache_position=torch.tensor([ids.shape[-1] + k - 1], device=ids.device),
                    use_cache=True)
                cache = out.past_key_values
                if k < n_decode and forced is None:
                    tokens.append(int(out.logits[0, -1].argmax().item()))
            values = out.logits[0, -1].float().cpu().tolist()
    finally:
        handle.remove()
    if edited != sorted(steps):
        raise RuntimeError(f"Edited unexpected steps: {edited} vs {sorted(steps)}")
    return {"prefix_ids": list(forced) if forced is not None else tokens,
            "edited_steps": edited, "logits": values}


def summarize(logits, tokenizer, target_ids, top_k):
    import math
    target = target_metrics(logits, target_ids)
    ranking = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:top_k]
    maximum = max(logits)
    denominator = sum(math.exp(v - maximum) for v in logits)
    return {
        "targets": target,
        "open_minus_alibaba_logit": target["open"]["logit"] - target["alibaba"]["logit"],
        "top_tokens": [{"id": i, "text": tokenizer.decode([i], skip_special_tokens=False),
                        "logit": logits[i], "probability": math.exp(logits[i] - maximum) / denominator}
                       for i in ranking],
        "greedy_token_id": ranking[0],
        "greedy_token_text": tokenizer.decode([ranking[0]], skip_special_tokens=False),
    }


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["prefix", "step7_edited", "greedy_token_text", "open_probability",
                  "alibaba_probability", "open_minus_alibaba_logit", "edited_steps"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in record["results"]:
            writer.writerow({"prefix": r["prefix"], "step7_edited": r["step7_edited"],
                             "greedy_token_text": r["greedy_token_text"],
                             "open_probability": r["targets"]["open"]["probability"],
                             "alibaba_probability": r["targets"]["alibaba"]["probability"],
                             "open_minus_alibaba_logit": r["open_minus_alibaba_logit"],
                             "edited_steps": ";".join(map(str, r["edited_steps"]))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision", default=DEFAULT_SHA)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--neuron", type=int, default=386)
    p.add_argument("--dose", type=float, default=-98.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results/experiment-006-logit-causality.json"))
    args = p.parse_args()
    if args.top_k < 1:
        p.error("--top-k must be positive")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {args.model}@{args.revision} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager").to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError("Layer out of range")
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        raise ValueError("Neuron out of range")
    ids = chat_ids(tokenizer, args.prompt, device)
    target_ids = {"alibaba": choose_token_id(tokenizer, " Alibaba"),
                  "open": choose_token_id(tokenizer, " Open")}

    prefixes = {}
    for kind in ("baseline", "first6"):
        rollout = run_prefix(model, ids, target, args.neuron, args.dose,
                             intervention_steps(kind, False))
        prefixes[kind] = rollout["prefix_ids"]
        print(f"{kind} prefix: {tokenizer.decode(prefixes[kind])!r}", flush=True)

    rows = []
    for kind in ("baseline", "first6"):
        for edit7 in (False, True):
            trial = run_prefix(model, ids, target, args.neuron, args.dose,
                               replay_steps(kind, edit7), forced=prefixes[kind])
            row = {"prefix": kind, "prefix_ids": prefixes[kind],
                   "prefix_text": tokenizer.decode(prefixes[kind]),
                   "step7_edited": edit7, "edited_steps": trial["edited_steps"],
                   **summarize(trial["logits"], tokenizer, target_ids, args.top_k)}
            rows.append(row)
            print(f"{kind:>8} | step7={edit7} | next={row['greedy_token_text']!r} | "
                  f"Open−Alibaba={row['open_minus_alibaba_logit']:.3f}", flush=True)

    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "requested_revision": args.revision,
              "loaded_config_commit": getattr(model.config, "_commit_hash", None),
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": device, "dtype": str(dtype), "prompt": args.prompt,
              "layer": args.layer, "neuron": args.neuron, "dose": args.dose,
              "target_token_ids": target_ids,
              "method": "fixed 7-token prefix, replay cached steps 1..7, compare step7 off/on",
              "note": "Token probabilities are softmax of full float32-cast logits; prefixes are fixed within each pair.",
              "results": rows}
    save(args.output, record)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
