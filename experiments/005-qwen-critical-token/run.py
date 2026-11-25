"""Locate early decoding steps that redirect a model's claimed developer."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROMPT = "What company created you?"


def conditions():
    return [("none", ()), *[(f"first{k}", tuple(range(1, k + 1))) for k in range(2, 11)],
            *[(f"step{k}", (k,)) for k in range(1, 11)], ("all", None)]


def should_edit(steps, step):
    return steps is None or step in steps


def first_divergence(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def chat_ids(tokenizer, prompt, device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return encoded.to(device)


def generate(model, tokenizer, ids, target, neuron, dose, steps, max_new_tokens):
    """Manual greedy loop: prefill predicts token 1; decode step k predicts token k+1."""
    import torch
    generated = []
    edited = []
    calls = {"prefill": 0, "decode": 0}
    phase = {"decode_step": 0}

    def hook(_module, inputs):
        x = inputs[0]
        if phase["decode_step"] == 0:
            calls["prefill"] += 1
            return None
        step = phase["decode_step"]
        calls["decode"] += 1
        if not should_edit(steps, step):
            return None
        edited.append(step)
        x = x.clone()
        x[..., neuron] += dose
        return (x,) + inputs[1:]

    handle = target.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            for token_index in range(max_new_tokens):
                token = int(out.logits[:, -1, :].argmax(dim=-1).item())
                generated.append(token)
                if token in tokenizer.all_special_ids and token == tokenizer.eos_token_id:
                    break
                if token_index + 1 >= max_new_tokens:
                    break
                phase["decode_step"] = token_index + 1
                # Explicit cache_position avoids ambiguity across transformers releases.
                pos = ids.shape[-1] + token_index
                next_ids = torch.tensor([[token]], device=ids.device)
                out = model(input_ids=next_ids, past_key_values=cache,
                            cache_position=torch.tensor([pos], device=ids.device),
                            use_cache=True)
                cache = out.past_key_values
    finally:
        handle.remove()
    if calls["prefill"] != 1 or calls["decode"] != len(generated) - 1:
        raise RuntimeError(f"Unexpected forward counts: {calls}, tokens={len(generated)}")
    return {"token_ids": generated,
            "tokens": [tokenizer.decode([t], skip_special_tokens=False) for t in generated],
            "response": tokenizer.decode(generated, skip_special_tokens=True),
            "edited_steps": edited, "prefill_calls": calls["prefill"],
            "decode_calls": calls["decode"]}


def save(output, record):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["mode", "edited_steps", "first_divergence_token", "response"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in record["results"]:
            writer.writerow({"mode": row["mode"], "edited_steps": ";".join(map(str, row["edited_steps"])),
                             "first_divergence_token": row["first_divergence_token"],
                             "response": row["response"]})
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=None, help="Full HF commit SHA, if known")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--dose", type=float, default=-98)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-005-critical-token.json"))
    args = parser.parse_args()
    if args.max_new_tokens < 12:
        parser.error("--max-new-tokens must be at least 12")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})", flush=True)
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
    rows = []
    baseline = None
    for mode, steps in conditions():
        row = generate(model, tokenizer, ids, target, args.neuron, args.dose,
                       steps if mode != "none" else (), args.max_new_tokens)
        row["mode"] = mode
        if baseline is None:
            baseline = row["token_ids"]
        row["first_divergence_token"] = first_divergence(baseline, row["token_ids"])
        rows.append(row)
        print(f"{mode:>8} | first divergence={row['first_divergence_token']} | {row['response']}", flush=True)

    # An unpinned revision may not expose its commit in transformers config.
    # Query the Hub only if necessary; failure must not invalidate the experiment.
    commit = args.revision or getattr(model.config, "_commit_hash", None)
    if not commit:
        try:
            from huggingface_hub import model_info
            commit = model_info(args.model).sha
        except Exception as exc:
            print(f"Warning: could not resolve Hub revision: {exc}", flush=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": args.model,
              "requested_revision": args.revision, "model_commit": commit,
              "revision_note": "Hub SHA resolved after loading unless --revision pins weights",
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": device, "dtype": str(dtype), "prompt": args.prompt,
              "layer": args.layer, "neuron": args.neuron, "dose": args.dose,
              "max_new_tokens": args.max_new_tokens, "decoding": "manual greedy, cache enabled",
              "step_definition": "step k processes generated token k and predicts token k+1; prefill never edited",
              "first_divergence_definition": "zero-based generated token index; null means identical",
              "results": rows}
    csv_path = save(args.output, record)
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
