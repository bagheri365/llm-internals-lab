"""Localize a Qwen3-4B MLP activation intervention across generation stages."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = ("Who are you?", "Who developed you?", "What company created you?", "Tell me about yourself.")
MODES = ("none", "all", "prefill", "last_prompt", "decode")


def conditions(prompts=PROMPTS, modes=MODES):
    return [(prompt, mode) for prompt in prompts for mode in modes]


def steer_activation(x, neuron, dose, mode):
    """Return an edited copy of [batch, sequence, mlp_width] activations.

    With KV caching, prefill typically has seq_len > 1 and decoding seq_len == 1.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "none" or dose == 0:
        return x
    is_prefill = x.shape[-2] > 1
    if mode == "prefill" and not is_prefill:
        return x
    if mode == "last_prompt" and not is_prefill:
        return x
    if mode == "decode" and is_prefill:
        return x
    edited = x.clone()
    if mode == "last_prompt":
        edited[..., -1, neuron] += dose
    else:
        edited[..., neuron] += dose
    return edited


def input_ids_from_template(tokenizer, prompt, device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return encoded.to(device)


def save_results(output, record):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["prompt", "mode", "dose", "response"])
        writer.writeheader()
        writer.writerows(record["results"])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=None, help="Optional exact Hugging Face commit SHA")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--dose", type=float, default=-98)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-003-localization.json"))
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError(f"Layer {args.layer} is out of range")
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        raise ValueError(f"Neuron {args.neuron} is out of range")

    rows = []
    for prompt, mode in conditions():
        ids = input_ids_from_template(tokenizer, prompt, device)
        if ids.shape[-1] <= 1:
            raise ValueError("Expected a multi-token prompt to distinguish prefill from decoding")
        calls = {"prefill": 0, "decode": 0}

        def hook(_module, inputs):
            stage = "prefill" if inputs[0].shape[-2] > 1 else "decode"
            calls[stage] += 1
            return (steer_activation(inputs[0], args.neuron, args.dose, mode),) + inputs[1:]

        handle = target.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                output = model.generate(
                    ids, do_sample=False, use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True)
        finally:
            handle.remove()
        rows.append({"prompt": prompt, "mode": mode, "dose": 0 if mode == "none" else args.dose, "response": response})
        print(f"{prompt!r} | {mode} | calls={calls}: {response}", flush=True)
        if calls["prefill"] != 1:
            raise RuntimeError(f"Unexpected prefill call count: {calls}; inspect cache behavior")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "requested_revision": args.revision,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device, "dtype": str(dtype),
        "layer": args.layer, "neuron": args.neuron,
        "dose": args.dose, "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False, "decoding": "greedy", "use_cache": True,
        "intervention": "additive pre-hook on MLP down_proj input; prefill/decode inferred from sequence length",
        "results": rows,
    }
    csv_path = save_results(args.output, record)
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
