"""Dose sweep and neighboring-neuron controls for Qwen3-4B.

Run from the repository root:
  python experiments/001-qwen-identity-steering/sweep.py
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def conditions(neuron=386):
    doses = [0, -10, -30, -60, -80, -90, -98, -110, 98]
    return [(neuron, dose) for dose in doses] + [(neuron - 1, -98), (neuron + 1, -98)]


def input_ids_from_template(tokenizer, prompt, device):
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    if hasattr(formatted, "input_ids"):
        formatted = formatted.input_ids
    return formatted.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--prompt", default="Who are you?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-001-sweep.json"))
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError(f"Layer {args.layer} is out of range")
    target = model.model.layers[args.layer].mlp.down_proj
    width = target.in_features
    ids = input_ids_from_template(tokenizer, args.prompt, device)
    rows = []

    for neuron, dose in conditions(args.neuron):
        if not 0 <= neuron < width:
            raise ValueError(f"Neuron {neuron} is out of range (width={width})")

        def hook(_module, inputs):
            activation = inputs[0].clone()
            activation[..., neuron] += dose
            return (activation,) + inputs[1:]

        handle = target.register_forward_pre_hook(hook) if dose else None
        try:
            with torch.inference_mode():
                output = model.generate(
                    ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()
        print(f"neuron={neuron} dose={dose}: {response}", flush=True)
        rows.append({"neuron": neuron, "dose": dose, "response": response})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device,
        "dtype": str(dtype),
        "layer": args.layer,
        "prompt": args.prompt,
        "intervention": "additive down_proj input, all positions, zero-based indices",
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "results": rows,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["neuron", "dose", "response"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
