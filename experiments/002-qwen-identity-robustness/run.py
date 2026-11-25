"""Multi-prompt robustness test for a single Qwen3-4B MLP activation edit.

Run from repository root. Requires torch and transformers.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = [
    "Who are you?",
    "Who developed you?",
    "What company created you?",
    "What is your model name?",
    "Tell me about yourself.",
    "What is 17 plus 25?",  # Non-identity control
]
DOSES = (0, -98, -110)


def conditions(prompts=PROMPTS, doses=DOSES):
    return [(prompt, dose) for prompt in prompts for dose in doses]


def input_ids_from_template(tokenizer, prompt, device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt",
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
        writer = csv.DictWriter(stream, fieldnames=["prompt", "dose", "response"])
        writer.writeheader()
        writer.writerows(record["results"])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=None, help="Optional pinned HF commit")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-002-robustness.json"))
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
    for prompt, dose in conditions():
        ids = input_ids_from_template(tokenizer, prompt, device)

        def hook(_module, inputs):
            activation = inputs[0].clone()
            activation[..., args.neuron] += dose
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
        rows.append({"prompt": prompt, "dose": dose, "response": response})
        print(f"prompt={prompt!r} dose={dose}: {response}", flush=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "requested_revision": args.revision,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device,
        "dtype": str(dtype),
        "layer": args.layer,
        "neuron": args.neuron,
        "intervention": "additive down_proj input, all positions, zero-based indices",
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
        "decoding": "greedy",
        "results": rows,
    }
    csv_path = save_results(args.output, record)
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
