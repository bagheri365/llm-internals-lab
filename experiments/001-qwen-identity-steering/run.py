"""Reproduce a proposed Qwen3-4B single-MLP-neuron identity intervention."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def add_dose_to_neuron(activation, neuron: int, dose: float):
    """Return a copy of the activation with an additive intervention."""
    if not 0 <= neuron < activation.shape[-1]:
        raise IndexError(f"Neuron {neuron} outside last dimension {activation.shape[-1]}")
    changed = activation.clone()
    changed[..., neuron] += dose
    return changed


def input_ids_from_chat_template(encoded):
    """Normalize chat-template output to a tensor for model.generate."""
    if hasattr(encoded, "shape"):
        return encoded
    return encoded["input_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--dose", type=float, default=-98.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-001.json"))
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager"
    ).to(device).eval()
    target = model.model.layers[args.layer].mlp.down_proj
    if args.neuron >= target.in_features or args.neuron < 0:
        raise ValueError("Neuron index outside down_proj input width")
    prompt = "Who are you?"
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt",
    )
    tokens = input_ids_from_chat_template(tokens).to(device)

    def generate(dose):
        def hook(_module, inputs):
            return (add_dose_to_neuron(inputs[0], args.neuron, dose),) + inputs[1:]

        handle = target.register_forward_pre_hook(hook) if dose else None
        try:
            with torch.inference_mode():
                output = model.generate(
                    tokens, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(output[0, tokens.shape[-1]:], skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()

    observations = []
    for dose in (0.0, args.dose):
        response = generate(dose)
        print(f"\nDose {dose}: {response}")
        observations.append({"dose": dose, "response": response})

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device, "dtype": str(dtype), "prompt": prompt,
        "layer_zero_based": args.layer, "neuron_zero_based": args.neuron,
        "intervention": "additive pre-hook on MLP down_proj input at every forward pass",
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "enable_thinking": False,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
