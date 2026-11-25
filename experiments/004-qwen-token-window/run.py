"""Test which cached decoding steps are sufficient for identity steering."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = ("Who are you?", "Who developed you?", "What company created you?", "Tell me about yourself.")
MODES = ("none", "first1", "first5", "first10", "after10", "all")


def conditions(prompts=PROMPTS, modes=MODES):
    return [(prompt, mode) for prompt in prompts for mode in modes]


def should_edit(mode, decode_step):
    """decode_step is 1-based; step 1 processes the first generated token."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if decode_step < 1:
        raise ValueError("decode_step must be positive")
    return {"none": False, "first1": decode_step == 1,
            "first5": decode_step <= 5, "first10": decode_step <= 10,
            "after10": decode_step > 10, "all": True}[mode]


def edit_activation(x, neuron, dose):
    if not 0 <= neuron < x.shape[-1]:
        raise ValueError("Neuron index out of range")
    edited = x.clone()
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
        writer = csv.DictWriter(stream, fieldnames=["prompt", "mode", "dose", "prefill_calls", "decode_calls", "extra_final_forward", "edited_steps", "response"])
        writer.writeheader()
        for row in record["results"]:
            writer.writerow({**row, "edited_steps": ";".join(map(str, row["edited_steps"]))})
    return csv_path


def validate_decode_calls(calls, generated_count):
    """Some generate() versions run an extra final cached forward pass.

    Such a pass does not produce an additional returned token. Record it,
    but do not treat it as a generation failure.
    """
    if generated_count < 1:
        raise RuntimeError(f"No tokens generated: {generated_count}")
    if calls["prefill"] != 1 or calls["decode"] not in (generated_count - 1, generated_count):
        raise RuntimeError(f"Unexpected forward-pass counts {calls}, generated {generated_count} tokens")
    return calls["decode"] == generated_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=None, help="Optional exact Hugging Face commit SHA")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--dose", type=float, default=-98)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/experiment-004-token-window.json"))
    args = parser.parse_args()
    if args.max_new_tokens < 12:
        parser.error("--max-new-tokens must be at least 12 to test after10")

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
        raise ValueError("Layer out of range")
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        raise ValueError("Neuron out of range")

    rows = []
    for prompt, mode in conditions():
        ids = input_ids_from_template(tokenizer, prompt, device)
        if ids.shape[-1] <= 1:
            raise ValueError("Expected multi-token prompt for cache-based stage detection")
        calls = {"prefill": 0, "decode": 0}
        edited_steps = []

        def hook(_module, inputs):
            x = inputs[0]
            if x.shape[-2] > 1:
                calls["prefill"] += 1
                return None
            calls["decode"] += 1
            step = calls["decode"]
            if should_edit(mode, step):
                edited_steps.append(step)
                return (edit_activation(x, args.neuron, args.dose),) + inputs[1:]
            return None

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
        generated_count = output.shape[-1] - ids.shape[-1]
        extra_final_forward = validate_decode_calls(calls, generated_count)
        row = {"prompt": prompt, "mode": mode, "dose": 0 if mode == "none" else args.dose,
               "prefill_calls": calls["prefill"], "decode_calls": calls["decode"],
               "extra_final_forward": extra_final_forward,
               "edited_steps": edited_steps, "response": response}
        rows.append(row)
        print(f"{prompt!r} | {mode} | edits={len(edited_steps)} | {response}", flush=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "requested_revision": args.revision,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "device": device, "dtype": str(dtype), "layer": args.layer, "neuron": args.neuron,
        "dose": args.dose, "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False, "decoding": "greedy", "use_cache": True,
        "intervention": "down_proj input; cached decode steps only; 1-based steps; prefill never edited",
        "results": rows,
    }
    csv_path = save_results(args.output, record)
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
