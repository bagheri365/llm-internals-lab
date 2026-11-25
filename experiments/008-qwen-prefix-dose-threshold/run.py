"""Experiment 008: clean-history fixed-prefix dose-response at decode step 7."""
import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPT = "What company created you?"
HELPER = Path(__file__).resolve().parents[1] / "006-qwen-logit-causality" / "run.py"
DEFAULT_DOSES = "0,-20,-40,-60,-70,-80,-90,-98,-105,-110,-120"


def parse_doses(raw):
    values = [float(part.strip()) for part in raw.split(",")]
    if not values or any(not __import__("math").isfinite(v) or v > 0 for v in values):
        raise ValueError("Doses must be finite, nonpositive comma-separated numbers")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate doses are not allowed")
    if 0.0 not in values:
        raise ValueError("Include dose 0 as the no-edit control")
    return sorted(values, reverse=True)


def first_flip(rows, prefix):
    """Smallest tested absolute dose that yields Open as greedy token; not a continuous threshold."""
    eligible = [r for r in rows if r["prefix"] == prefix and
                r["greedy_token_id"] == r["target_token_ids"]["open"]]
    return min((abs(r["dose"]) for r in eligible), default=None)


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["prefix", "dose", "greedy_token_text", "open_probability",
              "alibaba_probability", "open_minus_alibaba_logit", "edited_steps"]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in record["results"]:
            writer.writerow({"prefix": r["prefix"], "dose": r["dose"],
                             "greedy_token_text": r["greedy_token_text"],
                             "open_probability": r["targets"]["open"]["probability"],
                             "alibaba_probability": r["targets"]["alibaba"]["probability"],
                             "open_minus_alibaba_logit": r["open_minus_alibaba_logit"],
                             "edited_steps": ";".join(map(str, r["edited_steps"]))})


def load_helper():
    if not HELPER.is_file():
        raise FileNotFoundError(f"Experiment 006 required: {HELPER}")
    spec = importlib.util.spec_from_file_location("exp006_dose_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision", default=SHA)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--neuron", type=int, default=386)
    p.add_argument("--prefix-source-dose", type=float, default=-98.0,
                   help="Dose used only to discover the first6-generated text prefix")
    p.add_argument("--doses", default=DEFAULT_DOSES)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output", type=Path,
                   default=Path("results/experiment-008-prefix-dose-threshold.json"))
    args = p.parse_args()
    if args.top_k < 1:
        p.error("--top-k must be positive")
    try:
        doses = parse_doses(args.doses)
    except ValueError as error:
        p.error(str(error))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    exp6 = load_helper()
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
    ids = exp6.chat_ids(tokenizer, args.prompt, device)
    target_ids = {"alibaba": exp6.choose_token_id(tokenizer, " Alibaba"),
                  "open": exp6.choose_token_id(tokenizer, " Open")}
    prefixes = {}
    for kind, steps in (("baseline", set()), ("first6", set(range(1, 7)))):
        trial = exp6.run_prefix(model, ids, target, args.neuron,
                                args.prefix_source_dose, steps)
        prefixes[kind] = trial["prefix_ids"]
        print(f"{kind} prefix: {tokenizer.decode(prefixes[kind])!r}", flush=True)
    if prefixes["baseline"] == prefixes["first6"]:
        raise RuntimeError("Prefixes are identical; cannot test prefix sensitivity")

    rows = []
    for kind in ("baseline", "first6"):
        for dose in doses:
            # Only step 7 is edited. The preceding six cached steps are ALWAYS clean.
            # A zero dose is an unhooked control, rather than a nominal zero-valued edit.
            steps = {7} if dose != 0 else set()
            trial = exp6.run_prefix(model, ids, target, args.neuron, dose,
                                    steps, forced=prefixes[kind])
            row = {"prefix": kind, "prefix_ids": prefixes[kind],
                   "prefix_text": tokenizer.decode(prefixes[kind]),
                   "history": "clean", "dose": dose,
                   "edited_steps": trial["edited_steps"],
                   "target_token_ids": target_ids,
                   **exp6.summarize(trial["logits"], tokenizer, target_ids, args.top_k)}
            rows.append(row)
            print(f"{kind:>8} | dose={dose:>7g} | next={row['greedy_token_text']!r} | "
                  f"margin={row['open_minus_alibaba_logit']:+.4f}", flush=True)

    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "requested_revision": args.revision,
              "loaded_config_commit": getattr(model.config, "_commit_hash", None),
              "torch_version": torch.__version__,
              "transformers_version": transformers.__version__,
              "device": device, "dtype": str(dtype), "prompt": args.prompt,
              "layer": args.layer, "neuron": args.neuron,
              "prefix_source_dose": args.prefix_source_dose,
              "target_token_ids": target_ids, "prefixes": prefixes,
              "doses": doses,
              "method": "fixed seven-token prefix, clean steps 1..6, step7 dose sweep",
              "note": "First flip is a tested-grid observation, not a continuous threshold; nonmonotonicity is possible.",
              "results": rows,
              "first_open_greedy_magnitude_on_grid": {
                  kind: first_flip(rows, kind) for kind in ("baseline", "first6")}}
    save(args.output, record)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
