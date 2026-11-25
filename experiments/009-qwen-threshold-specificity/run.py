"""Experiment 009: fine dose grid and adjacent-neuron specificity at decode step 7."""
import argparse
import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPT = "What company created you?"
HELPER = Path(__file__).resolve().parents[1] / "006-qwen-logit-causality" / "run.py"
DEFAULT_DOSES = "0,-60,-62,-64,-66,-68,-70,-72,-80,-98,-110,-120,-130,-140,-150"
DEFAULT_NEURONS = "385,386,387"
ORGANIZATIONS = (" Alibaba", " Open", " Microsoft", " Meta", " Google", " Azure")


def parse_doses(raw):
    values = [float(x.strip()) for x in raw.split(",")]
    if not values or not all(math.isfinite(x) and x <= 0 for x in values):
        raise ValueError("Doses must be finite and nonpositive")
    if len(set(values)) != len(values) or 0.0 not in values:
        raise ValueError("Doses must be unique and include 0")
    return sorted(values, reverse=True)


def parse_neurons(raw):
    values = [int(x.strip()) for x in raw.split(",")]
    if not values or len(values) != len(set(values)) or any(x < 0 for x in values):
        raise ValueError("Neurons must be unique nonnegative integers")
    return values


def first_greedy(rows, prefix, neuron, target_id):
    matches = [abs(r["dose"]) for r in rows if r["prefix"] == prefix
               and r["neuron"] == neuron and r["greedy_token_id"] == target_id]
    return min(matches) if matches else None


def first_margin_crossing(rows, prefix, neuron):
    matches = [abs(r["dose"]) for r in rows if r["prefix"] == prefix
               and r["neuron"] == neuron and r["open_minus_alibaba_logit"] > 0]
    return min(matches) if matches else None


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = ["prefix", "neuron", "dose", "greedy_token_text", "greedy_token_id",
               "open_minus_alibaba_logit", "alibaba_probability", "open_probability",
               "microsoft_probability", "meta_probability", "google_probability",
               "azure_probability", "edited_steps"]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in record["results"]:
            data = {k: row[k] for k in ("prefix", "neuron", "dose", "greedy_token_text",
                                       "greedy_token_id", "open_minus_alibaba_logit")}
            data.update({f"{name}_probability": val["probability"]
                         for name, val in row["targets"].items()})
            data["edited_steps"] = ";".join(map(str, row["edited_steps"]))
            writer.writerow(data)


def load_helper():
    if not HELPER.is_file():
        raise FileNotFoundError(f"Experiment 006 required: {HELPER}")
    spec = importlib.util.spec_from_file_location("exp006_specificity_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=SHA)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--source-neuron", type=int, default=386)
    parser.add_argument("--prefix-source-dose", type=float, default=-98.0)
    parser.add_argument("--neurons", default=DEFAULT_NEURONS)
    parser.add_argument("--doses", default=DEFAULT_DOSES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("results/experiment-009-threshold-specificity.json"))
    args = parser.parse_args()
    try:
        doses, neurons = parse_doses(args.doses), parse_neurons(args.neurons)
    except ValueError as exc:
        parser.error(str(exc))
    if args.top_k < 1 or not math.isfinite(args.prefix_source_dose):
        parser.error("--top-k must be positive and --prefix-source-dose finite")

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
    if any(n >= target.in_features for n in neurons) or not 0 <= args.source_neuron < target.in_features:
        raise ValueError("Neuron out of range")
    ids = exp6.chat_ids(tokenizer, args.prompt, device)
    token_ids = {name.strip().lower(): exp6.choose_token_id(tokenizer, name)
                 for name in ORGANIZATIONS}
    prefixes = {}
    for kind, steps in (("baseline", set()), ("first6", set(range(1, 7)))):
        rollout = exp6.run_prefix(model, ids, target, args.source_neuron,
                                  args.prefix_source_dose, steps)
        prefixes[kind] = rollout["prefix_ids"]
        print(f"{kind} prefix: {tokenizer.decode(prefixes[kind])!r}", flush=True)
    if prefixes["baseline"] == prefixes["first6"]:
        raise RuntimeError("Prefix discovery did not yield distinct token sequences")

    rows = []
    for kind, prefix in prefixes.items():
        for neuron in neurons:
            for dose in doses:
                steps = {7} if dose != 0 else set()
                trial = exp6.run_prefix(model, ids, target, neuron, dose,
                                        steps, forced=prefix)
                summary = exp6.summarize(trial["logits"], tokenizer, token_ids, args.top_k)
                row = {"prefix": kind, "prefix_ids": prefix,
                       "prefix_text": tokenizer.decode(prefix), "history": "clean",
                       "neuron": neuron, "dose": dose,
                       "edited_steps": trial["edited_steps"], **summary}
                rows.append(row)
                print(f"{kind:>8} n={neuron} dose={dose:>6g} "
                      f"greedy={row['greedy_token_text']!r} "
                      f"margin={row['open_minus_alibaba_logit']:+.3f}", flush=True)

    thresholds = {kind: {str(neuron): {
        "first_open_greedy_magnitude_on_grid": first_greedy(rows, kind, neuron, token_ids["open"]),
        "first_positive_open_minus_alibaba_margin_magnitude_on_grid": first_margin_crossing(rows, kind, neuron),
    } for neuron in neurons} for kind in prefixes}
    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "requested_revision": args.revision,
              "loaded_config_commit": getattr(model.config, "_commit_hash", None),
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": device, "dtype": str(dtype), "prompt": args.prompt,
              "layer": args.layer, "source_neuron": args.source_neuron,
              "prefix_source_dose": args.prefix_source_dose,
              "neurons": neurons, "doses": doses, "target_token_ids": token_ids,
              "prefixes": prefixes,
              "method": "teacher-forced fixed seven-token prefixes; clean steps 1-6; one neuron edited at decode step 7",
              "note": "Thresholds are grid observations, not continuous estimates. Single-token organization candidates do not exhaust the vocabulary.",
              "results": rows, "thresholds": thresholds}
    save(args.output, record)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
