"""Experiment 007: factorial fixed-prefix and cached-history interventions."""
import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPT = "What company created you?"
HELPER_PATH = Path(__file__).resolve().parents[1] / "006-qwen-logit-causality" / "run.py"


def conditions():
    return [(prefix, history, edit7)
            for prefix in ("baseline", "first6")
            for history in ("clean", "edited")
            for edit7 in (False, True)]


def steps_for(history, edit7):
    if history not in ("clean", "edited"):
        raise ValueError(f"Unknown history: {history}")
    return (set(range(1, 7)) if history == "edited" else set()) | ({7} if edit7 else set())


def pairwise_effects(rows):
    """Step-7 margin shifts within fixed-prefix, fixed-history pairs."""
    lookup = {(r["prefix"], r["history"], r["step7_edited"]): r for r in rows}
    effects = []
    for prefix in ("baseline", "first6"):
        for history in ("clean", "edited"):
            off = lookup[(prefix, history, False)]["open_minus_alibaba_logit"]
            on = lookup[(prefix, history, True)]["open_minus_alibaba_logit"]
            effects.append({"prefix": prefix, "history": history,
                            "step7_margin_shift": on - off})
    return effects


def load_helper():
    if not HELPER_PATH.is_file():
        raise FileNotFoundError("Experiment 006 is required: " + str(HELPER_PATH))
    spec = importlib.util.spec_from_file_location("exp006_helper", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["prefix", "history", "step7_edited", "greedy_token_text",
              "open_probability", "alibaba_probability", "open_minus_alibaba_logit", "edited_steps"]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in record["results"]:
            writer.writerow({"prefix": r["prefix"], "history": r["history"],
                             "step7_edited": r["step7_edited"],
                             "greedy_token_text": r["greedy_token_text"],
                             "open_probability": r["targets"]["open"]["probability"],
                             "alibaba_probability": r["targets"]["alibaba"]["probability"],
                             "open_minus_alibaba_logit": r["open_minus_alibaba_logit"],
                             "edited_steps": ";".join(map(str, r["edited_steps"]))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=SHA)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--neuron", type=int, default=386)
    parser.add_argument("--dose", type=float, default=-98.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("results/experiment-007-cache-disentanglement.json"))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    exp6 = load_helper()
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
    ids = exp6.chat_ids(tokenizer, args.prompt, device)
    target_ids = {"alibaba": exp6.choose_token_id(tokenizer, " Alibaba"),
                  "open": exp6.choose_token_id(tokenizer, " Open")}
    prefixes = {}
    for name, steps in (("baseline", set()), ("first6", set(range(1, 7)))):
        trial = exp6.run_prefix(model, ids, target, args.neuron, args.dose, steps)
        prefixes[name] = trial["prefix_ids"]
        print(f"{name} prefix: {tokenizer.decode(prefixes[name])!r}", flush=True)
    if prefixes["baseline"] == prefixes["first6"]:
        print("WARNING: naturally generated prefixes are identical; text-prefix comparison is uninformative", flush=True)

    rows = []
    for prefix, history, edit7 in conditions():
        steps = steps_for(history, edit7)
        trial = exp6.run_prefix(model, ids, target, args.neuron, args.dose,
                                steps, forced=prefixes[prefix])
        row = {"prefix": prefix, "prefix_ids": prefixes[prefix],
               "prefix_text": tokenizer.decode(prefixes[prefix]),
               "history": history, "step7_edited": edit7,
               "edited_steps": trial["edited_steps"],
               **exp6.summarize(trial["logits"], tokenizer, target_ids, args.top_k)}
        rows.append(row)
        print(f"{prefix:>8} | {history:>6} | step7={edit7} | "
              f"next={row['greedy_token_text']!r} | "
              f"margin={row['open_minus_alibaba_logit']:+.3f}", flush=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "requested_revision": args.revision,
              "loaded_config_commit": getattr(model.config, "_commit_hash", None),
              "torch_version": torch.__version__,
              "transformers_version": transformers.__version__,
              "device": device, "dtype": str(dtype), "prompt": args.prompt,
              "layer": args.layer, "neuron": args.neuron, "dose": args.dose,
              "target_token_ids": target_ids, "prefixes": prefixes,
              "method": "2x2x2 teacher-forced prefix, prior intervention history, step7 intervention",
              "results": rows, "step7_effects": pairwise_effects(rows)}
    save(args.output, record)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
